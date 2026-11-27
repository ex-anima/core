#include "exanima/inference.h"
#include "training_detail.h"

#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>

namespace {

void require(bool test, const char* message) {
    if (!test) {
        throw std::runtime_error(message);
    }
}

void check_gradients() {
    const exanima::detail::Layout shape{2, 2, 2};
    std::vector<float> parameters{0.4f, 0.2f, -0.3f, 0.65f, 0.1f, 0.25f,
                                  -0.7f, 0.4f, 0.9f, -0.1f, 0.3f, -0.2f};
    const exanima::Demonstration example{{0.3f, 0.9f}, 1};
    std::vector<float> hidden(shape.hidden), output(shape.output), scratch(shape.size());
    const auto loss = [&] {
        return exanima::detail::loss_and_gradient(shape, parameters, example, hidden, output, scratch);
    };
    loss();
    const auto analytic = scratch;
    constexpr float step = 0.001f;
    for (std::size_t i = 0; i < parameters.size(); ++i) {
        const float original = parameters[i];
        parameters[i] = original + step;
        const double above = loss();
        parameters[i] = original - step;
        const double below = loss();
        parameters[i] = original;
        const double numeric = (above - below) / (2 * step);
        require(std::abs(numeric - analytic[i]) < 0.0002, "Gradient differs from finite difference");
    }
}

void requires_load_failure(const std::filesystem::path& directory) {
    try {
        exanima::load_model(directory);
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error("Invalid model accepted");
}

void check_roundtrip() {
    const exanima::ModelSpec spec{{"x", "y"}, 8, {"left", "right"}};
    const std::vector<exanima::Demonstration> samples{
        {{0.1f, 0.2f}, 0}, {{0.2f, 0.4f}, 0}, {{0.8f, 0.1f}, 1}, {{0.9f, 0.25f}, 1}};
    const auto result = exanima::fit(spec, samples);
    require(result.final_loss < result.initial_loss * 0.1, "Training did not reduce loss");
    for (const auto& sample : samples) {
        require(result.model.predict(sample.observation) == sample.label, "Training did not learn the sample");
    }
    const auto directory = std::filesystem::temp_directory_path()
        / ("exanima-roundtrip-" + std::to_string(std::random_device{}()));
    exanima::save_model(result.model, directory);
    const auto model = exanima::load_model(directory);
    require(model.parameters() == result.model.parameters(), "Model parameter roundtrip differs");
    for (const auto& sample : samples) {
        require(model.logits(sample.observation) == result.model.logits(sample.observation),
                "Loaded model predicts different logits");
    }
    const auto binary = directory / "weights.bin";
    {
        std::fstream weights(binary, std::ios::binary | std::ios::in | std::ios::out);
        constexpr std::array<unsigned char, 4> nan{0, 0, 0xc0, 0x7f};
        weights.write(reinterpret_cast<const char*>(nan.data()), nan.size());
    }
    requires_load_failure(directory);
    std::filesystem::resize_file(binary, 0);
    requires_load_failure(directory);
    std::filesystem::remove(binary);
    std::filesystem::remove(directory / "model.json");
    std::filesystem::remove(directory);
}

}  // namespace

int main() {
    try {
        check_gradients();
        check_roundtrip();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
