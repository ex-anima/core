#include "exanima/training.h"
#include "format.h"

#include <algorithm>
#include <fstream>
#include <iostream>

namespace {

int run(const std::filesystem::path& config, const std::filesystem::path& dataset,
        const std::filesystem::path& directory) {
    if (std::filesystem::file_size(config) > 4 * 1024 * 1024) {
        throw std::invalid_argument("Training configuration exceeds the size limit");
    }
    std::ifstream config_file(config, std::ios::binary);
    const auto document = nlohmann::json::parse(config_file);
    const auto spec = exanima::detail::read_spec(document);
    exanima::TrainingOptions options;
    if (document.contains("seed")) {
        options.seed = static_cast<std::uint32_t>(
            exanima::detail::bounded_size(document.at("seed"), std::numeric_limits<std::uint32_t>::max()));
    }
    if (document.contains("epochs")) {
        options.epochs = static_cast<int>(exanima::detail::bounded_size(document.at("epochs"), 10000));
    }
    if (document.contains("learning_rate")) {
        options.learning_rate = exanima::detail::read_number(document.at("learning_rate"));
    }
    if (document.contains("balance_classes")) {
        const auto& value = document.at("balance_classes");
        if (!value.is_boolean()) {
            throw std::invalid_argument("balance_classes must be a boolean");
        }
        options.balance_classes = value.get<bool>();
    }

    std::ifstream stream(dataset, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Cannot open demonstration file");
    }
    std::vector<exanima::Demonstration> examples;
    std::string line;
    while (std::getline(stream, line)) {
        const auto row = nlohmann::json::parse(line);
        const auto label = row.at("decision").get<std::string>();
        const auto index = std::find(spec.labels.begin(), spec.labels.end(), label);
        if (index == spec.labels.end()) {
            throw std::invalid_argument("Demonstration uses an unknown decision");
        }
        examples.push_back({exanima::detail::read_observation(row.at("observation"), spec.input_features.size()),
                            static_cast<std::size_t>(index - spec.labels.begin())});
    }
    if (stream.bad()) {
        throw std::runtime_error("Cannot read demonstrations");
    }
    const auto result = exanima::fit(spec, examples, options);
    exanima::save_model(result.model, directory);
    const nlohmann::json summary = {{"examples", examples.size()}, {"initial_loss", result.initial_loss},
        {"final_loss", result.final_loss}, {"epochs", options.epochs}, {"seed", options.seed}};
    std::cout << summary.dump() << '\n';
    return 0;
}

}  // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
#else
int main(int argc, char** argv) {
#endif
    if (argc != 4) {
        std::cerr << "Usage: exanima-train CONFIG.json DEMONSTRATIONS.jsonl NEW_MODEL_DIRECTORY\n";
        return 1;
    }
    try {
        return run(argv[1], argv[2], argv[3]);
    } catch (const std::exception& error) {
        std::cerr << "Training: " << error.what() << '\n';
        return 1;
    }
}
