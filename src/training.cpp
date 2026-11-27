#include "exanima/training.h"
#include "training_detail.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

namespace exanima::detail {

double loss_and_gradient(Layout shape, const std::vector<float>& parameters,
                         const Demonstration& example, std::vector<float>& hidden,
                         std::vector<float>& output, std::vector<float>& gradient) {
    forward(shape, parameters, example.observation, hidden, output);
    if (!finite(output)) {
        throw std::runtime_error("Training produced nonfinite logits");
    }
    const double largest = *std::max_element(output.begin(), output.end());
    double denominator = 0;
    for (const float score : output) {
        denominator += std::exp(static_cast<double>(score) - largest);
    }
    const double loss = largest - output[example.label] + std::log(denominator);

    for (std::size_t c = 0; c < shape.output; ++c) {
        const double probability = std::exp(static_cast<double>(output[c]) - largest) / denominator;
        const float delta = static_cast<float>(probability - (c == example.label ? 1.0 : 0.0));
        gradient[shape.bias2() + c] = delta;
        for (std::size_t h = 0; h < shape.hidden; ++h) {
            gradient[shape.weights2() + c * shape.hidden + h] = delta * hidden[h];
        }
    }
    for (std::size_t h = 0; h < shape.hidden; ++h) {
        float delta = 0;
        if (hidden[h] > 0) {
            for (std::size_t c = 0; c < shape.output; ++c) {
                delta += parameters[shape.weights2() + c * shape.hidden + h] * gradient[shape.bias2() + c];
            }
        }
        gradient[shape.bias1() + h] = delta;
        for (std::size_t i = 0; i < shape.input; ++i) {
            gradient[h * shape.input + i] = delta * example.observation[i];
        }
    }
    return loss;
}

}  // namespace exanima::detail

namespace exanima {
namespace {

void initialize(std::mt19937& random, std::vector<float>::iterator begin,
                std::vector<float>::iterator end, std::size_t fan_sum) {
    const float radius = std::sqrt(6.0f / static_cast<float>(fan_sum));
    std::uniform_real_distribution<float> distribution(-radius, radius);
    std::generate(begin, end, [&] { return distribution(random); });
}

}  // namespace

TrainingResult fit(const ModelSpec& spec, const std::vector<Demonstration>& examples,
                   const TrainingOptions& options) {
    const auto shape = detail::validate_spec(spec);
    if (examples.empty() || options.epochs <= 0 || options.epochs > 10000
        || !std::isfinite(options.learning_rate) || options.learning_rate <= 0 || options.learning_rate > 1) {
        throw std::invalid_argument("Invalid training examples or options");
    }
    std::vector<std::size_t> counts(shape.output, 0);
    for (const auto& example : examples) {
        if (example.observation.size() != shape.input || !detail::finite(example.observation)
            || example.label >= shape.output) {
            throw std::invalid_argument("Demonstration shape, values or label do not match the model");
        }
        ++counts[example.label];
    }

    std::vector<float> class_scale(shape.output, 1);
    if (options.balance_classes) {
        double normalizer = 0;
        for (auto count : counts) {
            normalizer += std::sqrt(static_cast<double>(count));
        }
        for (std::size_t c = 0; c < shape.output; ++c) {
            if (counts[c] > 0) {
                const double weight = 1.0 / std::sqrt(static_cast<double>(counts[c]));
                class_scale[c] = static_cast<float>(weight * static_cast<double>(examples.size()) / normalizer);
            }
        }
    }

    std::mt19937 random(options.seed);
    std::vector<float> parameters(shape.size(), 0);
    auto address = [&](std::size_t offset) {
        return parameters.begin() + static_cast<std::ptrdiff_t>(offset);
    };
    initialize(random, parameters.begin(), address(shape.bias1()), shape.input + shape.hidden);
    initialize(random, address(shape.weights2()), address(shape.bias2()), shape.hidden + shape.output);

    std::vector<float> hidden(shape.hidden);
    std::vector<float> output(shape.output);
    std::vector<float> gradient(shape.size());
    const auto loss = [&](const Demonstration& sample) {
        return detail::loss_and_gradient(shape, parameters, sample, hidden, output, gradient);
    };
    const auto average_loss = [&] {
        double total = 0;
        for (const auto& sample : examples) {
            total += loss(sample);
        }
        return total / static_cast<double>(examples.size());
    };

    std::vector<std::size_t> order(examples.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    const double initial = average_loss();
    for (int epoch = 0; epoch < options.epochs; ++epoch) {
        std::shuffle(order.begin(), order.end(), random);
        for (const auto index : order) {
            const auto& sample = examples[index];
            loss(sample);
            const float rate = options.learning_rate * class_scale[sample.label];
            for (std::size_t i = 0; i < parameters.size(); ++i) {
                parameters[i] -= rate * gradient[i];
            }
        }
    }
    const double final = average_loss();
    return {Model(spec, std::move(parameters)), initial, final};
}

}  // namespace exanima
