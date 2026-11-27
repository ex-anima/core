#include "exanima/model.h"
#include "layout.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace exanima::detail {
namespace {

void check_names(const std::vector<std::string>& names) {
    std::unordered_set<std::string> unique;
    for (const auto& name : names) {
        if (name.empty() || name.size() > 256 || !unique.insert(name).second) {
            throw std::invalid_argument("Feature names and labels must be nonempty and unique");
        }
    }
}

}  // namespace

Layout validate_spec(const ModelSpec& spec) {
    const Layout layout{spec.input_features.size(), spec.hidden_units, spec.labels.size()};
    if (layout.input == 0 || layout.input > 4096 || layout.hidden == 0 || layout.hidden > 4096
        || layout.output < 2 || layout.output > 4096 || layout.size() > 1'000'000) {
        throw std::invalid_argument("Unsupported model dimensions");
    }
    check_names(spec.input_features);
    check_names(spec.labels);
    return layout;
}

bool finite(const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); });
}

void forward(Layout shape, const std::vector<float>& parameters,
             const std::vector<float>& input, std::vector<float>& hidden, std::vector<float>& output) {
    for (std::size_t h = 0; h < shape.hidden; ++h) {
        float value = parameters[shape.bias1() + h];
        for (std::size_t i = 0; i < shape.input; ++i) {
            value += parameters[h * shape.input + i] * input[i];
        }
        if (!std::isfinite(value)) {
            throw std::runtime_error("Hidden layer produced a nonfinite value");
        }
        hidden[h] = std::max(value, 0.0f);
    }
    for (std::size_t c = 0; c < shape.output; ++c) {
        float value = parameters[shape.bias2() + c];
        for (std::size_t h = 0; h < shape.hidden; ++h) {
            value += parameters[shape.weights2() + c * shape.hidden + h] * hidden[h];
        }
        output[c] = value;
    }
}

}  // namespace exanima::detail

namespace exanima {

Model::Model(ModelSpec spec, std::vector<float> parameters)
    : spec_(std::move(spec)), parameters_(std::move(parameters)) {
    const auto shape = detail::validate_spec(spec_);
    if (parameters_.size() != shape.size() || !detail::finite(parameters_)) {
        throw std::invalid_argument("Parameter count or values do not match the model");
    }
}

std::vector<float> Model::logits(const std::vector<float>& observation) const {
    const detail::Layout shape{spec_.input_features.size(), spec_.hidden_units, spec_.labels.size()};
    if (observation.size() != shape.input || !detail::finite(observation)) {
        throw std::invalid_argument("Observation shape or values do not match the model");
    }
    std::vector<float> hidden(shape.hidden);
    std::vector<float> output(shape.output);
    detail::forward(shape, parameters_, observation, hidden, output);
    if (!detail::finite(output)) {
        throw std::runtime_error("Inference produced nonfinite logits");
    }
    return output;
}

std::size_t Model::predict(const std::vector<float>& observation) const {
    const auto scores = logits(observation);
    return static_cast<std::size_t>(std::max_element(scores.begin(), scores.end()) - scores.begin());
}

}  // namespace exanima
