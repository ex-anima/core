#pragma once

#include "exanima/model.h"

namespace exanima::detail {

struct Layout {
    std::size_t input;
    std::size_t hidden;
    std::size_t output;

    std::size_t bias1() const { return input * hidden; }
    std::size_t weights2() const { return bias1() + hidden; }
    std::size_t bias2() const { return weights2() + hidden * output; }
    std::size_t size() const { return bias2() + output; }
};

Layout validate_spec(const ModelSpec& spec);
bool finite(const std::vector<float>& values);
void forward(Layout layout, const std::vector<float>& parameters,
             const std::vector<float>& input, std::vector<float>& hidden, std::vector<float>& output);

}  // namespace exanima::detail
