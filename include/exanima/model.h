#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace exanima {

struct ModelSpec {
    std::vector<std::string> input_features;
    std::size_t hidden_units = 0;
    std::vector<std::string> labels;
};

class Model {
public:
    Model(ModelSpec spec, std::vector<float> parameters);

    const ModelSpec& spec() const { return spec_; }
    const std::vector<float>& parameters() const { return parameters_; }
    std::vector<float> logits(const std::vector<float>& observation) const;
    std::size_t predict(const std::vector<float>& observation) const;

private:
    ModelSpec spec_;
    std::vector<float> parameters_;
};

}  // namespace exanima
