#pragma once

#include "exanima/model.h"

#include <cstdint>
#include <filesystem>

namespace exanima {

struct Demonstration {
    std::vector<float> observation;
    std::size_t label = 0;
};

struct TrainingOptions {
    std::uint32_t seed = 42;
    int epochs = 200;
    float learning_rate = 0.03f;
    bool balance_classes = false;
};

struct TrainingResult {
    Model model;
    double initial_loss;
    double final_loss;
};

TrainingResult fit(const ModelSpec& spec, const std::vector<Demonstration>& examples,
                   const TrainingOptions& options = {});
void save_model(const Model& model, const std::filesystem::path& directory);

}  // namespace exanima
