#pragma once

#include "exanima/training.h"
#include "layout.h"

namespace exanima::detail {

double loss_and_gradient(Layout layout, const std::vector<float>& parameters,
                         const Demonstration& example, std::vector<float>& hidden,
                         std::vector<float>& output, std::vector<float>& gradient);

}  // namespace exanima::detail
