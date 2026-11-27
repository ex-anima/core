#pragma once

#include "exanima/model.h"

#include <filesystem>

namespace exanima {

Model load_model(const std::filesystem::path& directory);

}  // namespace exanima
