#pragma once

#include "exanima/model.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace exanima::detail {

inline constexpr auto model_format = "exanima-mlp-v1";
inline constexpr auto dtype = "float32";
inline constexpr auto byte_order = "little";
inline constexpr auto parameter_order = "w1,b1,w2,b2";

static_assert(sizeof(float) == sizeof(std::uint32_t) && std::numeric_limits<float>::is_iec559,
              "The model format requires IEEE 754 binary32");

inline std::size_t bounded_size(const nlohmann::json& value, std::size_t maximum) {
    if (!value.is_number_unsigned() || value.get<std::uint64_t>() > maximum) {
        throw std::invalid_argument("Expected a bounded nonnegative integer");
    }
    return value.get<std::size_t>();
}

inline ModelSpec read_spec(const nlohmann::json& document) {
    return {document.at("input_features").get<std::vector<std::string>>(),
            bounded_size(document.at("hidden_units"), 4096),
            document.at("labels").get<std::vector<std::string>>()};
}

inline float read_number(const nlohmann::json& value) {
    if (!value.is_number()) {
        throw std::invalid_argument("Expected a JSON number");
    }
    const float converted = value.get<float>();
    if (!std::isfinite(converted)) {
        throw std::invalid_argument("Expected a finite float");
    }
    return converted;
}

inline std::vector<float> read_observation(const nlohmann::json& row, std::size_t size) {
    if (!row.is_array() || row.size() != size) {
        throw std::invalid_argument("Observation length does not match the model");
    }
    std::vector<float> observation;
    observation.reserve(size);
    for (const auto& value : row) {
        observation.push_back(read_number(value));
    }
    return observation;
}

inline std::uint32_t float_bits(float value) {
    std::uint32_t bits{};
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

inline float bits_float(std::uint32_t bits) {
    float value{};
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

}  // namespace exanima::detail
