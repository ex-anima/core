#include "exanima/inference.h"
#include "format.h"
#include "layout.h"

#include <array>
#include <fstream>
#include <utility>

namespace exanima {

Model load_model(const std::filesystem::path& directory) {
    const auto metadata_path = directory / "model.json";
    if (std::filesystem::file_size(metadata_path) > 4 * 1024 * 1024) {
        throw std::invalid_argument("Model metadata is too large");
    }
    std::ifstream metadata(metadata_path, std::ios::binary);
    const auto document = nlohmann::json::parse(metadata);
    if (document.at("format") != detail::model_format || document.at("dtype") != detail::dtype
        || document.at("byte_order") != detail::byte_order
        || document.at("parameter_order") != detail::parameter_order) {
        throw std::invalid_argument("Unsupported model format");
    }
    auto spec = detail::read_spec(document);
    const auto layout = detail::validate_spec(spec);
    const auto path = directory / "weights.bin";
    if (detail::bounded_size(document.at("parameter_count"), 1'000'000) != layout.size()
        || std::filesystem::file_size(path) != layout.size() * sizeof(float)) {
        throw std::invalid_argument("Parameter file length does not match model metadata");
    }
    std::ifstream binary(path, std::ios::binary);
    binary.exceptions(std::ios::failbit | std::ios::badbit);
    std::vector<float> parameters(layout.size());
    for (auto& value : parameters) {
        std::array<unsigned char, 4> bytes{};
        binary.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
        std::uint32_t bits = 0;
        for (unsigned byte = 0; byte < bytes.size(); ++byte) {
            bits |= static_cast<std::uint32_t>(bytes[byte]) << (8 * byte);
        }
        value = detail::bits_float(bits);
    }
    if (binary.peek() != std::char_traits<char>::eof()) {
        throw std::invalid_argument("Unexpected trailing parameters");
    }
    return Model(std::move(spec), std::move(parameters));
}

}  // namespace exanima
