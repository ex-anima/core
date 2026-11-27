#include "exanima/training.h"
#include "format.h"

#include <array>
#include <fstream>

namespace exanima {

void save_model(const Model& model, const std::filesystem::path& directory) {
    if (!std::filesystem::create_directory(directory)) {
        throw std::invalid_argument("Model output directory already exists");
    }
    std::ofstream binary(directory / "weights.bin", std::ios::binary);
    binary.exceptions(std::ios::failbit | std::ios::badbit);
    for (const float value : model.parameters()) {
        const auto bits = detail::float_bits(value);
        std::array<unsigned char, 4> bytes{};
        for (unsigned byte = 0; byte < bytes.size(); ++byte) {
            bytes[byte] = static_cast<unsigned char>((bits >> (8 * byte)) & 0xff);
        }
        binary.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
    binary.close();

    const auto& spec = model.spec();
    const nlohmann::json document = {
        {"format", detail::model_format}, {"dtype", detail::dtype}, {"byte_order", detail::byte_order},
        {"parameter_order", detail::parameter_order}, {"parameter_count", model.parameters().size()},
        {"input_features", spec.input_features}, {"hidden_units", spec.hidden_units}, {"labels", spec.labels}
    };
    const auto temporary = directory / "model.json.partial";
    std::ofstream metadata(temporary, std::ios::binary);
    metadata.exceptions(std::ios::failbit | std::ios::badbit);
    metadata << document.dump(2) << '\n';
    metadata.close();
    std::filesystem::rename(temporary, directory / "model.json");
}

}  // namespace exanima
