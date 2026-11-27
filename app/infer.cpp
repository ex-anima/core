#include "exanima/inference.h"
#include "format.h"

#include <algorithm>
#include <iostream>

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
#else
int main(int argc, char** argv) {
#endif
    if (argc != 2) {
        std::cerr << "Usage: exanima-infer MODEL_DIRECTORY < observations.jsonl\n";
        return 1;
    }
    try {
        const auto model = exanima::load_model(argv[1]);
        std::string line;
        while (std::getline(std::cin, line)) {
            const auto observation = exanima::detail::read_observation(
                nlohmann::json::parse(line), model.spec().input_features.size());
            const auto scores = model.logits(observation);
            const auto index = static_cast<std::size_t>(std::max_element(scores.begin(), scores.end()) - scores.begin());
            const nlohmann::json response = {{"index", index}, {"label", model.spec().labels[index]},
                                             {"logits", scores}};
            std::cout << response.dump() << std::endl;
        }
        return std::cin.bad() || !std::cout ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "Inference: " << error.what() << '\n';
        return 1;
    }
}
