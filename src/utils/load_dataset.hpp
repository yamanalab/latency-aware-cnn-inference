#pragma once

#include <string>
#include <vector>

namespace utils {

template <template <typename...> class Container, typename Image>
void normalize(Container<Image>& images);

std::vector<std::vector<float>> load_test_images(
    const std::string& dataset_name);
std::vector<unsigned char> load_test_labels(const std::string& dataset_name);

}  // namespace utils
