#include "load_dataset.hpp"
#include "cifar/cifar10_reader.hpp"
#include "constants.hpp"
#include "mnist/mnist_reader.hpp"

constexpr size_t NORMALIZE_DENOM = 255;
constexpr double NORMALIZE_MEAN = 0.5;
constexpr double NORMALIZE_STD = 0.5;
const std::string MNIST_DATASET_PATH =
    constants::fname::DATASETS_DIR + "/" + constants::dataset::MNIST;
const std::string CIFAR10_DATASET_PATH =
    constants::fname::DATASETS_DIR + "/" + constants::dataset::CIFAR10;

namespace utils {

template <template <typename...> class Container, typename Image>
void normalize(Container<Image>& images) {
  size_t image_count = images.size();
  size_t pixel_count_per_image = images[0].size();
  for (size_t i = 0; i < image_count; ++i) {
    for (size_t j = 0; j < pixel_count_per_image; ++j) {
      images[i][j] /= NORMALIZE_DENOM;  // [0, 1]
      images[i][j] =
          (images[i][j] - NORMALIZE_MEAN) / NORMALIZE_STD;  // [-1, 1]
    }
  }
}

std::vector<std::vector<float>> load_mnist_test_images() {
  mnist::MNIST_dataset<std::vector, std::vector<float>, uint8_t> dataset =
      mnist::read_dataset<std::vector, std::vector, float, uint8_t>(
          MNIST_DATASET_PATH, 1, 0);

  normalize(dataset.test_images);

  return dataset.test_images;
}

std::vector<unsigned char> load_mnist_test_labels() {
  mnist::MNIST_dataset<std::vector, std::vector<float>, uint8_t> dataset =
      mnist::read_dataset<std::vector, std::vector, float, uint8_t>(
          MNIST_DATASET_PATH, 1, 0);

  return dataset.test_labels;
}

std::vector<std::vector<float>> load_cifar10_test_images() {
  cifar::CIFAR10_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
      cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(
          CIFAR10_DATASET_PATH, 1, 0);

  auto test_images = std::move(dataset.test_images);

  size_t image_count = test_images.size();
  size_t pixel_count_per_image = test_images[0].size();
  std::vector<std::vector<float>> float_test_images(
      image_count, std::vector<float>(pixel_count_per_image));

  // Translate from uint8_t to float
  for (size_t i = 0; i < image_count; ++i) {
    for (size_t j = 0; j < pixel_count_per_image; ++j) {
      float_test_images[i][j] =
          static_cast<float>(std::move(test_images[i][j]));
    }
  }

  normalize(float_test_images);

  return float_test_images;
}

std::vector<unsigned char> load_cifar10_test_labels() {
  cifar::CIFAR10_dataset<std::vector, std::vector<float>, uint8_t> dataset =
      cifar::read_dataset<std::vector, std::vector, float, uint8_t>(
          CIFAR10_DATASET_PATH, 1, 0);

  return dataset.test_labels;
}

std::vector<std::vector<float>> load_test_images(
    const std::string& dataset_name) {
  std::vector<std::vector<float>> test_images;
  if (dataset_name == constants::dataset::MNIST) {
    test_images = load_mnist_test_images();
  } else if (dataset_name == constants::dataset::CIFAR10) {
    test_images = load_cifar10_test_images();
  }

  return test_images;
}

std::vector<unsigned char> load_test_labels(const std::string& dataset_name) {
  std::vector<unsigned char> test_labels;
  if (dataset_name == constants::dataset::MNIST) {
    test_labels = load_mnist_test_labels();
  } else if (dataset_name == constants::dataset::CIFAR10) {
    test_labels = load_cifar10_test_labels();
  }

  return test_labels;
}

}  // namespace utils
