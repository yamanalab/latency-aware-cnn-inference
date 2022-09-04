#pragma once

#include <seal/seal.h>

#include "layers/layer.hpp"

namespace cnn {

class Network {
public:
  Network();
  ~Network();

  types::float2d predict(types::float4d& x_4d);

  void add_layer(std::shared_ptr<Layer> layer) { layers_.push_back(layer); }

private:
  std::vector<std::shared_ptr<Layer>> layers_;
};

}  // namespace cnn

namespace cnn::encrypted {

class Network {
public:
  Network();
  ~Network();

  // seal::Ciphertext predict(std::vector<seal::Ciphertext>& x_cts);
  std::vector<seal::Ciphertext> predict(std::vector<seal::Ciphertext>& x_cts);

  void add_layer(std::shared_ptr<Layer> layer) { layers_.push_back(layer); }

private:
  std::vector<std::shared_ptr<Layer>> layers_;
};

}  // namespace cnn::encrypted

namespace cnn::encrypted::batch {

class Network {
public:
  Network();
  ~Network();

  std::vector<seal::Ciphertext> predict(types::Ciphertext3d& x_ct_3d);

  void add_layer(std::shared_ptr<Layer> layer) { layers_.push_back(layer); }

private:
  std::vector<std::shared_ptr<Layer>> layers_;
};

}  // namespace cnn::encrypted::batch
