#pragma once

#include "layer.hpp"

const std::string LINEAR_CLASS_NAME = "Linear";

namespace cnn {

class Linear : public Layer {
public:
  Linear();
  ~Linear();

  void forward(types::float2d& x) const override;

private:
};

}  // namespace cnn

namespace cnn::encrypted {

class Linear : public Layer {
public:
  Linear(const std::string layer_name,
         const std::vector<seal::Plaintext>& weights_pts,
         const std::vector<seal::Plaintext>& biases_pts,
         const std::shared_ptr<helper::he::SealTool> seal_tool);
  ~Linear();

  void forward(seal::Ciphertext& x_ct, seal::Ciphertext& y_ct) override;
  void forward(seal::Ciphertext& x_ct,
               std::vector<seal::Ciphertext>& y_cts) override;

private:
  std::vector<seal::Plaintext> weights_pts_;  // form of [OC]
  std::vector<seal::Plaintext> biases_pts_;   // form of [OC]
};

}  // namespace cnn::encrypted

namespace cnn::encrypted::batch {

class Linear : public Layer {
public:
  Linear(const std::string layer_name,
         types::Plaintext2d plain_weights,
         std::vector<seal::Plaintext> plain_biases,
         const std::shared_ptr<helper::he::SealTool> seal_tool);
  ~Linear();

  void forward(std::vector<seal::Ciphertext>& x_cts) override;

private:
  types::Plaintext2d plain_weights_;           // form of [OC, IC]
  std::vector<seal::Plaintext> plain_biases_;  // form of [OC]
};

}  // namespace cnn::encrypted::batch
