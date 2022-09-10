#pragma once

#include "layer.hpp"

const std::string FLATTEN_CLASS_NAME = "Flatten";

namespace cnn {

class Flatten : public Layer {
public:
  Flatten();
  ~Flatten();

  void forward(types::float4d& x, types::float2d& y) const override;
};

}  // namespace cnn

namespace cnn::encrypted {

class Flatten : public Layer {
public:
  Flatten(const std::string layer_name,
          const std::vector<int> rotation_map,
          const std::shared_ptr<helper::he::SealTool> seal_tool);
  Flatten();
  ~Flatten();

  void forward(std::vector<seal::Ciphertext>& x_cts,
               seal::Ciphertext& y_ct) const override;

private:
  std::vector<int> rotation_map_;
};

}  // namespace cnn::encrypted

namespace cnn::encrypted::batch {

class Flatten : public Layer {
public:
  Flatten(const std::string layer_name,
          const std::shared_ptr<helper::he::SealTool> seal_tool);
  ~Flatten();

  void forward(types::Ciphertext3d& x_ct_3d,
               std::vector<seal::Ciphertext>& x_cts) const override;

private:
};

}  // namespace cnn::encrypted::batch
