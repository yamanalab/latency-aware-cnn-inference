#pragma once

#include "layer.hpp"

const std::string AVG_POOL_2D_CLASS_NAME = "AvgPool2d";

namespace cnn {

class AvgPool2d : public Layer {
public:
  // AvgPool2d(const std::pair<std::size_t, std::size_t>& kernel,
  //           const std::pair<std::size_t, std::size_t>& stride,
  //           const std::pair<std::size_t, std::size_t>& padding);
  AvgPool2d();
  ~AvgPool2d();

  void forward(types::float4d& x) const override;

private:
};

}  // namespace cnn

namespace cnn::encrypted {

class AvgPool2d : public Layer {
public:
  AvgPool2d(const std::string layer_name,
            const std::size_t pool_hw_size,
            const seal::Plaintext& plain_mul_factor,
            const std::vector<int> rotation_map,
            const bool is_gap,
            const std::shared_ptr<helper::he::SealTool> seal_tool);
  AvgPool2d();
  ~AvgPool2d();

  void forward(std::vector<seal::Ciphertext>& x_cts,
               std::vector<seal::Ciphertext>& y_cts) override;

private:
  std::size_t pool_hw_size_;
  seal::Plaintext plain_mul_factor_;
  std::vector<int> rotation_map_;
  bool is_gap_;
};

}  // namespace cnn::encrypted

namespace cnn::encrypted::batch {

class AvgPool2d : public Layer {
public:
  AvgPool2d(const std::string layer_name,
            const seal::Plaintext plain_mul_factor,
            const std::size_t pool_h,
            const std::size_t pool_w,
            const std::size_t stride_h,
            const std::size_t stride_w,
            const std::pair<std::size_t, std::size_t> padding,
            const std::shared_ptr<helper::he::SealTool> seal_tool);
  ~AvgPool2d();

  bool isOutOfRangeInput(const int target_x,
                         const int target_y,
                         const std::size_t input_w,
                         const std::size_t input_h);

  void forward(types::Ciphertext3d& x_ct_3d) override;

private:
  seal::Plaintext plain_mul_factor_;
  std::size_t pool_h_;
  std::size_t pool_w_;
  std::size_t stride_h_;
  std::size_t stride_w_;
  std::size_t pad_top_;
  std::size_t pad_btm_;
  std::size_t pad_left_;
  std::size_t pad_right_;
};

}  // namespace cnn::encrypted::batch
