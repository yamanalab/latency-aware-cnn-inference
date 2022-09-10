#include "avg_pool_2d.hpp"

namespace cnn {

AvgPool2d::AvgPool2d() : Layer(ELayerType::AVG_POOL_2D) {}
AvgPool2d::~AvgPool2d() {}

void AvgPool2d::forward(types::float4d& x) const {}

}  // namespace cnn

namespace cnn::encrypted {

AvgPool2d::AvgPool2d(const std::string layer_name,
                     const std::size_t pool_hw_size,
                     const seal::Plaintext& plain_mul_factor,
                     const std::vector<int> rotation_map,
                     const bool is_gap,
                     const std::shared_ptr<helper::he::SealTool> seal_tool)
    : Layer(ELayerType::AVG_POOL_2D, layer_name, seal_tool),
      pool_hw_size_(pool_hw_size),
      plain_mul_factor_(plain_mul_factor),
      rotation_map_(rotation_map),
      is_gap_(is_gap) {
  if (!OPT_OPTION.enable_fold_pool_coeff || is_gap_) {
    CONSUMED_LEVEL++;
  }
}
AvgPool2d::AvgPool2d() {}
AvgPool2d::~AvgPool2d() {}

void AvgPool2d::forward(std::vector<seal::Ciphertext>& x_cts,
                        std::vector<seal::Ciphertext>& y_cts) {
  const std::size_t input_channel_size = x_cts.size();
  // std::vector<seal::Ciphertext> mid_cts(pool_hw_size_);
  types::Ciphertext2d mid_cts(input_channel_size,
                              std::vector<seal::Ciphertext>(pool_hw_size_));
  y_cts.resize(input_channel_size);

  std::cout << "\tForwarding " << layer_name() << "..." << std::endl;
  if (OPT_OPTION.enable_fold_pool_coeff && !is_gap_) {
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (std::size_t ci = 0; ci < input_channel_size; ++ci) {
      for (std::size_t i = 0; i < pool_hw_size_; ++i) {
        seal_tool_->evaluator().rotate_vector(x_cts[ci], rotation_map_[i],
                                              GALOIS_KEYS, mid_cts[ci][i]);
      }
    }

    // {
    //   for (std::size_t i = 0; i < pool_hw_size_; ++i) {
    //     seal::Plaintext plain_x;
    //     std::vector<double> x_values;
    //     std::cout << "Rotated (" << rotation_map_[i] << ") mid_cts[0][" << i
    //               << "]:" << std::endl;
    //     seal_tool_->decryptor().decrypt(mid_cts[0][i], plain_x);
    //     seal_tool_->encoder().decode(plain_x, x_values);
    //     for (int s = 0; s < 30; ++s) {
    //       std::cout << x_values[s] << ", ";
    //     }
    //     std::cout << std::endl;
    //   }
    // }

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (std::size_t i = 0; i < input_channel_size; ++i) {
      seal_tool_->evaluator().add_many(mid_cts[i], y_cts[i]);
      y_cts[i].scale() = seal_tool_->scale();
    }

    // {
    //   seal::Plaintext plain_y;
    //   std::vector<double> y_values(seal_tool_->slot_count());
    //   seal_tool_->decryptor().decrypt(y_cts[0], plain_y);
    //   seal_tool_->encoder().decode(plain_y, y_values);
    //   for (int s = 0; s < 10; ++s) {
    //     std::cout << "y_values[" << s << "]: " << y_values[s] << std::endl;
    //   }
    // }
  } else {
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (std::size_t ci = 0; ci < input_channel_size; ++ci) {
      for (std::size_t i = 0; i < pool_hw_size_; ++i) {
        seal_tool_->evaluator().rotate_vector(x_cts[ci], rotation_map_[i],
                                              GALOIS_KEYS, mid_cts[ci][i]);
        // seal_tool_->evaluator().multiply_plain_inplace(mid_cts[ci][i],
        //                                                plain_mul_factor_);
        // seal_tool_->evaluator().rescale_to_next_inplace(mid_cts[ci][i]);
      }
    }

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (std::size_t i = 0; i < input_channel_size; ++i) {
      seal_tool_->evaluator().add_many(mid_cts[i], y_cts[i]);
      seal_tool_->evaluator().multiply_plain_inplace(y_cts[i],
                                                     plain_mul_factor_);
      seal_tool_->evaluator().rescale_to_next_inplace(y_cts[i]);
      y_cts[i].scale() = seal_tool_->scale();
    }
  }
}

}  // namespace cnn::encrypted

namespace cnn::encrypted::batch {

AvgPool2d::AvgPool2d(const std::string layer_name,
                     const seal::Plaintext plain_mul_factor,
                     const std::size_t pool_h,
                     const std::size_t pool_w,
                     const std::size_t stride_h,
                     const std::size_t stride_w,
                     const std::pair<std::size_t, std::size_t> padding,
                     const std::shared_ptr<helper::he::SealTool> seal_tool)
    : Layer(ELayerType::AVG_POOL_2D, layer_name, seal_tool),
      plain_mul_factor_(plain_mul_factor),
      pool_h_(pool_h),
      pool_w_(pool_w),
      stride_h_(stride_h),
      stride_w_(stride_w),
      pad_top_(padding.first),
      pad_btm_(padding.first),
      pad_left_(padding.second),
      pad_right_(padding.second) {
  if (!OPT_OPTION.enable_fold_pool_coeff) {
    CONSUMED_LEVEL++;
  }
}
AvgPool2d::~AvgPool2d() {}

bool AvgPool2d::isOutOfRangeInput(const int target_x,
                                  const int target_y,
                                  const std::size_t input_w,
                                  const std::size_t input_h) {
  return target_x < 0 || target_y < 0 || target_x >= input_w ||
         target_y >= input_h;
}

void AvgPool2d::forward(types::Ciphertext3d& x_ct_3d) {
  const std::size_t input_c = x_ct_3d.size(), input_h = x_ct_3d.at(0).size(),
                    input_w = x_ct_3d.at(0).at(0).size(), output_c = input_c;
  const std::size_t output_h = static_cast<std::size_t>(
      ((input_h + pad_top_ + pad_btm_ - pool_h_) / stride_h_) + 1);
  const std::size_t output_w = static_cast<std::size_t>(
      ((input_w + pad_left_ + pad_right_ - pool_w_) / stride_w_) + 1);

  std::cout << "\tForwarding " << layer_name() << "..." << std::endl;
  std::cout << "\t  input shape: " << input_c << "x" << input_h << "x"
            << input_w << std::endl;

  types::Ciphertext3d output(
      output_c,
      types::Ciphertext2d(output_h, std::vector<seal::Ciphertext>(output_w)));
  int target_top, target_left, target_x, target_y;

  if (OPT_OPTION.enable_fold_pool_coeff) {
#ifdef _OPENMP
#pragma omp parallel for collapse(3) private(target_top, target_left, \
                                             target_x, target_y)
#endif
    for (size_t oc = 0; oc < output_c; ++oc) {
      for (size_t oh = 0; oh < output_h; ++oh) {
        for (size_t ow = 0; ow < output_w; ++ow) {
          target_top = oh * stride_h_ - pad_top_;
          target_left = ow * stride_w_ - pad_left_;
          for (size_t ph = 0; ph < pool_h_; ++ph) {
            for (size_t pw = 0; pw < pool_w_; ++pw) {
              target_x = target_left + pw;
              target_y = target_top + ph;
              if (isOutOfRangeInput(target_x, target_y, input_w, input_h))
                continue;
              if (ph == 0 && pw == 0) {
                output[oc][oh][ow] = x_ct_3d[oc][target_y][target_x];
              } else {
                seal_tool_->evaluator().add_inplace(
                    output[oc][oh][ow], x_ct_3d[oc][target_y][target_x]);
              }
            }
          }
        }
      }
    }
  } else {
#ifdef _OPENMP
#pragma omp parallel for collapse(3) private(target_top, target_left, \
                                             target_x, target_y)
#endif
    for (size_t oc = 0; oc < output_c; ++oc) {
      for (size_t oh = 0; oh < output_h; ++oh) {
        for (size_t ow = 0; ow < output_w; ++ow) {
          target_top = oh * stride_h_ - pad_top_;
          target_left = ow * stride_w_ - pad_left_;
          for (size_t ph = 0; ph < pool_h_; ++ph) {
            for (size_t pw = 0; pw < pool_w_; ++pw) {
              target_x = target_left + pw;
              target_y = target_top + ph;
              if (isOutOfRangeInput(target_x, target_y, input_w, input_h))
                continue;
              if (ph == 0 && pw == 0) {
                output[oc][oh][ow] = x_ct_3d[oc][target_y][target_x];
              } else {
                seal_tool_->evaluator().add_inplace(
                    output[oc][oh][ow], x_ct_3d[oc][target_y][target_x]);
              }
            }
          }
          seal_tool_->evaluator().multiply_plain_inplace(output[oc][oh][ow],
                                                         plain_mul_factor_);
          seal_tool_->evaluator().rescale_to_next_inplace(output[oc][oh][ow]);
        }
      }
    }
  }

  // x_ct_3d.resize(
  //     output_c,
  //     types::Ciphertext2d(output_h,
  //     std::vector<seal::Ciphertext>(output_w)));
  x_ct_3d.resize(output_c);
  for (std::size_t oc = 0; oc < output_c; ++oc) {
    x_ct_3d[oc].resize(output_h);
    for (std::size_t oh = 0; oh < output_h; ++oh) {
      x_ct_3d[oc][oh].resize(output_w);
    }
  }
#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
  for (std::size_t oc = 0; oc < output_c; ++oc) {
    for (std::size_t oh = 0; oh < output_h; ++oh) {
      for (std::size_t ow = 0; ow < output_w; ++ow) {
        x_ct_3d[oc][oh][ow] = std::move(output[oc][oh][ow]);
      }
    }
  }
}

}  // namespace cnn::encrypted::batch
