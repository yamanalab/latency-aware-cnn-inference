#include "conv_2d.hpp"
#include "util.hpp"

#include <Eigen/Dense>

using std::size_t;

namespace cnn {

Conv2d::Conv2d(
    const types::float4d& filters,
    const std::vector<float>& biases,
    const std::pair<std::size_t, std::size_t>& stride,
    const std::pair<std::string, std::pair<std::size_t, std::size_t>>& padding)
    : Layer(ELayerType::CONV_2D),
      filters_(filters),
      biases_(biases),
      stride_(stride) {
  if (padding.first.empty()) {
    pad_top_ = padding.second.first;
    pad_btm_ = padding.second.first;
    pad_left_ = padding.second.second;
    pad_right_ = padding.second.second;
  } else if (padding.first == "valid") {
    pad_top_ = 0;
    pad_btm_ = 0;
    pad_left_ = 0;
    pad_right_ = 0;
  } else if (padding.first == "same") {
    assert(stride_.first == 1 && stride_.second == 1);
    const size_t fh = filters_.at(0).at(0).size(),
                 fw = filters_.at(0).at(0).at(0).size();
    pad_top_ = (fh - 1) / 2;
    pad_btm_ = fh - pad_top_ - 1;
    pad_left_ = (fw - 1) / 2;
    pad_right_ = fw - pad_left_ - 1;
  }
}
Conv2d::~Conv2d() {}

void Conv2d::forward(types::float4d& x) const {
  const size_t batch_size = x.size(), ih = x.at(0).at(0).size(),
               iw = x.at(0).at(0).at(0).size();
  const size_t fn = filters_.size(), fh = filters_.at(0).at(0).size(),
               fw = filters_.at(0).at(0).at(0).size();
  const size_t oh = static_cast<int>(
                   ((ih + pad_top_ + pad_btm_ - fh) / stride_.first) + 1),
               ow = static_cast<int>(
                   ((iw + pad_left_ + pad_right_ - fw) / stride_.second) + 1);

  types::float4d padded_x =
      util::apply_zero_padding(x, pad_top_, pad_btm_, pad_left_, pad_right_);

  auto col =
      util::im2col(padded_x, fh, fw, oh, ow, stride_);  // [N*OH*OW, IC*FH*FW]
  types::float2d flattened_filters = util::flatten_4d_vector_to_2d(filters_);
  auto col_filters =
      util::convert_to_eigen_matrix(flattened_filters);  // [FN, IC*FH*FW]

  auto wx_matrix = col * col_filters.transpose();

  std::vector<float> biases_copy(biases_.size());
  std::copy(biases_.begin(), biases_.end(), biases_copy.begin());
  auto bias_vec = util::convert_to_eigen_vector(biases_copy);

  Eigen::MatrixXf y_matrix(wx_matrix.rows(),
                           wx_matrix.cols());  // [N*OH*OW, FN]
  for (size_t i = 0; i < y_matrix.rows(); ++i) {
    y_matrix.row(i) = wx_matrix.row(i) + bias_vec.transpose();
  }

  y_matrix.transposeInPlace();  // [FN, N*OH*OW]

  types::float2d y_2d_vec = util::convert_to_float_2d(y_matrix);
  x = util::reshape_2d_vector_to_4d(y_2d_vec, batch_size, fn, oh, ow);
}

}  // namespace cnn

namespace cnn::encrypted {

Conv2d::Conv2d(const std::string layer_name,
               const types::Plaintext3d& filters_pts,
               const std::vector<seal::Plaintext>& biases_pts,
               const std::vector<int> rotation_map,
               const std::shared_ptr<helper::he::SealTool> seal_tool)
    : Layer(ELayerType::CONV_2D, layer_name, seal_tool),
      filters_pts_(filters_pts),
      biases_pts_(biases_pts),
      rotation_map_(rotation_map) {
  CONSUMED_LEVEL++;
}
Conv2d::Conv2d() {}
Conv2d::~Conv2d() {}

void Conv2d::forward(std::vector<seal::Ciphertext>& x_cts,
                     std::vector<seal::Ciphertext>& y_cts) {
  const size_t filter_count = filters_pts_.size();
  const size_t input_channel_size = x_cts.size();
  const size_t filter_hw_size = filters_pts_.at(0).at(0).size();
  // std::vector<seal::Ciphertext> mid_cts(input_channel_size * filter_hw_size);
  types::Ciphertext2d mid_cts(
      filter_count,
      std::vector<seal::Ciphertext>(input_channel_size * filter_hw_size));
  y_cts.resize(filter_count);

  std::cout << "\tForwarding " << layer_name() << "..." << std::endl;
  // size_t mid_cts_idx;
  // std::cout << "rotation_map: " << std::endl;
  // for (size_t i = 0; i < filter_hw_size; ++i) {
  //   std::cout << rotation_map_[i] << ", ";
  // }
  // std::cout << std::endl;
  // {
  //   seal::Plaintext plain_x;
  //   std::vector<double> x_values;
  //   seal_tool_->decryptor().decrypt(x_cts[0], plain_x);
  //   seal_tool_->encoder().decode(plain_x, x_values);
  //   std::cout << "x_values:" << std::endl;
  //   for (int s = 0; s < 30; ++s) {
  //     std::cout << x_values[s] << ", ";
  //   }
  //   std::cout << std::endl;
  //   for (int c = 0; c < input_channel_size; ++c) {
  //     std::cout << "filter[0][" << c << "] values:" << std::endl;
  //     for (int i = 0; i < filter_hw_size; ++i) {
  //       seal_tool_->encoder().decode(filters_pts_[0][c][i], x_values);
  //       std::cout << x_values[0] << ", ";
  //     }
  //     std::cout << std::endl;
  //   }
  // }
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
  for (size_t ci = 0; ci < input_channel_size; ++ci) {
    for (size_t i = 0; i < filter_hw_size; ++i) {
      seal::Ciphertext rotated_ct;
      size_t mid_cts_idx = ci * filter_hw_size + i;
      seal_tool_->evaluator().rotate_vector(x_cts[ci], rotation_map_[i],
                                            GALOIS_KEYS, rotated_ct);
      for (size_t fi = 0; fi < filter_count; ++fi) {
        mid_cts[fi][mid_cts_idx] = rotated_ct;
      }
    }
  }
#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
  for (size_t fi = 0; fi < filter_count; ++fi) {
    for (size_t ci = 0; ci < input_channel_size; ++ci) {
      for (size_t i = 0; i < filter_hw_size; ++i) {
        // size_t mid_cts_idx = ci * filter_hw_size + i;
        // seal_tool_->evaluator().rotate_vector(
        //     x_cts[ci], rotation_map_[i], GALOIS_KEYS,
        //     mid_cts[fi][mid_cts_idx]);
        // {
        //   if (fi == 0) {
        //     seal::Plaintext plain_x;
        //     std::vector<double> x_values;
        //     std::cout << "Rotated (" << rotation_map_[i] << ") mid_cts[0]["
        //     //               << mid_cts_idx << "]:" << std::endl;
        //     seal_tool_->decryptor().decrypt(mid_cts[fi][mid_cts_idx],
        //     plain_x); seal_tool_->encoder().decode(plain_x, x_values); for
        //     (int s = 0; s < 30; ++s) {
        //       std::cout << x_values[s] << ", ";
        //     }
        //     std::cout << std::endl;
        //   }
        // }
        seal_tool_->evaluator().multiply_plain_inplace(
            mid_cts[fi][ci * filter_hw_size + i], filters_pts_[fi][ci][i]);
      }
    }
  }

  // {
  //   seal::Plaintext plain_x;
  //   std::vector<double> x_values;
  //   for (int i = 0; i < filter_hw_size; ++i) {
  //     std::cout << "mid_cts[0][" << i << "]: " << std::endl;
  //     seal_tool_->decryptor().decrypt(mid_cts[0][i], plain_x);
  //     seal_tool_->encoder().decode(plain_x, x_values);
  //     for (int s = 0; s < 30; ++s) {
  //       std::cout << x_values[s] << ", ";
  //     }
  //     std::cout << std::endl;
  //   }
  //   seal_tool_->encoder().decode(biases_pts_[0], x_values);
  //   std::cout << "biases_pts_[0]: " << std::endl;
  //   for (int s = 0; s < 30; ++s) {
  //     std::cout << x_values[s] << ", ";
  //   }
  //   std::cout << std::endl;
  // }

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (size_t i = 0; i < filter_count; ++i) {
    seal_tool_->evaluator().add_many(mid_cts[i], y_cts[i]);
    seal_tool_->evaluator().rescale_to_next_inplace(y_cts[i]);
    y_cts[i].scale() = seal_tool_->scale();
    // seal_tool_->evaluator().mod_switch_to_inplace(biases_pts_[i],
    //                                               y_cts[i].parms_id());
    seal_tool_->evaluator().add_plain_inplace(y_cts[i], biases_pts_[i]);
  }

  // {
  //   seal::Plaintext plain_bias;
  //   std::vector<double> bias_values(seal_tool_->slot_count());
  //   std::cout << "bias_values:" << std::endl;
  //   for (size_t i = 0; i < filter_count; ++i) {
  //     seal_tool_->encoder().decode(biases_pts_[i], bias_values);
  //     std::cout << bias_values[0] << ", ";
  //   }
  //   std::cout << std::endl;
  //   seal::Plaintext plain_y;
  //   std::vector<double> y_values(seal_tool_->slot_count());
  //   seal_tool_->decryptor().decrypt(y_cts[0], plain_y);
  //   seal_tool_->encoder().decode(plain_y, y_values);
  //   for (int s = 0; s < 10; ++s) {
  //     std::cout << "y_values[" << s << "]: " << y_values[s] << std::endl;
  //   }
  // }
}

}  // namespace cnn::encrypted

namespace cnn::encrypted::batch {

Conv2d::Conv2d(
    const std::string layer_name,
    const types::Plaintext4d& plain_filters,
    const std::vector<seal::Plaintext>& plain_biases,
    const std::shared_ptr<helper::he::SealTool> seal_tool,
    const std::size_t stride_h,
    const std::size_t stride_w,
    const std::pair<std::string, std::pair<std::size_t, std::size_t>> padding)
    : Layer(ELayerType::CONV_2D, layer_name, seal_tool),
      plain_filters_(plain_filters),
      plain_biases_(plain_biases),
      stride_h_(stride_h),
      stride_w_(stride_w) {
  if (padding.first.empty()) {
    pad_top_ = padding.second.first;
    pad_btm_ = padding.second.first;
    pad_left_ = padding.second.second;
    pad_right_ = padding.second.second;
  } else if (padding.first == "valid") {
    pad_top_ = 0;
    pad_btm_ = 0;
    pad_left_ = 0;
    pad_right_ = 0;
  } else if (padding.first == "same") {
    assert(stride_h == 1 && stride_w == 1);
    const size_t fh = plain_filters.at(0).at(0).size(),
                 fw = plain_filters.at(0).at(0).at(0).size();
    pad_top_ = (fh - 1) / 2;
    pad_btm_ = fh - pad_top_ - 1;
    pad_left_ = (fw - 1) / 2;
    pad_right_ = fw - pad_left_ - 1;
  }
  CONSUMED_LEVEL++;
}
Conv2d::~Conv2d() {}

bool Conv2d::isOutOfRangeInput(const int target_x,
                               const int target_y,
                               const std::size_t input_w,
                               const std::size_t input_h) {
  return target_x < 0 || target_y < 0 || target_x >= input_w ||
         target_y >= input_h;
}

void Conv2d::forward(types::Ciphertext3d& x_ct_3d) {
  const std::size_t input_c = x_ct_3d.size(), input_h = x_ct_3d.at(0).size(),
                    input_w = x_ct_3d.at(0).at(0).size(),
                    output_c = plain_filters_.size(),
                    filter_h = plain_filters_.at(0).at(0).size(),
                    filter_w = plain_filters_.at(0).at(0).at(0).size();
  const std::size_t
      output_h = static_cast<std::size_t>(
          ((input_h + pad_top_ + pad_btm_ - filter_h) / stride_h_) + 1),
      output_w = static_cast<std::size_t>(
          ((input_w + pad_left_ + pad_right_ - filter_w) / stride_w_) + 1);

  std::cout << "\tForwarding " << layer_name() << "..." << std::endl;
  std::cout << "\t  input shape: " << input_c << "x" << input_h << "x"
            << input_w << std::endl;

  types::Ciphertext3d output(
      output_c,
      types::Ciphertext2d(output_h, std::vector<seal::Ciphertext>(output_w)));
  std::vector<std::vector<std::vector<bool>>> output_exist_map(
      output_c, std::vector<std::vector<bool>>(
                    output_h, std::vector<bool>(output_w, false)));

  int target_top, target_left, target_x, target_y;
  std::size_t within_range_counter;
  seal::Ciphertext weighted_pixel;

#ifdef _OPENMP
#pragma omp parallel for collapse(3) private(                          \
    target_top, target_left, target_x, target_y, within_range_counter, \
    weighted_pixel)
#endif
  for (std::size_t oc = 0; oc < output_c; ++oc) {
    for (std::size_t oh = 0; oh < output_h; ++oh) {
      for (std::size_t ow = 0; ow < output_w; ++ow) {
        target_top = oh * stride_h_ - pad_top_;
        target_left = ow * stride_w_ - pad_left_;
        for (std::size_t ic = 0; ic < input_c; ++ic) {
          for (std::size_t fh = 0; fh < filter_h; ++fh) {
            for (std::size_t fw = 0; fw < filter_w; ++fw) {
              target_x = target_left + fw;
              target_y = target_top + fh;
              if (isOutOfRangeInput(target_x, target_y, input_w, input_h))
                continue;
              seal_tool_->evaluator().multiply_plain(
                  x_ct_3d[ic][target_y][target_x],
                  plain_filters_[oc][ic][fh][fw], weighted_pixel);
              if (!output_exist_map[oc][oh][ow]) {
                output[oc][oh][ow] = weighted_pixel;
                output_exist_map[oc][oh][ow] = true;
              } else {
                seal_tool_->evaluator().add_inplace(output[oc][oh][ow],
                                                    weighted_pixel);
              }
            }
          }
        }
        seal_tool_->evaluator().rescale_to_next_inplace(output[oc][oh][ow]);
        output[oc][oh][ow].scale() = seal_tool_->scale();
        seal_tool_->evaluator().add_plain_inplace(output[oc][oh][ow],
                                                  plain_biases_[oc]);
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
