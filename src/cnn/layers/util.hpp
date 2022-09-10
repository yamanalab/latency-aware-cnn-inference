#pragma once

#include <Eigen/Dense>

#include "utils/types.hpp"

namespace cnn::util {

Eigen::MatrixXf im2col(types::float4d& x,
                       const std::size_t& fh,
                       const std::size_t& fw,
                       const std::size_t& oh,
                       const std::size_t& ow,
                       const std::pair<std::size_t, std::size_t>& stride);

types::float4d apply_zero_padding(types::float4d& x,
                                  const std::size_t& pad_top,
                                  const std::size_t& pad_btm,
                                  const std::size_t& pad_left,
                                  const std::size_t& pad_right);

Eigen::MatrixXf convert_to_eigen_matrix(types::float2d& vec_2d);

Eigen::VectorXf convert_to_eigen_vector(std::vector<float>& vec);

types::float2d convert_to_float_2d(Eigen::MatrixXf& matrix);

/**
 * @brief flatten vector4d -> vector2d
 * @param vec_4d input in the form of [FN, IC, FH, FW]
 * @return 2d vector in the form of [FN, IC*FH*FW]
 */
template <typename T>
types::vector2d<T> flatten_4d_vector_to_2d(const types::vector4d<T>& vec_4d) {
  const size_t row_size = vec_4d.size(), dj = vec_4d.at(0).size(),
               dk = vec_4d.at(0).at(0).size(),
               dl = vec_4d.at(0).at(0).at(0).size();
  const size_t col_size = dj * dk * dl;

  types::vector2d<T> vec_2d(row_size, std::vector<T>(col_size));
  for (size_t i = 0, idx = 0; i < row_size; ++i, idx = 0) {
    for (size_t j = 0; j < dj; ++j) {
      for (size_t k = 0; k < dk; ++k) {
        for (size_t l = 0; l < dl; ++l) {
          vec_2d.at(i).at(idx++) = vec_4d.at(i).at(j).at(k).at(l);
        }
      }
    }
  }

  return vec_2d;
}

/**
 * @brief reshape vector2d -> vector4d
 * @param vec_2d input in the form of [C, N*H*W]
 * @return 4d vector in the form of [N, C, H, W]
 */
template <typename T>
types::vector4d<T> reshape_2d_vector_to_4d(types::vector2d<T>& vec_2d,
                                           const size_t n,
                                           const size_t& c,
                                           const size_t& h,
                                           const size_t& w) {
  const size_t hw = h * w, row_size = n * h * w;

  types::vector4d<T> vec_4d(
      n, types::vector3d<T>(c, types::vector2d<T>(h, std::vector<T>(w))));
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < c; ++j) {
      for (size_t k = 0; k < h; ++k) {
        for (size_t l = 0; l < w; ++l) {
          vec_4d.at(i).at(j).at(k).at(l) =
              vec_2d.at(j * row_size).at(i * hw + k * w + l);
        }
      }
    }
  }

  return vec_4d;
}

}  // namespace cnn::util
