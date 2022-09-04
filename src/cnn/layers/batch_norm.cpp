#include "batch_norm.hpp"

namespace cnn {

BatchNorm::BatchNorm() : Layer(ELayerType::BATCH_NORM) {}
BatchNorm::~BatchNorm() {}

void BatchNorm::forward(types::float4d& x) const {}

void BatchNorm::forward(types::float2d& x) const {}

}  // namespace cnn

namespace cnn::encrypted {

BatchNorm::BatchNorm(const std::string layer_name,
                     const std::vector<seal::Plaintext>& weights_pts,
                     const std::vector<seal::Plaintext>& biases_pts,
                     const std::shared_ptr<helper::he::SealTool> seal_tool)
    : Layer(ELayerType::BATCH_NORM, layer_name, seal_tool),
      weights_pts_(weights_pts),
      biases_pts_(biases_pts) {}
BatchNorm::BatchNorm() {}
BatchNorm::~BatchNorm() {}

void BatchNorm::forward(std::vector<seal::Ciphertext>& x_cts,
                        std::vector<seal::Ciphertext>& y_cts) {
  const size_t input_channel_size = x_cts.size();
  y_cts.resize(input_channel_size);

  std::cout << "\tForwarding " << layer_name() << "..." << std::endl;
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (size_t i = 0; i < input_channel_size; ++i) {
    y_cts[i] = x_cts[i];
    seal_tool_->evaluator().multiply_plain_inplace(y_cts[i], weights_pts_[i]);
    seal_tool_->evaluator().rescale_to_next_inplace(y_cts[i]);
    y_cts[i].scale() = seal_tool_->scale();
    seal_tool_->evaluator().add_plain_inplace(y_cts[i], biases_pts_[i]);
  }
}

void BatchNorm::forward(seal::Ciphertext& x_ct, seal::Ciphertext& y_ct) {}

}  // namespace cnn::encrypted

namespace cnn::encrypted::batch {

BatchNorm::BatchNorm(const std::string layer_name,
                     const std::vector<seal::Plaintext>& plain_weights,
                     const std::vector<seal::Plaintext>& plain_biases,
                     const std::shared_ptr<helper::he::SealTool> seal_tool)
    : Layer(ELayerType::BATCH_NORM, layer_name, seal_tool),
      plain_weights_(plain_weights),
      plain_biases_(plain_biases) {
  CONSUMED_LEVEL++;
}
BatchNorm::~BatchNorm() {}

void BatchNorm::forward(types::Ciphertext3d& x_ct_3d) {
  const std::size_t input_c = x_ct_3d.size(), input_h = x_ct_3d.at(0).size(),
                    input_w = x_ct_3d.at(0).at(0).size(), output_c = input_c,
                    output_h = input_h, output_w = input_w;

  std::cout << "\tForwarding " << layer_name() << "..." << std::endl;
  std::cout << "\t  input shape: " << input_c << "x" << input_h << "x"
            << input_w << std::endl;

#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
  for (std::size_t c = 0; c < input_c; ++c) {
    for (std::size_t h = 0; h < input_h; ++h) {
      for (std::size_t w = 0; w < input_w; ++w) {
        seal_tool_->evaluator().multiply_plain_inplace(x_ct_3d[c][h][w],
                                                       plain_weights_[c]);
        seal_tool_->evaluator().rescale_to_next_inplace(x_ct_3d[c][h][w]);
        x_ct_3d[c][h][w].scale() = seal_tool_->scale();
        seal_tool_->evaluator().add_plain_inplace(x_ct_3d[c][h][w],
                                                  plain_biases_[c]);
      }
    }
  }
}

void BatchNorm::forward(std::vector<seal::Ciphertext>& x_cts) {
  const std::size_t input_c = x_cts.size(), output_c = x_cts.size();

  std::cout << "\tForwarding " << layer_name() << "..." << std::endl;
  std::cout << "\t  input size: " << input_c << std::endl;

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (std::size_t c = 0; c < input_c; ++c) {
    seal_tool_->evaluator().multiply_plain_inplace(x_cts[c], plain_weights_[c]);
    seal_tool_->evaluator().rescale_to_next_inplace(x_cts[c]);
    x_cts[c].scale() = seal_tool_->scale();
    seal_tool_->evaluator().add_plain_inplace(x_cts[c], plain_biases_[c]);
  }
}

}  // namespace cnn::encrypted::batch
