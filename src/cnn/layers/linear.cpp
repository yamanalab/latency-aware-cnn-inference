#include "linear.hpp"

namespace cnn {

Linear::Linear() : Layer(ELayerType::LINEAR) {}
Linear::~Linear() {}

void Linear::forward(types::float2d& x) const {}

}  // namespace cnn

namespace cnn::encrypted {

Linear::Linear(const std::string layer_name,
               const std::vector<seal::Plaintext>& weights_pts,
               const std::vector<seal::Plaintext>& biases_pts,
               const std::shared_ptr<helper::he::SealTool> seal_tool)
    : Layer(ELayerType::LINEAR, layer_name, seal_tool),
      weights_pts_(weights_pts),
      biases_pts_(biases_pts) {
  CONSUMED_LEVEL++;
}
Linear::~Linear() {}

void Linear::forward(seal::Ciphertext& x_ct, seal::Ciphertext& y_ct) {}

void Linear::forward(seal::Ciphertext& x_ct,
                     std::vector<seal::Ciphertext>& y_cts) {
  const std::size_t output_c = weights_pts_.size();
  y_cts.resize(output_c);

  std::cout << "\tForwarding " << layer_name() << "..." << std::endl;
  seal::Ciphertext wx_ct;
#ifdef _OPENMP
#pragma omp parallel for private(wx_ct)
#endif
  for (std::size_t i = 0; i < output_c; ++i) {
    seal_tool_->evaluator().multiply_plain(x_ct, weights_pts_[i], wx_ct);
    seal_tool_->evaluator().rescale_to_next_inplace(wx_ct);
    helper::he::total_sum(wx_ct, y_cts[i], seal_tool_->slot_count(),
                          seal_tool_->evaluator(), GALOIS_KEYS);
    y_cts[i].scale() = seal_tool_->scale();
    // if (i == 0) {
    //   seal::Plaintext plain_y;
    //   std::vector<double> y_values, bias_values;
    //   seal_tool_->decryptor().decrypt(y_cts[i], plain_y);
    //   seal_tool_->encoder().decode(plain_y, y_values);
    //   for (int s = 0; s < 10; ++s) {
    //     std::cout << "y_values[" << s << "]: " << y_values[s] << std::endl;
    //   }
    // }
    seal_tool_->evaluator().add_plain_inplace(y_cts[i], biases_pts_[i]);
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
}

}  // namespace cnn::encrypted

namespace cnn::encrypted::batch {

Linear::Linear(const std::string layer_name,
               types::Plaintext2d plain_weights,
               std::vector<seal::Plaintext> plain_biases,
               const std::shared_ptr<helper::he::SealTool> seal_tool)
    : Layer(ELayerType::LINEAR, layer_name, seal_tool),
      plain_weights_(plain_weights),
      plain_biases_(plain_biases) {
  CONSUMED_LEVEL++;
}
Linear::~Linear() {}

void Linear::forward(std::vector<seal::Ciphertext>& x_cts) {
  const std::size_t input_c = x_cts.size(), output_c = plain_weights_.size();

  std::cout << "\tForwarding " << layer_name() << "..." << std::endl;
  std::cout << "\t  input size: " << input_c << std::endl;

  std::vector<seal::Ciphertext> output(output_c);
  seal::Ciphertext weighted_unit;

#ifdef _OPENMP
#pragma omp parallel for private(weighted_unit)
#endif
  for (std::size_t oc = 0; oc < output_c; ++oc) {
    for (std::size_t ic = 0; ic < input_c; ++ic) {
      seal_tool_->evaluator().multiply_plain(x_cts[ic], plain_weights_[oc][ic],
                                             weighted_unit);
      if (ic == 0) {
        output[oc] = weighted_unit;
      } else {
        seal_tool_->evaluator().add_inplace(output[oc], weighted_unit);
      }
    }
    seal_tool_->evaluator().rescale_to_next_inplace(output[oc]);
    output[oc].scale() = seal_tool_->scale();
    seal_tool_->evaluator().add_plain_inplace(output[oc], plain_biases_[oc]);
  }

  x_cts.resize(output_c);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (size_t oc = 0; oc < output_c; ++oc) {
    x_cts[oc] = std::move(output[oc]);
  }
}

}  // namespace cnn::encrypted::batch
