#include "activation.hpp"

namespace cnn {

Activation::Activation() : Layer(ELayerType::ACTIVATION) {}
Activation::~Activation() {}

void Activation::forward(types::float4d& x) const {}

void Activation::forward(types::float2d& x) const {}

}  // namespace cnn

namespace cnn::encrypted {

Activation::Activation(const std::string layer_name,
                       const EActivationType activation_type,
                       std::vector<seal::Plaintext>& plain_poly_coeffs,
                       const std::shared_ptr<helper::he::SealTool> seal_tool)
    : Layer(ELayerType::ACTIVATION, layer_name, seal_tool),
      activation_type_(activation_type),
      plain_poly_coeffs_(plain_poly_coeffs) {
  if (activation_type == EActivationType::SQUARE) {
    CONSUMED_LEVEL++;
  } else if (activation_type == EActivationType::DEG2_POLY_APPROX) {
    if (OPT_OPTION.enable_fold_act_coeff) {
      CONSUMED_LEVEL++;
    } else {
      CONSUMED_LEVEL += 2;
    }
  } else if (activation_type == EActivationType::DEG4_POLY_APPROX) {
    if (OPT_OPTION.enable_fold_act_coeff) {
      CONSUMED_LEVEL += 2;
    } else {
      CONSUMED_LEVEL += 3;
    }
  }
}
Activation::Activation() {}
Activation::~Activation() {}

void Activation::forward(std::vector<seal::Ciphertext>& x_cts,
                         std::vector<seal::Ciphertext>& y_cts) {
  const size_t input_channel_size = x_cts.size();
  y_cts.resize(input_channel_size);

  std::cout << "\tForwarding " << layer_name() << "..." << std::endl;
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (size_t i = 0; i < input_channel_size; ++i) {
    y_cts[i] = x_cts[i];
    activate(y_cts[i]);
  }
}

void Activation::forward(seal::Ciphertext& x_ct, seal::Ciphertext& y_ct) {}

void Activation::activate(seal::Ciphertext& x) const {
  if (activation_type_ == EActivationType::SQUARE) {
    square(x);
  } else if (activation_type_ == EActivationType::DEG2_POLY_APPROX) {
    if (OPT_OPTION.enable_fold_act_coeff) {
      deg2_opt_poly_act(x);
    } else {
      deg2_poly_act(x);
    }
  } else if (activation_type_ == EActivationType::DEG4_POLY_APPROX) {
    if (OPT_OPTION.enable_fold_act_coeff) {
      deg4_opt_poly_act(x);
    } else {
      deg4_poly_act(x);
    }
  }
}

void Activation::square(seal::Ciphertext& x) const {
  /* Assume that input level is l */
  // Calculate x^2 (Level: l-1)
  seal_tool_->evaluator().square_inplace(x);
  seal_tool_->evaluator().relinearize_inplace(x, seal_tool_->relin_keys());
  seal_tool_->evaluator().rescale_to_next_inplace(x);
}

void Activation::deg2_poly_act(seal::Ciphertext& x) const {
  seal::Ciphertext x2, ax2, bx;

  /* Assume that input level is l */
  // Calculate x^2 (Level: l-1)
  seal_tool_->evaluator().square(x, x2);
  seal_tool_->evaluator().relinearize_inplace(x2, seal_tool_->relin_keys());
  seal_tool_->evaluator().rescale_to_next_inplace(x2);
  // Reduce modulus of x (Level: l-l)
  seal_tool_->evaluator().mod_switch_to_next_inplace(x);

  // Calculate ax^2 (Level: l-2)
  seal_tool_->evaluator().multiply_plain(x2, plain_poly_coeffs_[0], ax2);
  // Calculate bx (Level: l-2)
  seal_tool_->evaluator().multiply_plain_inplace(x, plain_poly_coeffs_[1]);

  // Normalize scales
  ax2.scale() = seal_tool_->scale();
  x.scale() = seal_tool_->scale();

  // Calculate ax^2 + bx + c (Level: l-2)
  seal_tool_->evaluator().add_inplace(x, ax2);
  seal_tool_->evaluator().rescale_to_next_inplace(x);
  x.scale() = seal_tool_->scale();
  seal_tool_->evaluator().add_plain_inplace(x, plain_poly_coeffs_[2]);
}

void Activation::deg2_opt_poly_act(seal::Ciphertext& x) const {
  seal::Ciphertext x2;

  /* Assume that input level is l */
  // Calculate x^2 (Level: l-1)
  seal_tool_->evaluator().square(x, x2);
  seal_tool_->evaluator().relinearize_inplace(x2, seal_tool_->relin_keys());
  // Calculate b'x (Level: l-1)
  seal_tool_->evaluator().multiply_plain_inplace(x, plain_poly_coeffs_[0]);

  // Normalize scales
  x2.scale() = seal_tool_->scale();
  x.scale() = seal_tool_->scale();

  // Calculate x^2 + b'x + c'
  seal_tool_->evaluator().add_inplace(x, x2);
  seal_tool_->evaluator().rescale_to_next_inplace(x);
  x.scale() = seal_tool_->scale();
  seal_tool_->evaluator().add_plain_inplace(x, plain_poly_coeffs_[1]);
}

void Activation::deg4_poly_act(seal::Ciphertext& x) const {
  seal::Ciphertext x2, x4;

  /* Assume that input level is l */
  // Calculate x^2 (Level: l-1)
  seal_tool_->evaluator().square(x, x2);
  seal_tool_->evaluator().relinearize_inplace(x2, seal_tool_->relin_keys());
  seal_tool_->evaluator().rescale_to_next_inplace(x2);
  // Calculate x^4 (Level: l-2)
  seal_tool_->evaluator().square(x2, x4);
  seal_tool_->evaluator().relinearize_inplace(x4, seal_tool_->relin_keys());
  seal_tool_->evaluator().rescale_to_next_inplace(x4);
  // Reduce modulus of x^2 (Level: l-2)
  seal_tool_->evaluator().mod_switch_to_next_inplace(x2);
  // Reduce modulus of x (Level: l-2)
  seal_tool_->evaluator().mod_switch_to_next_inplace(x);
  seal_tool_->evaluator().mod_switch_to_next_inplace(x);

  // Calculate ax^4 (Level: l-3)
  seal_tool_->evaluator().multiply_plain_inplace(x4, plain_poly_coeffs_[0]);
  // Calculate bx^2 (Level: l-3)
  seal_tool_->evaluator().multiply_plain_inplace(x2, plain_poly_coeffs_[1]);
  // Calculate cx (Level: l-3)
  seal_tool_->evaluator().multiply_plain_inplace(x, plain_poly_coeffs_[2]);

  // Normalize scales
  x4.scale() = seal_tool_->scale();
  x2.scale() = seal_tool_->scale();
  x.scale() = seal_tool_->scale();

  // Calculate ax^4 + bx^2 + cx + d (Level: l-3)
  seal_tool_->evaluator().add_inplace(x, x2);
  seal_tool_->evaluator().add_inplace(x, x4);
  seal_tool_->evaluator().rescale_to_next_inplace(x);
  x.scale() = seal_tool_->scale();
  seal_tool_->evaluator().add_plain_inplace(x, plain_poly_coeffs_[3]);
}

void Activation::deg4_opt_poly_act(seal::Ciphertext& x) const {
  seal::Ciphertext x2, x4;

  /* Assume that input level is l */
  // Calculate x^2 (Level: l-1)
  seal_tool_->evaluator().square(x, x2);
  seal_tool_->evaluator().relinearize_inplace(x2, seal_tool_->relin_keys());
  seal_tool_->evaluator().rescale_to_next_inplace(x2);
  // Calculate x^4 (Level: l-2)
  seal_tool_->evaluator().square(x2, x4);
  seal_tool_->evaluator().relinearize_inplace(x4, seal_tool_->relin_keys());
  // Reduce modulus of x (Level: l-1)
  seal_tool_->evaluator().mod_switch_to_next_inplace(x);

  // Calculate b'x^2 (Level: l-2)
  seal_tool_->evaluator().multiply_plain_inplace(x2, plain_poly_coeffs_[0]);
  // Calculate c'x (Level: l-2)
  seal_tool_->evaluator().multiply_plain_inplace(x, plain_poly_coeffs_[1]);

  // Normalize scales
  x4.scale() = seal_tool_->scale();
  x2.scale() = seal_tool_->scale();
  x.scale() = seal_tool_->scale();
  // Calculate x^4 + b'x^2 + c'x + d' (Level: l-2)
  seal_tool_->evaluator().add_inplace(x, x2);
  seal_tool_->evaluator().add_inplace(x, x4);
  seal_tool_->evaluator().rescale_to_next_inplace(x);
  x.scale() = seal_tool_->scale();
  seal_tool_->evaluator().add_plain_inplace(x, plain_poly_coeffs_[2]);
}

}  // namespace cnn::encrypted

namespace cnn::encrypted::batch {

Activation::Activation(const std::string layer_name,
                       const EActivationType activation_type,
                       const std::shared_ptr<helper::he::SealTool> seal_tool)
    : Layer(ELayerType::ACTIVATION, layer_name, seal_tool),
      activation_type_(activation_type) {
  if (activation_type == EActivationType::SQUARE) {
    CONSUMED_LEVEL++;
  } else {
    seal::Plaintext plain_coeff;
    std::size_t num_mod_switch;
    if (activation_type == EActivationType::DEG2_POLY_APPROX) {
      if (OPT_OPTION.enable_fold_act_coeff) {
        num_mod_switch = CONSUMED_LEVEL;
      } else {
        num_mod_switch = CONSUMED_LEVEL + 1;
      }
    } else if (activation_type == EActivationType::DEG4_POLY_APPROX) {
      if (OPT_OPTION.enable_fold_act_coeff) {
        num_mod_switch = CONSUMED_LEVEL + 1;
      } else {
        num_mod_switch = CONSUMED_LEVEL + 2;
      }
    }

    for (const double& coeff : POLY_ACT_COEFFS) {
      seal_tool->encoder().encode(coeff, seal_tool->scale(), plain_coeff);
      for (std::size_t lv = 0; lv < num_mod_switch; ++lv) {
        seal_tool->evaluator().mod_switch_to_next_inplace(plain_coeff);
      }
      plain_poly_coeffs_.push_back(plain_coeff);
    }
    seal_tool->evaluator().mod_switch_to_next_inplace(
        plain_poly_coeffs_.back());

    if (activation_type == EActivationType::DEG2_POLY_APPROX) {
      if (OPT_OPTION.enable_fold_act_coeff) {
        CONSUMED_LEVEL++;
      } else {
        CONSUMED_LEVEL += 2;
      }
    } else if (activation_type == EActivationType::DEG4_POLY_APPROX) {
      if (OPT_OPTION.enable_fold_act_coeff) {
        CONSUMED_LEVEL += 2;
      } else {
        CONSUMED_LEVEL += 3;
      }
    }
  }
}
Activation::~Activation() {}

void Activation::forward(types::Ciphertext3d& x_ct_3d) {
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
        activate(x_ct_3d[c][h][w]);
      }
    }
  }
}

void Activation::forward(std::vector<seal::Ciphertext>& x_cts) {
  const std::size_t input_c = x_cts.size(), output_c = input_c;

  std::cout << "\tForwarding " << layer_name() << "..." << std::endl;
  std::cout << "\t  input size: " << input_c << std::endl;

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (std::size_t c = 0; c < input_c; ++c) {
    activate(x_cts[c]);
  }
}

void Activation::activate(seal::Ciphertext& x) const {
  if (activation_type_ == EActivationType::SQUARE) {
    square(x);
  } else if (activation_type_ == EActivationType::DEG2_POLY_APPROX) {
    if (OPT_OPTION.enable_fold_act_coeff) {
      deg2_opt_poly_act(x);
    } else {
      deg2_poly_act(x);
    }
  } else if (activation_type_ == EActivationType::DEG4_POLY_APPROX) {
    if (OPT_OPTION.enable_fold_act_coeff) {
      deg4_opt_poly_act(x);
    } else {
      deg4_poly_act(x);
    }
  }
}

void Activation::square(seal::Ciphertext& x) const {
  /* Assume that input level is l */
  // Calculate x^2 (Level: l-1)
  seal_tool_->evaluator().square_inplace(x);
  seal_tool_->evaluator().relinearize_inplace(x, seal_tool_->relin_keys());
  seal_tool_->evaluator().rescale_to_next_inplace(x);
}

void Activation::deg2_poly_act(seal::Ciphertext& x) const {
  seal::Ciphertext x2, ax2, bx;

  /* Assume that input level is l */
  // Calculate x^2 (Level: l-1)
  seal_tool_->evaluator().square(x, x2);
  seal_tool_->evaluator().relinearize_inplace(x2, seal_tool_->relin_keys());
  seal_tool_->evaluator().rescale_to_next_inplace(x2);
  // Reduce modulus of x (Level: l-l)
  seal_tool_->evaluator().mod_switch_to_next_inplace(x);

  // Calculate ax^2 (Level: l-2)
  seal_tool_->evaluator().multiply_plain(x2, plain_poly_coeffs_[0], ax2);
  // Calculate bx (Level: l-2)
  seal_tool_->evaluator().multiply_plain_inplace(x, plain_poly_coeffs_[1]);

  // Normalize scales
  ax2.scale() = seal_tool_->scale();
  x.scale() = seal_tool_->scale();

  // Calculate ax^2 + bx + c (Level: l-2)
  seal_tool_->evaluator().add_inplace(x, ax2);
  seal_tool_->evaluator().rescale_to_next_inplace(x);
  x.scale() = seal_tool_->scale();
  seal_tool_->evaluator().add_plain_inplace(x, plain_poly_coeffs_[2]);
}

void Activation::deg2_opt_poly_act(seal::Ciphertext& x) const {
  seal::Ciphertext x2;

  /* Assume that input level is l */
  // Calculate x^2 (Level: l-1)
  seal_tool_->evaluator().square(x, x2);
  seal_tool_->evaluator().relinearize_inplace(x2, seal_tool_->relin_keys());
  // Calculate b'x (Level: l-1)
  seal_tool_->evaluator().multiply_plain_inplace(x, plain_poly_coeffs_[0]);

  // Normalize scales
  x2.scale() = seal_tool_->scale();
  x.scale() = seal_tool_->scale();

  // Calculate x^2 + b'x + c'
  seal_tool_->evaluator().add_inplace(x, x2);
  seal_tool_->evaluator().rescale_to_next_inplace(x);
  x.scale() = seal_tool_->scale();
  seal_tool_->evaluator().add_plain_inplace(x, plain_poly_coeffs_[1]);
}

void Activation::deg4_poly_act(seal::Ciphertext& x) const {
  seal::Ciphertext x2, x4;

  /* Assume that input level is l */
  // Calculate x^2 (Level: l-1)
  seal_tool_->evaluator().square(x, x2);
  seal_tool_->evaluator().relinearize_inplace(x2, seal_tool_->relin_keys());
  seal_tool_->evaluator().rescale_to_next_inplace(x2);
  // Calculate x^4 (Level: l-2)
  seal_tool_->evaluator().square(x2, x4);
  seal_tool_->evaluator().relinearize_inplace(x4, seal_tool_->relin_keys());
  seal_tool_->evaluator().rescale_to_next_inplace(x4);
  // Reduce modulus of x^2 (Level: l-2)
  seal_tool_->evaluator().mod_switch_to_next_inplace(x2);
  // Reduce modulus of x (Level: l-2)
  seal_tool_->evaluator().mod_switch_to_next_inplace(x);
  seal_tool_->evaluator().mod_switch_to_next_inplace(x);

  // Calculate ax^4 (Level: l-3)
  seal_tool_->evaluator().multiply_plain_inplace(x4, plain_poly_coeffs_[0]);
  // Calculate bx^2 (Level: l-3)
  seal_tool_->evaluator().multiply_plain_inplace(x2, plain_poly_coeffs_[1]);
  // Calculate cx (Level: l-3)
  seal_tool_->evaluator().multiply_plain_inplace(x, plain_poly_coeffs_[2]);

  // Normalize scales
  x4.scale() = seal_tool_->scale();
  x2.scale() = seal_tool_->scale();
  x.scale() = seal_tool_->scale();

  // Calculate ax^4 + bx^2 + cx + d (Level: l-3)
  seal_tool_->evaluator().add_inplace(x, x2);
  seal_tool_->evaluator().add_inplace(x, x4);
  seal_tool_->evaluator().rescale_to_next_inplace(x);
  x.scale() = seal_tool_->scale();
  seal_tool_->evaluator().add_plain_inplace(x, plain_poly_coeffs_[3]);
}

void Activation::deg4_opt_poly_act(seal::Ciphertext& x) const {
  seal::Ciphertext x2, x4;

  /* Assume that input level is l */
  // Calculate x^2 (Level: l-1)
  seal_tool_->evaluator().square(x, x2);
  seal_tool_->evaluator().relinearize_inplace(x2, seal_tool_->relin_keys());
  seal_tool_->evaluator().rescale_to_next_inplace(x2);
  // Calculate x^4 (Level: l-2)
  seal_tool_->evaluator().square(x2, x4);
  seal_tool_->evaluator().relinearize_inplace(x4, seal_tool_->relin_keys());
  // Reduce modulus of x (Level: l-1)
  seal_tool_->evaluator().mod_switch_to_next_inplace(x);

  // Calculate b'x^2 (Level: l-2)
  seal_tool_->evaluator().multiply_plain_inplace(x2, plain_poly_coeffs_[0]);
  // Calculate c'x (Level: l-2)
  seal_tool_->evaluator().multiply_plain_inplace(x, plain_poly_coeffs_[1]);

  // Normalize scales
  x4.scale() = seal_tool_->scale();
  x2.scale() = seal_tool_->scale();
  x.scale() = seal_tool_->scale();
  // Calculate x^4 + b'x^2 + c'x + d' (Level: l-2)
  seal_tool_->evaluator().add_inplace(x, x2);
  seal_tool_->evaluator().add_inplace(x, x4);
  seal_tool_->evaluator().rescale_to_next_inplace(x);
  x.scale() = seal_tool_->scale();
  seal_tool_->evaluator().add_plain_inplace(x, plain_poly_coeffs_[2]);
}

}  // namespace cnn::encrypted::batch
