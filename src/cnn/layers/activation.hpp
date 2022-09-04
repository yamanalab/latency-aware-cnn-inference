#pragma once

#include "layer.hpp"

const std::string ACTIVATION_CLASS_NAME = "Activation";

namespace cnn {

class Activation : public Layer {
public:
  Activation();
  ~Activation();

  void forward(types::float4d& x) const override;

  void forward(types::float2d& x) const override;

private:
};

}  // namespace cnn

namespace cnn::encrypted {

class Activation : public Layer {
public:
  Activation(const std::string layer_name,
             const EActivationType activation_type,
             std::vector<seal::Plaintext>& plain_poly_coeffs,
             const std::shared_ptr<helper::he::SealTool> seal_tool);
  Activation();
  ~Activation();

  void forward(std::vector<seal::Ciphertext>& x_cts,
               std::vector<seal::Ciphertext>& y_cts) override;

  void forward(seal::Ciphertext& x_ct, seal::Ciphertext& y_ct) override;

private:
  EActivationType activation_type_;
  std::vector<seal::Plaintext> plain_poly_coeffs_;
  void activate(seal::Ciphertext& x) const;
  void square(seal::Ciphertext& x) const;
  void deg2_poly_act(seal::Ciphertext& x) const;
  void deg2_opt_poly_act(seal::Ciphertext& x) const;
  void deg4_poly_act(seal::Ciphertext& x) const;
  void deg4_opt_poly_act(seal::Ciphertext& x) const;
};

}  // namespace cnn::encrypted

namespace cnn::encrypted::batch {

class Activation : public Layer {
public:
  Activation(const std::string layer_name,
             const EActivationType activation_type,
             const std::shared_ptr<helper::he::SealTool> seal_tool);
  ~Activation();

  void forward(types::Ciphertext3d& x_ct_3d) override;

  void forward(std::vector<seal::Ciphertext>& x_cts) override;

private:
  EActivationType activation_type_;
  std::vector<seal::Plaintext> plain_poly_coeffs_;
  void activate(seal::Ciphertext& x) const;
  void square(seal::Ciphertext& x) const;
  void deg2_poly_act(seal::Ciphertext& x) const;
  void deg2_opt_poly_act(seal::Ciphertext& x) const;
  void deg4_poly_act(seal::Ciphertext& x) const;
  void deg4_opt_poly_act(seal::Ciphertext& x) const;
};

}  // namespace cnn::encrypted::batch
