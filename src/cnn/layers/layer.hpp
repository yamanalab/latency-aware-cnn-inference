#pragma once

#include <seal/seal.h>
#include <memory>

#include "layer_type.hpp"
#include "utils/globals.hpp"
#include "utils/helper.hpp"
#include "utils/types.hpp"

namespace cnn {

class Forwardable {  // Interface
public:
  virtual ~Forwardable() {}

  /**
   * @param x input in the form of [N, C, H, W]
   */
  virtual void forward(types::float4d& x) const = 0;
  /**
   * @param x input in the form of [N, C]
   */
  virtual void forward(types::float2d& x) const = 0;
  /**
   * @param x input in the form of [N, C, H, W]
   * @param[out] y flattened output in the form of [N, C]
   */
  virtual void forward(types::float4d& x, types::float2d& y) const = 0;
};

class Layer : public Forwardable {
public:
  Layer(const ELayerType& layer_type);
  virtual ~Layer();

  const ELayerType& layer_type() const { return layer_type_; };

  virtual void forward(types::float4d& x) const override {
    std::cerr << __PRETTY_FUNCTION__ << " is not implemented." << std::endl;
  }

  virtual void forward(types::float2d& x) const override {
    std::cerr << __PRETTY_FUNCTION__ << " is not implemented." << std::endl;
  }

  virtual void forward(types::float4d& x, types::float2d& y) const override {
    std::cerr << __PRETTY_FUNCTION__ << " is not implemented." << std::endl;
  }

private:
  ELayerType layer_type_;
};

}  // namespace cnn

namespace cnn::encrypted {

class Forwardable {  // Interface
public:
  virtual ~Forwardable() {}

  /**
   * @param x_cts input ciphertexts (size: number of input channels)
   * @param[out] y_cts output ciphertexts (size: number of output channels)
   */
  virtual void forward(std::vector<seal::Ciphertext>& x_cts,
                       std::vector<seal::Ciphertext>& y_cts) = 0;
  /**
   * @param x_ct input ciphertext
   * @param[out] y_ct output ciphertext
   */
  virtual void forward(seal::Ciphertext& x_ct, seal::Ciphertext& y_ct) = 0;
  /**
   * @param x_ct input ciphertext
   * @param[out] y_ctx output ciphertexts (size: number of output channels)
   * @note for Linear
   */
  virtual void forward(seal::Ciphertext& x_ct,
                       std::vector<seal::Ciphertext>& y_cts) = 0;
  /**
   * @param x_cts input ciphertexts (size: number of input channels)
   * @param[out] y_ct flattened output ciphertext
   */
  virtual void forward(std::vector<seal::Ciphertext>& x_cts,
                       seal::Ciphertext& y_ct) const = 0;
};

class Layer : public Forwardable {
public:
  Layer(const ELayerType layer_type,
        const std::string layer_name,
        const std::shared_ptr<helper::he::SealTool> seal_tool);
  Layer();
  virtual ~Layer();

  const ELayerType layer_type() const { return layer_type_; };
  const std::string layer_name() const { return layer_name_; };

  virtual void forward(std::vector<seal::Ciphertext>& x_cts,
                       std::vector<seal::Ciphertext>& y_cts) override {
    std::cerr << __PRETTY_FUNCTION__ << " is not implemented." << std::endl;
  }

  virtual void forward(seal::Ciphertext& x_ct,
                       seal::Ciphertext& y_ct) override {
    std::cerr << __PRETTY_FUNCTION__ << " is not implemented." << std::endl;
  }

  virtual void forward(seal::Ciphertext& x_ct,
                       std::vector<seal::Ciphertext>& y_cts) override {
    std::cerr << __PRETTY_FUNCTION__ << " is not implemented." << std::endl;
  }

  virtual void forward(std::vector<seal::Ciphertext>& x_cts,
                       seal::Ciphertext& y_ct) const override {
    std::cerr << __PRETTY_FUNCTION__ << " is not implemented." << std::endl;
  }

protected:
  ELayerType layer_type_;
  std::string layer_name_;
  std::shared_ptr<helper::he::SealTool> seal_tool_;
};

}  // namespace cnn::encrypted

namespace cnn::encrypted::batch {

class Forwardable {  // Interface
public:
  virtual ~Forwardable() {}

  /**
   * @param x_ct_3d input ciphertexts in the form of [C, H, W]
   */
  virtual void forward(types::Ciphertext3d& x_ct_3d) = 0;

  /**
   * @param x_cts input ciphertexts in the form of [UNITS]
   */
  virtual void forward(std::vector<seal::Ciphertext>& x_cts) = 0;

  /**
   * @param x_ct_3d input in the form of [C, H, W]
   * @return std::vector<seal::Ciphertext>
   *         flattened output in the form of [C * H * W]
   */
  virtual void forward(types::Ciphertext3d& x_ct_3d,
                       std::vector<seal::Ciphertext>& x_cts) const = 0;
};

class Layer : public Forwardable {
public:
  Layer(const cnn::encrypted::ELayerType layer_type,
        const std::string layer_name,
        const std::shared_ptr<helper::he::SealTool> seal_tool);
  Layer();
  virtual ~Layer();

  const ELayerType layer_type() const { return layer_type_; };
  const std::string layer_name() const { return layer_name_; };

  virtual void forward(types::Ciphertext3d& x_ct_3d) override {
    std::cerr << __PRETTY_FUNCTION__ << " is not implemented." << std::endl;
  }

  virtual void forward(std::vector<seal::Ciphertext>& x_cts) override {
    std::cerr << __PRETTY_FUNCTION__ << " is not implemented." << std::endl;
  }

  virtual void forward(types::Ciphertext3d& x_ct_3d,
                       std::vector<seal::Ciphertext>& x_cts) const override {
    std::cerr << __PRETTY_FUNCTION__ << " is not implemented." << std::endl;
  }

protected:
  cnn::encrypted::ELayerType layer_type_;
  std::string layer_name_;
  std::shared_ptr<helper::he::SealTool> seal_tool_;
};

}  // namespace cnn::encrypted::batch
