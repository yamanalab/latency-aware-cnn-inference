#pragma once

#include "layer.hpp"

namespace cnn {

class LayerBuilder {
public:
  static std::shared_ptr<Layer> build(picojson::object& layer,
                                      const std::string& model_params_path);
};

std::shared_ptr<Layer> build_conv_2d(picojson::object& layer_info,
                                     const std::string& model_params_path);
std::shared_ptr<Layer> build_avg_pool_2d(picojson::object& layer_info,
                                         const std::string& model_params_path);
std::shared_ptr<Layer> build_activation(picojson::object& layer_info,
                                        const std::string& model_params_path);
std::shared_ptr<Layer> build_batch_norm(picojson::object& layer_info,
                                        const std::string& model_params_path);
std::shared_ptr<Layer> build_linear(picojson::object& layer_info,
                                    const std::string& model_params_path);
std::shared_ptr<Layer> build_flatten(picojson::object& layer_info,
                                     const std::string& model_params_path);

}  // namespace cnn

namespace cnn::encrypted {

class LayerBuilder {
public:
  static std::shared_ptr<Layer> build(
      picojson::object& layer,
      const std::string& model_params_path,
      const std::shared_ptr<helper::he::SealTool> seal_tool);

  static std::shared_ptr<Layer> build(
      picojson::object& layer,
      picojson::object& next_layer,
      picojson::array::const_iterator& layers_iterator,
      const std::string& model_params_path,
      const std::shared_ptr<helper::he::SealTool> seal_tool);
};

std::shared_ptr<Layer> build_conv_2d(
    picojson::object& layer_info,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool> seal_tool);
std::shared_ptr<Layer> build_avg_pool_2d(
    picojson::object& layer_info,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool> seal_tool);
std::shared_ptr<Layer> build_activation(
    picojson::object& layer_info,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool> seal_tool);
std::shared_ptr<Layer> build_batch_norm(
    picojson::object& layer_info,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool> seal_tool);
std::shared_ptr<Layer> build_linear(
    picojson::object& layer_info,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool> seal_tool);
std::shared_ptr<Layer> build_flatten(
    picojson::object& layer_info,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool> seal_tool);
std::shared_ptr<Layer> build_conv_2d_fused_batch_norm(
    picojson::object& conv_layer_info,
    picojson::object& bn_layer_info,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool> seal_tool);
std::shared_ptr<Layer> build_linear_fused_batch_norm(
    picojson::object& linear_layer_info,
    picojson::object& bn_layer_info,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool> seal_tool);

}  // namespace cnn::encrypted

namespace cnn::encrypted::batch {

class LayerBuilder {
public:
  static std::shared_ptr<Layer> build(
      picojson::object& layer,
      const std::string& model_params_path,
      const std::shared_ptr<helper::he::SealTool> seal_tool);

  static std::shared_ptr<Layer> build(
      picojson::object& layer,
      picojson::object& next_layer,
      picojson::array::const_iterator& layers_iterator,
      const std::string& model_params_path,
      const std::shared_ptr<helper::he::SealTool> seal_tool);
};

std::shared_ptr<Layer> build_conv_2d(
    picojson::object& layer_info,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool> seal_tool);
std::shared_ptr<Layer> build_avg_pool_2d(
    picojson::object& layer_info,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool> seal_tool);
std::shared_ptr<Layer> build_activation(
    picojson::object& layer_info,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool> seal_tool);
std::shared_ptr<Layer> build_batch_norm(
    picojson::object& layer_info,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool> seal_tool);
std::shared_ptr<Layer> build_linear(
    picojson::object& layer_info,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool> seal_tool);
std::shared_ptr<Layer> build_flatten(
    picojson::object& layer_info,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool> seal_tool);
std::shared_ptr<Layer> build_conv_2d_fused_batch_norm(
    picojson::object& conv_layer_info,
    picojson::object& bn_layer_info,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool> seal_tool);
std::shared_ptr<Layer> build_linear_fused_batch_norm(
    picojson::object& linear_layer_info,
    picojson::object& bn_layer_info,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool> seal_tool);

}  // namespace cnn::encrypted::batch
