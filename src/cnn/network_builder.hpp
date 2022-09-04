#pragma once

#include "network.hpp"

namespace cnn {

class NetworkBuilder {
public:
  static Network build(const std::string& model_structure_path,
                       const std::string& model_params_path);
};

}  // namespace cnn

namespace cnn::encrypted {

class NetworkBuilder {
public:
  static Network build(const std::string& model_structure_path,
                       const std::string& model_params_path,
                       const std::shared_ptr<helper::he::SealTool> seal_tool);
};

}  // namespace cnn::encrypted

namespace cnn::encrypted::batch {

class NetworkBuilder {
public:
  static Network build(const std::string& model_structure_path,
                       const std::string& model_params_path,
                       const std::shared_ptr<helper::he::SealTool> seal_tool);
};

}  // namespace cnn::encrypted::batch
