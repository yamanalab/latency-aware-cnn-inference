#include "layer.hpp"

namespace cnn {

Layer::Layer(const ELayerType& layer_type) : layer_type_(layer_type) {}
Layer::~Layer() {}

}  // namespace cnn

namespace cnn::encrypted {

Layer::Layer(const ELayerType layer_type,
             const std::string layer_name,
             const std::shared_ptr<helper::he::SealTool> seal_tool)
    : layer_type_(layer_type), layer_name_(layer_name), seal_tool_(seal_tool) {}
Layer::Layer() {}
Layer::~Layer() {}

}  // namespace cnn::encrypted

namespace cnn::encrypted::batch {

Layer::Layer(const ELayerType layer_type,
             const std::string layer_name,
             const std::shared_ptr<helper::he::SealTool> seal_tool)
    : layer_type_(layer_type), layer_name_(layer_name), seal_tool_(seal_tool) {}
Layer::Layer() {}
Layer::~Layer() {}

}  // namespace cnn::encrypted::batch
