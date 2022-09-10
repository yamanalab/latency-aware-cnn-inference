#include "network_builder.hpp"
#include "layers/layer_builder.hpp"
#include "utils/helper.hpp"

picojson::array load_layers(const std::string& model_structure_path) {
  picojson::object json_obj = helper::json::read_json(model_structure_path);
  return json_obj["structure"].get<picojson::array>();
}

namespace cnn {

Network NetworkBuilder::build(const std::string& model_structure_path,
                              const std::string& model_params_path) {
  Network network;

  picojson::array layers = load_layers(model_structure_path);
  for (picojson::array::const_iterator it = layers.cbegin(),
                                       layers_end = layers.cend();
       it != layers_end; ++it) {
    picojson::object layer = (*it).get<picojson::object>();
    network.add_layer(LayerBuilder::build(layer, model_params_path));
  }

  return network;
}

}  // namespace cnn

namespace cnn::encrypted {

Network NetworkBuilder::build(
    const std::string& model_structure_path,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool> seal_tool) {
  Network network;
  picojson::array layers = load_layers(model_structure_path);

  if (OPT_OPTION.enable_fuse_linear_layers) {
    for (picojson::array::const_iterator it = layers.cbegin(),
                                         layers_end = layers.cend();
         it != layers_end; ++it) {
      picojson::object layer = (*it).get<picojson::object>();

      if (it + 1 != layers_end) {
        picojson::object next_layer = (*(it + 1)).get<picojson::object>();
        network.add_layer(LayerBuilder::build(layer, next_layer, it,
                                              model_params_path, seal_tool));
      } else {
        network.add_layer(
            LayerBuilder::build(layer, model_params_path, seal_tool));
      }
      // {
      //   std::cout << "OUTPUT_HW_SLOT_IDX(" << OUTPUT_H << "x" << OUTPUT_W
      //             << "):" << std::endl;
      //   for (int i = 0; i < OUTPUT_H; ++i) {
      //     for (int j = 0; j < OUTPUT_W; ++j) {
      //       std::cout << OUTPUT_HW_SLOT_IDX[i][j] << ", ";
      //     }
      //     std::cout << std::endl;
      //   }
      // }
    }
  } else {
    for (picojson::array::const_iterator it = layers.cbegin(),
                                         layers_end = layers.cend();
         it != layers_end; ++it) {
      picojson::object layer = (*it).get<picojson::object>();
      network.add_layer(
          LayerBuilder::build(layer, model_params_path, seal_tool));
    }
  }

  return network;
}

}  // namespace cnn::encrypted

namespace cnn::encrypted::batch {

Network NetworkBuilder::build(
    const std::string& model_structure_path,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool> seal_tool) {
  Network network;
  picojson::array layers = load_layers(model_structure_path);

  if (OPT_OPTION.enable_fuse_linear_layers) {
    for (picojson::array::const_iterator it = layers.cbegin(),
                                         layers_end = layers.cend();
         it != layers_end; ++it) {
      picojson::object layer = (*it).get<picojson::object>();

      if (it + 1 != layers_end) {
        picojson::object next_layer = (*(it + 1)).get<picojson::object>();
        network.add_layer(LayerBuilder::build(layer, next_layer, it,
                                              model_params_path, seal_tool));
      } else {
        network.add_layer(
            LayerBuilder::build(layer, model_params_path, seal_tool));
      }
    }
  } else {
    for (picojson::array::const_iterator it = layers.cbegin(),
                                         layers_end = layers.cend();
         it != layers_end; ++it) {
      picojson::object layer = (*it).get<picojson::object>();
      network.add_layer(
          LayerBuilder::build(layer, model_params_path, seal_tool));
    }
  }

  return network;
}

}  // namespace cnn::encrypted::batch
