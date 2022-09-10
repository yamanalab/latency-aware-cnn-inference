#include "flatten.hpp"

namespace cnn {

Flatten::Flatten() : Layer(ELayerType::FLATTEN) {}
Flatten::~Flatten() {}

void Flatten::forward(types::float4d& x, types::float2d& y) const {
  y.reserve(x.size());
  size_t units_size =
      x.at(0).size() * x.at(0).at(0).size() * x.at(0).at(0).at(0).size();
  std::vector<float> units;

  for (const auto& channels_3d : x) {
    units.reserve(units_size);
    for (const auto& channel_2d : channels_3d) {
      for (const auto& row : channel_2d) {
        for (const auto& e : row) {
          units.push_back(e);
        }
      }
    }
    y.push_back(units);
    units.clear();
  }
}

}  // namespace cnn

namespace cnn::encrypted {

Flatten::Flatten(const std::string layer_name,
                 const std::vector<int> rotation_map,
                 const std::shared_ptr<helper::he::SealTool> seal_tool)
    : Layer(ELayerType::FLATTEN, layer_name, seal_tool),
      rotation_map_(rotation_map) {}
Flatten::Flatten() {}
Flatten::~Flatten() {}

void Flatten::forward(std::vector<seal::Ciphertext>& x_cts,
                      seal::Ciphertext& y_ct) const {
  std::cout << "\tForwarding " << layer_name() << "..." << std::endl;
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (std::size_t i = 0; i < x_cts.size(); ++i) {
    seal_tool_->evaluator().rotate_vector_inplace(x_cts[i], rotation_map_[i],
                                                  GALOIS_KEYS);
  }
  seal_tool_->evaluator().add_many(x_cts, y_ct);
}

}  // namespace cnn::encrypted

namespace cnn::encrypted::batch {

Flatten::Flatten(const std::string layer_name,
                 const std::shared_ptr<helper::he::SealTool> seal_tool)
    : Layer(ELayerType::FLATTEN, layer_name, seal_tool) {}
Flatten::~Flatten() {}

void Flatten::forward(types::Ciphertext3d& x_ct_3d,
                      std::vector<seal::Ciphertext>& x_cts) const {
  const std::size_t input_c = x_ct_3d.size(), input_h = x_ct_3d.at(0).size(),
                    input_w = x_ct_3d.at(0).at(0).size(),
                    output_c = input_c * input_h * input_w;

  std::cout << "\tForwarding " << layer_name() << "..." << std::endl;
  std::cout << "\t  input shape: " << input_c << "x" << input_h << "x"
            << input_w << std::endl;

  x_cts.resize(output_c);

#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
  for (std::size_t ic = 0; ic < input_c; ++ic) {
    for (std::size_t ih = 0; ih < input_h; ++ih) {
      for (std::size_t iw = 0; iw < input_w; ++iw) {
        x_cts[ic * (input_h * input_w) + ih * input_w + iw] =
            std::move(x_ct_3d[ic][ih][iw]);
      }
    }
  }
}

}  // namespace cnn::encrypted::batch
