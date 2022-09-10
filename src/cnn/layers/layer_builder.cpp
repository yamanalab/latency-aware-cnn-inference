#include "layer_builder.hpp"
#include "activation.hpp"
#include "avg_pool_2d.hpp"
#include "batch_norm.hpp"
#include "conv_2d.hpp"
#include "flatten.hpp"
#include "linear.hpp"
#include "picojson.h"
#include "utils/globals.hpp"

#include <H5Cpp.h>
#include <stdlib.h>
#include <cassert>
#include <functional>
#include <unordered_map>

/**
 * Round target encode value when smaller than threshold (EPSILON)
 */
double round_value(double& value) {
  int sign = 1;
  if (value != 0.0) {
    sign = value / fabs(value);
  }
  value = ROUND_THRESHOLD * sign;
}

namespace cnn {

std::shared_ptr<Layer> build_conv_2d(picojson::object& layer_info,
                                     const std::string& model_params_path) {
  /* Read structure info */
  const std::string layer_name = layer_info["name"].get<std::string>();
  const std::size_t filter_size = layer_info["filters"].get<double>();
  const picojson::array filter_hw =
      layer_info["kernel_size"].get<picojson::array>();
  const std::size_t filter_h = filter_hw[0].get<double>();
  const std::size_t filter_w = filter_hw[1].get<double>();
  const picojson::array stride_hw = layer_info["stride"].get<picojson::array>();
  const std::size_t stride_h = stride_hw[0].get<double>();
  const std::size_t stride_w = stride_hw[1].get<double>();
  const picojson::array padding_hw =
      layer_info["padding"].get<picojson::array>();
  const std::size_t padding_h = padding_hw[0].get<double>();
  const std::size_t padding_w = padding_hw[1].get<double>();

  /* Read params data */
  H5::H5File params_file(model_params_path, H5F_ACC_RDONLY);
  H5::Group group = params_file.openGroup("/" + layer_name);
  H5::DataSet weight_data = group.openDataSet("weight");
  H5::DataSet bias_data = group.openDataSet("bias");

  H5::DataSpace weight_space = weight_data.getSpace();
  int weight_rank = weight_space.getSimpleExtentNdims();
  hsize_t weight_shape[weight_rank];
  int ndims = weight_space.getSimpleExtentDims(weight_shape);

  const std::size_t filter_n = weight_shape[0], in_channel = weight_shape[1],
                    filter_height = weight_shape[2],
                    filter_width = weight_shape[3];
  std::vector<float> flattened_filters(filter_n * in_channel * filter_height *
                                       filter_width);
  std::vector<float> biases(filter_n);

  weight_data.read(flattened_filters.data(), H5::PredType::NATIVE_FLOAT);
  bias_data.read(biases.data(), H5::PredType::NATIVE_FLOAT);

  types::float4d filters(
      filter_n,
      types::float3d(
          in_channel,
          types::float2d(filter_height, std::vector<float>(filter_width))));
  for (std::size_t fn = 0; fn < filter_n; ++fn) {
    for (std::size_t ic = 0; ic < in_channel; ++ic) {
      for (std::size_t fh = 0; fh < filter_height; ++fh) {
        for (std::size_t fw = 0; fw < filter_width; ++fw) {
          filters[fn][ic][fh][fw] =
              flattened_filters[fn * (in_channel * filter_height * filter_w) +
                                ic * (filter_height * filter_w) +
                                fh * filter_w + fw];
        }
      }
    }
  }

  return std::make_shared<Conv2d>(
      filters, biases, std::make_pair(stride_h, stride_w),
      std::make_pair("", std::make_pair(padding_h, padding_w)));
}

std::shared_ptr<Layer> build_avg_pool_2d(picojson::object& layer_info,
                                         const std::string& model_params_path) {
  return std::make_shared<AvgPool2d>();
}

std::shared_ptr<Layer> build_activation(picojson::object& layer_info,
                                        const std::string& model_params_path) {
  return std::make_shared<Activation>();
}

std::shared_ptr<Layer> build_batch_norm(picojson::object& layer_info,
                                        const std::string& model_params_path) {
  return std::make_shared<BatchNorm>();
}

std::shared_ptr<Layer> build_linear(picojson::object& layer_info,
                                    const std::string& model_params_path) {
  return std::make_shared<Linear>();
}

std::shared_ptr<Layer> build_flatten(picojson::object& layer_info,
                                     const std::string& model_params_path) {
  return std::make_shared<Flatten>();
}

std::unordered_map<std::string,
                   std::function<std::shared_ptr<Layer>(picojson::object&,
                                                        const std::string&)>>
    BUILD_LAYER_MAP{
        {CONV_2D_CLASS_NAME, build_conv_2d},
        {AVG_POOL_2D_CLASS_NAME, build_avg_pool_2d},
        {ACTIVATION_CLASS_NAME, build_activation},
        {BATCH_NORM_CLASS_NAME, build_batch_norm},
        {LINEAR_CLASS_NAME, build_linear},
        {FLATTEN_CLASS_NAME, build_flatten},
    };

std::shared_ptr<Layer> LayerBuilder::build(
    picojson::object& layer,
    const std::string& model_params_path) {
  const std::string layer_class_name = layer["class_name"].get<std::string>();
  if (auto map_iter = BUILD_LAYER_MAP.find(layer_class_name);
      map_iter != BUILD_LAYER_MAP.end()) {
    picojson::object layer_info = layer["info"].get<picojson::object>();
    const std::string layer_name = layer_info["name"].get<std::string>();
    std::cout << "  Building " << layer_name << "..." << std::endl;

    return map_iter->second(layer_info, model_params_path);
  } else {
    throw std::runtime_error("\"" + layer_class_name +
                             "\" is not registered as layer class");
  }
}

}  // namespace cnn

namespace cnn::encrypted {

template <typename T>
std::vector<T> flatten_2d_vector(const types::vector2d<T>& vec_2d) {
  const std::size_t height = vec_2d.size(), width = vec_2d.at(0).size();

  std::vector<T> flattened(height * width);
  for (size_t h = 0; h < height; ++h) {
    for (size_t w = 0; w < width; ++w) {
      flattened[h * width + w] = vec_2d[h][w];
    }
  }

  return flattened;
}

std::shared_ptr<Layer> build_conv_2d(
    picojson::object& layer_info,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool> seal_tool) {
  /* Read structure info */
  const std::string layer_name = layer_info["name"].get<std::string>();
  std::cout << "  Building " << layer_name << "..." << std::endl;
  const std::size_t filter_size = layer_info["filters"].get<double>();
  const picojson::array filter_hw =
      layer_info["kernel_size"].get<picojson::array>();
  const std::size_t filter_h = filter_hw[0].get<double>();
  const std::size_t filter_w = filter_hw[1].get<double>();
  const picojson::array stride_hw = layer_info["stride"].get<picojson::array>();
  const std::size_t stride_h = stride_hw[0].get<double>();
  const std::size_t stride_w = stride_hw[1].get<double>();
  std::pair<std::string, std::pair<std::size_t, std::size_t>> padding;
  std::size_t padding_h, padding_w;
  if (layer_info["padding"].is<std::string>()) {
    const std::string pad_str = layer_info["padding"].get<std::string>();
    padding_h = 0, padding_w = 0;
    padding = {pad_str, {padding_h, padding_w}};
  } else if (layer_info["padding"].is<picojson::array>()) {
    const picojson::array padding_hw =
        layer_info["padding"].get<picojson::array>();
    padding_h = padding_hw[0].get<double>();
    padding_w = padding_hw[1].get<double>();
    padding = {"", {padding_h, padding_w}};
  }

  {
    OUTPUT_C = filter_size;
    OUTPUT_H = static_cast<std::size_t>(
        ((INPUT_H + padding_h + padding_h - filter_h) / stride_h) + 1);
    OUTPUT_W = static_cast<std::size_t>(
        ((INPUT_W + padding_w + padding_w - filter_w) / stride_w) + 1);

    OUTPUT_HW_SLOT_IDX.resize(OUTPUT_H);
    for (int i = 0; i < OUTPUT_H; ++i) {
      OUTPUT_HW_SLOT_IDX[i].resize(OUTPUT_W);
      for (int j = 0; j < OUTPUT_W; ++j) {
        OUTPUT_HW_SLOT_IDX[i][j] =
            INPUT_HW_SLOT_IDX[i * stride_h][j * stride_w];
      }
    }

    int col_idx, row_idx, step;
    std::size_t stride = INPUT_HW_SLOT_IDX[0][1] - INPUT_HW_SLOT_IDX[0][0];
    KERNEL_HW_ROTATION_STEP.resize(filter_h);
    for (int i = 0; i < filter_h; ++i) {
      KERNEL_HW_ROTATION_STEP[i].resize(filter_w);
      for (int j = 0; j < filter_w; ++j) {
        col_idx = i - padding_h;
        row_idx = stride * (j - padding_w);
        if (col_idx < 0) {
          step = -INPUT_HW_SLOT_IDX[-col_idx][0] + row_idx;
        } else {
          step = INPUT_HW_SLOT_IDX[col_idx][0] + row_idx;
        }
        KERNEL_HW_ROTATION_STEP[i][j] = step;
        USE_ROTATION_STEPS.insert(step);
      }
    }
  }

  /* Read params data */
  H5::H5File params_file(model_params_path, H5F_ACC_RDONLY);
  H5::Group group = params_file.openGroup("/" + layer_name);
  H5::DataSet weight_data = group.openDataSet("weight");
  H5::DataSet bias_data = group.openDataSet("bias");

  H5::DataSpace weight_space = weight_data.getSpace();
  int weight_rank = weight_space.getSimpleExtentNdims();
  hsize_t weight_shape[weight_rank];
  int ndims = weight_space.getSimpleExtentDims(weight_shape);

  const std::size_t filter_n = weight_shape[0], in_channel = weight_shape[1],
                    filter_height = weight_shape[2],
                    filter_width = weight_shape[3];
  std::vector<float> flattened_filters(filter_n * in_channel * filter_height *
                                       filter_width);
  std::vector<float> biases(filter_n);

  weight_data.read(flattened_filters.data(), H5::PredType::NATIVE_FLOAT);
  bias_data.read(biases.data(), H5::PredType::NATIVE_FLOAT);

  double folding_value = 1, weight;
  if (OPT_OPTION.enable_fold_act_coeff && SHOULD_MUL_ACT_COEFF &&
      OPT_OPTION.enable_fold_pool_coeff && SHOULD_MUL_POOL_COEFF) {
    folding_value = POLY_ACT_HIGHEST_DEG_COEFF * CURRENT_POOL_MUL_COEFF;
    SHOULD_MUL_ACT_COEFF = false;
    SHOULD_MUL_POOL_COEFF = false;
  } else if (OPT_OPTION.enable_fold_act_coeff && SHOULD_MUL_ACT_COEFF) {
    folding_value = POLY_ACT_HIGHEST_DEG_COEFF;
    SHOULD_MUL_ACT_COEFF = false;
  } else if (OPT_OPTION.enable_fold_pool_coeff && SHOULD_MUL_POOL_COEFF) {
    folding_value = CURRENT_POOL_MUL_COEFF;
    SHOULD_MUL_POOL_COEFF = false;
  }

  // types::float4d filters(
  //     filter_n,
  //     types::float3d(
  //         in_channel,
  //         types::float2d(filter_height, std::vector<float>(filter_width))));
  types::Plaintext3d plain_filters(
      filter_n,
      types::Plaintext2d(in_channel, std::vector<seal::Plaintext>(
                                         filter_height * filter_width)));
  std::vector<seal::Plaintext> plain_biases(filter_n);

  std::size_t pos;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) private(weight, pos)
#endif
  for (std::size_t fn = 0; fn < filter_n; ++fn) {
    for (std::size_t ic = 0; ic < in_channel; ++ic) {
      for (std::size_t fh = 0; fh < filter_height; ++fh) {
        for (std::size_t fw = 0; fw < filter_width; ++fw) {
          std::vector<double> weight_values_in_slot(seal_tool->slot_count(), 0);
          pos = fh * filter_width + fw;
          // filters[fn][ic][fh][fw] =
          //     flattened_filters[fn * (in_channel * filter_height * filter_w)
          //     +
          //                       ic * (filter_height * filter_w) +
          //                       fh * filter_w + fw];
          weight =
              folding_value *
              flattened_filters[fn * (in_channel * filter_height * filter_w) +
                                ic * (filter_height * filter_w) +
                                fh * filter_w + fw];
          if (fabs(weight) < ROUND_THRESHOLD) {
            round_value(weight);
          }
          for (int i = 0; i < OUTPUT_H; ++i) {
            for (int j = 0; j < OUTPUT_W; ++j) {
              weight_values_in_slot[OUTPUT_HW_SLOT_IDX[i][j]] = weight;
            }
          }
          seal_tool->encoder().encode(weight_values_in_slot, seal_tool->scale(),
                                      plain_filters[fn][ic][pos]);
          for (std::size_t lv = 0; lv < CONSUMED_LEVEL; ++lv) {
            seal_tool->evaluator().mod_switch_to_next_inplace(
                plain_filters[fn][ic][pos]);
          }
        }
      }
    }
  }

  // std::vector<double> bias_values_in_slot(seal_tool->slot_count(), 0);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (std::size_t fn = 0; fn < filter_n; ++fn) {
    std::vector<double> bias_values_in_slot(seal_tool->slot_count(), 0);
    for (int i = 0; i < OUTPUT_H; ++i) {
      for (int j = 0; j < OUTPUT_W; ++j) {
        bias_values_in_slot[OUTPUT_HW_SLOT_IDX[i][j]] = biases[fn];
      }
    }
    seal_tool->encoder().encode(bias_values_in_slot, seal_tool->scale(),
                                plain_biases[fn]);
    for (std::size_t lv = 0; lv < CONSUMED_LEVEL + 1; ++lv) {
      seal_tool->evaluator().mod_switch_to_next_inplace(plain_biases[fn]);
    }
  }

  {
    INPUT_C = OUTPUT_C;
    INPUT_H = OUTPUT_H;
    INPUT_W = OUTPUT_W;
    INPUT_HW_SLOT_IDX.resize(OUTPUT_H);
    for (int i = 0; i < OUTPUT_H; ++i) {
      INPUT_HW_SLOT_IDX[i].resize(OUTPUT_W);
      for (int j = 0; j < OUTPUT_W; ++j) {
        INPUT_HW_SLOT_IDX[i][j] = OUTPUT_HW_SLOT_IDX[i][j];
      }
    }
  }

  return std::make_shared<Conv2d>(layer_name, plain_filters, plain_biases,
                                  flatten_2d_vector(KERNEL_HW_ROTATION_STEP),
                                  seal_tool);
}

std::shared_ptr<Layer> build_avg_pool_2d(
    picojson::object& layer_info,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool> seal_tool) {
  /* Read structure info */
  const std::string layer_name = layer_info["name"].get<std::string>();
  std::cout << "  Building " << layer_name << "..." << std::endl;
  const picojson::array pool_hw =
      layer_info["kernel_size"].get<picojson::array>();
  const std::size_t pool_height = pool_hw[0].get<double>();
  const std::size_t pool_width = pool_hw[1].get<double>();
  const picojson::array stride_hw = layer_info["stride"].get<picojson::array>();
  const std::size_t stride_height = stride_hw[0].get<double>();
  const std::size_t stride_width = stride_hw[1].get<double>();
  const picojson::array padding_hw =
      layer_info["padding"].get<picojson::array>();
  const std::size_t padding_height = padding_hw[0].get<double>();
  const std::size_t padding_width = padding_hw[1].get<double>();
  const std::pair<std::size_t, std::size_t> padding = {padding_height,
                                                       padding_width};

  {
    OUTPUT_H = static_cast<std::size_t>(
        ((INPUT_H + padding_height + padding_height - pool_height) /
         stride_height) +
        1);
    OUTPUT_W = static_cast<std::size_t>(
        ((INPUT_W + padding_width + padding_width - pool_width) /
         stride_width) +
        1);

    OUTPUT_HW_SLOT_IDX.resize(OUTPUT_H);
    for (int i = 0; i < OUTPUT_H; ++i) {
      OUTPUT_HW_SLOT_IDX[i].resize(OUTPUT_W);
      for (int j = 0; j < OUTPUT_W; ++j) {
        OUTPUT_HW_SLOT_IDX[i][j] =
            INPUT_HW_SLOT_IDX[i * stride_height][j * stride_width];
      }
    }

    int col_idx, row_idx, step;
    std::size_t stride = INPUT_HW_SLOT_IDX[0][1] - INPUT_HW_SLOT_IDX[0][0];
    KERNEL_HW_ROTATION_STEP.resize(pool_height);
    for (int i = 0; i < pool_height; ++i) {
      KERNEL_HW_ROTATION_STEP[i].resize(pool_width);
      for (int j = 0; j < pool_width; ++j) {
        col_idx = i - padding_height;
        row_idx = stride * (j - padding_width);
        if (col_idx < 0) {
          step = -INPUT_HW_SLOT_IDX[-col_idx][0] + row_idx;
        } else {
          step = INPUT_HW_SLOT_IDX[col_idx][0] + row_idx;
        }
        KERNEL_HW_ROTATION_STEP[i][j] = step;
        USE_ROTATION_STEPS.insert(step);
      }
    }
  }

  bool is_gap = false;
  seal::Plaintext plain_mul_factor;
  std::size_t pool_hw_size = pool_height * pool_width;

  if (INPUT_H == pool_height &&
      INPUT_W == pool_width) {  // GAP (Global Average Pooling)
    is_gap = true;
    double pool_mul_factor = 1.0 / pool_hw_size;
    if (OPT_OPTION.enable_fold_act_coeff && SHOULD_MUL_ACT_COEFF) {
      pool_mul_factor *= POLY_ACT_HIGHEST_DEG_COEFF;
      SHOULD_MUL_ACT_COEFF = false;
    }
    if (OPT_OPTION.enable_fold_pool_coeff && SHOULD_MUL_POOL_COEFF) {
      pool_mul_factor *= CURRENT_POOL_MUL_COEFF;
      SHOULD_MUL_POOL_COEFF = false;
    }
    std::vector<double> pool_mul_factors_in_slot(seal_tool->slot_count(), 0);
    for (int i = 0; i < OUTPUT_H; ++i) {
      for (size_t j = 0; j < OUTPUT_W; ++j) {
        pool_mul_factors_in_slot[OUTPUT_HW_SLOT_IDX[i][j]] = pool_mul_factor;
      }
    }
    seal_tool->encoder().encode(pool_mul_factors_in_slot, seal_tool->scale(),
                                plain_mul_factor);
    for (std::size_t lv = 0; lv < CONSUMED_LEVEL; ++lv) {
      seal_tool->evaluator().mod_switch_to_next_inplace(plain_mul_factor);
    }
  } else if (OPT_OPTION.enable_fold_pool_coeff && SHOULD_MUL_POOL_COEFF) {
    CURRENT_POOL_MUL_COEFF *= (1.0 / pool_hw_size);
  } else if (OPT_OPTION.enable_fold_pool_coeff) {
    CURRENT_POOL_MUL_COEFF = 1.0 / pool_hw_size;
    SHOULD_MUL_POOL_COEFF = true;
  } else if (OPT_OPTION.enable_fold_act_coeff && SHOULD_MUL_ACT_COEFF) {
    double pool_mul_factor = POLY_ACT_HIGHEST_DEG_COEFF / pool_hw_size;
    std::vector<double> pool_mul_factors_in_slot(seal_tool->slot_count(), 0);
    for (int i = 0; i < OUTPUT_H; ++i) {
      for (size_t j = 0; j < OUTPUT_W; ++j) {
        pool_mul_factors_in_slot[OUTPUT_HW_SLOT_IDX[i][j]] = pool_mul_factor;
      }
    }
    seal_tool->encoder().encode(pool_mul_factors_in_slot, seal_tool->scale(),
                                plain_mul_factor);
    SHOULD_MUL_ACT_COEFF = false;
    for (std::size_t lv = 0; lv < CONSUMED_LEVEL; ++lv) {
      seal_tool->evaluator().mod_switch_to_next_inplace(plain_mul_factor);
    }
  } else {
    double pool_mul_factor = 1.0 / pool_hw_size;
    std::vector<double> pool_mul_factors_in_slot(seal_tool->slot_count(), 0);
    for (int i = 0; i < OUTPUT_H; ++i) {
      for (size_t j = 0; j < OUTPUT_W; ++j) {
        pool_mul_factors_in_slot[OUTPUT_HW_SLOT_IDX[i][j]] = pool_mul_factor;
      }
    }
    seal_tool->encoder().encode(pool_mul_factors_in_slot, seal_tool->scale(),
                                plain_mul_factor);
    for (std::size_t lv = 0; lv < CONSUMED_LEVEL; ++lv) {
      seal_tool->evaluator().mod_switch_to_next_inplace(plain_mul_factor);
    }
  }

  {
    INPUT_H = OUTPUT_H;
    INPUT_W = OUTPUT_W;
    INPUT_HW_SLOT_IDX.resize(OUTPUT_H);
    for (int i = 0; i < OUTPUT_H; ++i) {
      INPUT_HW_SLOT_IDX[i].resize(OUTPUT_W);
      for (int j = 0; j < OUTPUT_W; ++j) {
        INPUT_HW_SLOT_IDX[i][j] = OUTPUT_HW_SLOT_IDX[i][j];
      }
    }
  }

  return std::make_shared<AvgPool2d>(layer_name, pool_hw_size, plain_mul_factor,
                                     flatten_2d_vector(KERNEL_HW_ROTATION_STEP),
                                     is_gap, seal_tool);
}

std::shared_ptr<Layer> build_activation(
    picojson::object& layer_info,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool> seal_tool) {
  /* Read structure info */
  const std::string layer_name = layer_info["name"].get<std::string>();
  const std::string function_name = layer_info["function"].get<std::string>();
  std::cout << "  Building " << layer_name << " (" << function_name << ")..."
            << std::endl;

  seal::Plaintext plain_coeff;
  std::size_t num_mod_switch;
  if (ACTIVATION_TYPE == EActivationType::DEG2_POLY_APPROX) {
    if (OPT_OPTION.enable_fold_act_coeff) {
      num_mod_switch = CONSUMED_LEVEL;
    } else {
      num_mod_switch = CONSUMED_LEVEL + 1;
    }
  } else if (ACTIVATION_TYPE == EActivationType::DEG4_POLY_APPROX) {
    if (OPT_OPTION.enable_fold_act_coeff) {
      num_mod_switch = CONSUMED_LEVEL + 1;
    } else {
      num_mod_switch = CONSUMED_LEVEL + 2;
    }
  }

  std::vector<seal::Plaintext> plain_poly_coeffs;
  if (ACTIVATION_TYPE == EActivationType::DEG2_POLY_APPROX ||
      ACTIVATION_TYPE == EActivationType::DEG4_POLY_APPROX) {
    for (const double& coeff : POLY_ACT_COEFFS) {
      std::vector<double> coeff_values_in_slot(seal_tool->slot_count(), 0);
      for (int i = 0; i < INPUT_H; ++i) {
        for (int j = 0; j < INPUT_W; ++j) {
          coeff_values_in_slot[INPUT_HW_SLOT_IDX[i][j]] = coeff;
        }
      }
      seal_tool->encoder().encode(coeff_values_in_slot, seal_tool->scale(),
                                  plain_coeff);
      for (std::size_t lv = 0; lv < num_mod_switch; ++lv) {
        seal_tool->evaluator().mod_switch_to_next_inplace(plain_coeff);
      }
      plain_poly_coeffs.push_back(plain_coeff);
    }
    seal_tool->evaluator().mod_switch_to_next_inplace(plain_poly_coeffs.back());
  }

  if ((ACTIVATION_TYPE == EActivationType::DEG2_POLY_APPROX ||
       ACTIVATION_TYPE == EActivationType::DEG4_POLY_APPROX) &&
      OPT_OPTION.enable_fold_act_coeff) {
    SHOULD_MUL_ACT_COEFF = true;
  }

  return std::make_shared<Activation>(layer_name, ACTIVATION_TYPE,
                                      plain_poly_coeffs, seal_tool);
}

std::shared_ptr<Layer> build_batch_norm(
    picojson::object& layer_info,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool> seal_tool) {
  /* Read structure info */
  const std::string layer_name = layer_info["name"].get<std::string>();
  std::cout << "  Building " << layer_name << "..." << std::endl;
  const std::string eps_str = layer_info["eps"].get<std::string>();
  const double eps = std::atof(eps_str.c_str());

  /* Read params data */
  H5::H5File params_file(model_params_path, H5F_ACC_RDONLY);
  H5::Group group = params_file.openGroup("/" + layer_name);
  H5::DataSet gamma_data = group.openDataSet("weight");
  H5::DataSet beta_data = group.openDataSet("bias");
  H5::DataSet running_mean_data = group.openDataSet("running_mean");
  H5::DataSet running_var_data = group.openDataSet("running_var");

  H5::DataSpace gamma_space = gamma_data.getSpace();
  int gamma_rank = gamma_space.getSimpleExtentNdims();
  hsize_t gamma_shape[gamma_rank];
  int ndims = gamma_space.getSimpleExtentDims(gamma_shape);

  const std::size_t num_features = gamma_shape[0];

  std::vector<float> gammas(num_features), betas(num_features),
      running_means(num_features), running_vars(num_features);

  gamma_data.read(gammas.data(), H5::PredType::NATIVE_FLOAT);
  beta_data.read(betas.data(), H5::PredType::NATIVE_FLOAT);
  running_mean_data.read(running_means.data(), H5::PredType::NATIVE_FLOAT);
  running_var_data.read(running_vars.data(), H5::PredType::NATIVE_FLOAT);

  double weight, bias;
  std::vector<double> weight_values_in_slot(seal_tool->slot_count(), 0),
      bias_values_in_slot(seal_tool->slot_count(), 0);
  std::vector<seal::Plaintext> plain_weights(num_features),
      plain_biases(num_features);

#ifdef _OPENMP
#pragma omp parallel for private(weight, bias)
#endif
  for (size_t i = 0; i < num_features; ++i) {
    weight = gammas[i] / std::sqrt(running_vars[i] + eps);
    bias = betas[i] - (weight * running_means[i]);
    for (int i = 0; i < INPUT_H; ++i) {
      for (int j = 0; j < INPUT_W; ++j) {
        weight_values_in_slot[INPUT_HW_SLOT_IDX[i][j]] = weight;
        bias_values_in_slot[INPUT_HW_SLOT_IDX[i][j]] = bias;
      }
    }

    seal_tool->encoder().encode(weight_values_in_slot, seal_tool->scale(),
                                plain_weights[i]);
    seal_tool->encoder().encode(bias_values_in_slot, seal_tool->scale(),
                                plain_biases[i]);
    for (size_t lv = 0; lv < CONSUMED_LEVEL; ++lv) {
      seal_tool->evaluator().mod_switch_to_next_inplace(plain_weights[i]);
      seal_tool->evaluator().mod_switch_to_next_inplace(plain_biases[i]);
    }
    seal_tool->evaluator().mod_switch_to_next_inplace(plain_biases[i]);
  }

  return std::make_shared<BatchNorm>(layer_name, plain_weights, plain_biases,
                                     seal_tool);
}

std::shared_ptr<Layer> build_linear(
    picojson::object& layer_info,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool> seal_tool) {
  /* Read structure info */
  const std::string layer_name = layer_info["name"].get<std::string>();
  std::cout << "  Building " << layer_name << "..." << std::endl;
  const std::size_t unit_size = layer_info["units"].get<double>();

  { OUTPUT_UNITS = unit_size; }

  /* Read params data */
  H5::H5File params_file(model_params_path, H5F_ACC_RDONLY);
  H5::Group group = params_file.openGroup("/" + layer_name);
  H5::DataSet weight_data = group.openDataSet("weight");
  H5::DataSet bias_data = group.openDataSet("bias");

  H5::DataSpace weight_space = weight_data.getSpace();
  int weight_rank = weight_space.getSimpleExtentNdims();
  hsize_t weight_shape[weight_rank];
  int ndims = weight_space.getSimpleExtentDims(weight_shape);

  const std::size_t out_channel = weight_shape[0], in_channel = weight_shape[1];
  std::vector<float> flattened_weights(out_channel * in_channel);
  std::vector<float> biases(out_channel);

  weight_data.read(flattened_weights.data(), H5::PredType::NATIVE_FLOAT);
  bias_data.read(biases.data(), H5::PredType::NATIVE_FLOAT);

  double folding_value = 1, weight;
  if (OPT_OPTION.enable_fold_act_coeff && SHOULD_MUL_ACT_COEFF &&
      OPT_OPTION.enable_fold_pool_coeff && SHOULD_MUL_POOL_COEFF) {
    folding_value = POLY_ACT_HIGHEST_DEG_COEFF * CURRENT_POOL_MUL_COEFF;
    SHOULD_MUL_ACT_COEFF = false;
    SHOULD_MUL_POOL_COEFF = false;
  } else if (OPT_OPTION.enable_fold_act_coeff && SHOULD_MUL_ACT_COEFF) {
    folding_value = POLY_ACT_HIGHEST_DEG_COEFF;
    SHOULD_MUL_ACT_COEFF = false;
  } else if (OPT_OPTION.enable_fold_pool_coeff && SHOULD_MUL_POOL_COEFF) {
    folding_value = CURRENT_POOL_MUL_COEFF;
    SHOULD_MUL_POOL_COEFF = false;
  }

  std::vector<seal::Plaintext> plain_weights(out_channel),
      plain_biases(out_channel);

  int slot_idx;
  size_t pos;
#ifdef _OPENMP
#pragma omp parallel for private(weight, slot_idx, pos)
#endif
  for (std::size_t oc = 0; oc < out_channel; ++oc) {
    std::vector<double> weight_values_in_slot(seal_tool->slot_count(), 0);
    for (std::size_t c = 0; c < INPUT_C; ++c) {
      for (std::size_t h = 0; h < INPUT_H; ++h) {
        for (std::size_t w = 0; w < INPUT_W; ++w) {
          pos = c * (INPUT_H * INPUT_W) + h * INPUT_W + w;
          slot_idx = INPUT_HW_SLOT_IDX[h][w] + (-FLATTEN_ROTATION_STEP[c]);
          weight = folding_value * flattened_weights[oc * in_channel + pos];
          if (fabs(weight) < ROUND_THRESHOLD) {
            round_value(weight);
          }
          weight_values_in_slot[slot_idx] = weight;
        }
      }
    }
    seal_tool->encoder().encode(weight_values_in_slot, seal_tool->scale(),
                                plain_weights[oc]);
    for (std::size_t lv = 0; lv < CONSUMED_LEVEL; ++lv) {
      seal_tool->evaluator().mod_switch_to_next_inplace(plain_weights[oc]);
    }
  }

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (std::size_t oc = 0; oc < out_channel; ++oc) {
    seal_tool->encoder().encode(biases[oc], seal_tool->scale(),
                                plain_biases[oc]);
    for (size_t lv = 0; lv < CONSUMED_LEVEL + 1; ++lv) {
      seal_tool->evaluator().mod_switch_to_next_inplace(plain_biases[oc]);
    }
  }

  return std::make_shared<Linear>(layer_name, plain_weights, plain_biases,
                                  seal_tool);
}

std::shared_ptr<Layer> build_flatten(
    picojson::object& layer_info,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool> seal_tool) {
  /* Read structure info */
  const std::string layer_name = layer_info["name"].get<std::string>();
  std::cout << "  Building " << layer_name << "..." << std::endl;

  const std::size_t step_col = INPUT_HW_SLOT_IDX[0][1] -
                               INPUT_HW_SLOT_IDX[0][0],              // 4, 8
      step_row = INPUT_HW_SLOT_IDX[1][0] - INPUT_HW_SLOT_IDX[0][0];  // 112, 256
  const std::size_t slot_size_per_period = step_col * INPUT_W;       // 16, 32
  const std::size_t ct_size_gap_full_period = step_row / INPUT_W;    // 28, 64
  const std::size_t slot_size_gap_full_period =
      INPUT_HW_SLOT_IDX[INPUT_H - 1][INPUT_W - 1] +
      ((ct_size_gap_full_period - 1) / step_col) * slot_size_per_period +
      ((ct_size_gap_full_period - 1) % step_col) + 1;  // 448, 1024
  // {
  //   std::cout << "INPUT_H: " << INPUT_H << std::endl;
  //   std::cout << "INPUT_W: " << INPUT_W << std::endl;
  //   std::cout << "INPUT_C: " << INPUT_C << std::endl;
  //   std::cout << "step_col: " << step_col << std::endl;
  //   std::cout << "step_row: " << step_row << std::endl;
  //   std::cout << "slot_size_per_period: " << slot_size_per_period <<
  //   std::endl; std::cout << "ct_size_gap_full_period: " <<
  //   ct_size_gap_full_period
  //             << std::endl;
  //   std::cout << "slot_size_gap_full_period: " << slot_size_gap_full_period
  //             << std::endl;
  // }
  if (INPUT_H == 1 && INPUT_W == 1) {  // in case of GAP
    FLATTEN_ROTATION_STEP.resize(INPUT_C);
    for (int i = 0; i < INPUT_C; ++i) {
      FLATTEN_ROTATION_STEP[i] = -1 * i;
      USE_ROTATION_STEPS.insert(-1 * i);
    }
  } else {
    int step;
    FLATTEN_ROTATION_STEP.resize(INPUT_C);
    for (int i = 0; i < INPUT_C; ++i) {
      step =
          -((((i % ct_size_gap_full_period) / step_col) * slot_size_per_period +
             i % step_col) +
            ((i / ct_size_gap_full_period) * slot_size_gap_full_period));
      FLATTEN_ROTATION_STEP[i] = step;
      USE_ROTATION_STEPS.insert(step);
    }
  }

  { INPUT_UNITS = INPUT_C * INPUT_H * INPUT_W; }

  return std::make_shared<Flatten>(layer_name, FLATTEN_ROTATION_STEP,
                                   seal_tool);
}

std::shared_ptr<Layer> build_conv_2d_fused_batch_norm(
    picojson::object& conv_layer_info,
    picojson::object& bn_layer_info,
    const std::string& layer_name,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool> seal_tool) {
  /* Read Conv2d info */
  const std::string conv_layer_name =
      conv_layer_info["name"].get<std::string>();
  const std::size_t filter_size = conv_layer_info["filters"].get<double>();
  const picojson::array filter_hw =
      conv_layer_info["kernel_size"].get<picojson::array>();
  const std::size_t filter_h = filter_hw[0].get<double>();
  const std::size_t filter_w = filter_hw[1].get<double>();
  const picojson::array stride_hw =
      conv_layer_info["stride"].get<picojson::array>();
  const std::size_t stride_h = stride_hw[0].get<double>();
  const std::size_t stride_w = stride_hw[1].get<double>();
  std::pair<std::string, std::pair<std::size_t, std::size_t>> padding;
  std::size_t padding_h, padding_w;
  if (conv_layer_info["padding"].is<std::string>()) {
    const std::string pad_str = conv_layer_info["padding"].get<std::string>();
    padding_h = 0, padding_w = 0;
    padding = {pad_str, {padding_h, padding_w}};
  } else if (conv_layer_info["padding"].is<picojson::array>()) {
    const picojson::array padding_hw =
        conv_layer_info["padding"].get<picojson::array>();
    padding_h = padding_hw[0].get<double>();
    padding_w = padding_hw[1].get<double>();
    padding = {"", {padding_h, padding_w}};
  }

  {
    OUTPUT_C = filter_size;
    OUTPUT_H = static_cast<std::size_t>(
        ((INPUT_H + padding_h + padding_h - filter_h) / stride_h) + 1);
    OUTPUT_W = static_cast<std::size_t>(
        ((INPUT_W + padding_w + padding_w - filter_w) / stride_w) + 1);

    OUTPUT_HW_SLOT_IDX.resize(OUTPUT_H);
    for (int i = 0; i < OUTPUT_H; ++i) {
      OUTPUT_HW_SLOT_IDX[i].resize(OUTPUT_W);
      for (int j = 0; j < OUTPUT_W; ++j) {
        OUTPUT_HW_SLOT_IDX[i][j] =
            INPUT_HW_SLOT_IDX[i * stride_h][j * stride_w];
      }
    }

    int col_idx, row_idx, step;
    std::size_t stride = INPUT_HW_SLOT_IDX[0][1] - INPUT_HW_SLOT_IDX[0][0];
    KERNEL_HW_ROTATION_STEP.resize(filter_h);
    for (int i = 0; i < filter_h; ++i) {
      KERNEL_HW_ROTATION_STEP[i].resize(filter_w);
      for (int j = 0; j < filter_w; ++j) {
        col_idx = i - padding_h;
        row_idx = stride * (j - padding_w);
        if (col_idx < 0) {
          step = -INPUT_HW_SLOT_IDX[-col_idx][0] + row_idx;
        } else {
          step = INPUT_HW_SLOT_IDX[col_idx][0] + row_idx;
        }
        KERNEL_HW_ROTATION_STEP[i][j] = step;
        USE_ROTATION_STEPS.insert(step);
      }
    }
  }

  // Read params of Conv2d
  H5::H5File params_file(model_params_path, H5F_ACC_RDONLY);
  H5::Group conv_group = params_file.openGroup("/" + conv_layer_name);
  H5::DataSet conv_weight_data = conv_group.openDataSet("weight");
  H5::DataSet conv_bias_data = conv_group.openDataSet("bias");

  H5::DataSpace conv_weight_space = conv_weight_data.getSpace();
  int conv_weight_rank = conv_weight_space.getSimpleExtentNdims();
  hsize_t conv_weight_shape[conv_weight_rank];
  int conv_ndims = conv_weight_space.getSimpleExtentDims(conv_weight_shape);

  const std::size_t filter_n = conv_weight_shape[0],
                    in_channel = conv_weight_shape[1],
                    filter_height = conv_weight_shape[2],
                    filter_width = conv_weight_shape[3];
  std::vector<float> conv_flattened_filters(filter_n * in_channel *
                                            filter_height * filter_width);
  std::vector<float> conv_biases(filter_n);

  conv_weight_data.read(conv_flattened_filters.data(),
                        H5::PredType::NATIVE_FLOAT);
  conv_bias_data.read(conv_biases.data(), H5::PredType::NATIVE_FLOAT);

  /* Read BatchNorm info */
  const std::string bn_layer_name = bn_layer_info["name"].get<std::string>();
  const std::string eps_str = bn_layer_info["eps"].get<std::string>();
  const double eps = std::atof(eps_str.c_str());

  // Read params of BatchNorm
  H5::Group bn_group = params_file.openGroup("/" + bn_layer_name);
  H5::DataSet bn_gamma_data = bn_group.openDataSet("weight");
  H5::DataSet bn_beta_data = bn_group.openDataSet("bias");
  H5::DataSet bn_running_mean_data = bn_group.openDataSet("running_mean");
  H5::DataSet bn_running_var_data = bn_group.openDataSet("running_var");

  H5::DataSpace bn_gamma_space = bn_gamma_data.getSpace();
  int bn_gamma_rank = bn_gamma_space.getSimpleExtentNdims();
  hsize_t bn_gamma_shape[bn_gamma_rank];
  int bn_ndims = bn_gamma_space.getSimpleExtentDims(bn_gamma_shape);

  const std::size_t num_features = bn_gamma_shape[0];

  std::vector<float> bn_gammas(num_features), bn_betas(num_features),
      bn_running_means(num_features), bn_running_vars(num_features);

  bn_gamma_data.read(bn_gammas.data(), H5::PredType::NATIVE_FLOAT);
  bn_beta_data.read(bn_betas.data(), H5::PredType::NATIVE_FLOAT);
  bn_running_mean_data.read(bn_running_means.data(),
                            H5::PredType::NATIVE_FLOAT);
  bn_running_var_data.read(bn_running_vars.data(), H5::PredType::NATIVE_FLOAT);

  /* Calculate fused weights & biases */
  assert(filter_n == num_features);

  double folding_value = 1;
  if (OPT_OPTION.enable_fold_act_coeff && SHOULD_MUL_ACT_COEFF &&
      OPT_OPTION.enable_fold_pool_coeff && SHOULD_MUL_POOL_COEFF) {
    folding_value = POLY_ACT_HIGHEST_DEG_COEFF * CURRENT_POOL_MUL_COEFF;
    SHOULD_MUL_ACT_COEFF = false;
    SHOULD_MUL_POOL_COEFF = false;
  } else if (OPT_OPTION.enable_fold_act_coeff && SHOULD_MUL_ACT_COEFF) {
    folding_value = POLY_ACT_HIGHEST_DEG_COEFF;
    SHOULD_MUL_ACT_COEFF = false;
  } else if (OPT_OPTION.enable_fold_pool_coeff && SHOULD_MUL_POOL_COEFF) {
    folding_value = CURRENT_POOL_MUL_COEFF;
    SHOULD_MUL_POOL_COEFF = false;
  }

  std::vector<double> bn_weights(filter_n), bn_biases(filter_n);
  double fused_weight, fused_bias;
  types::Plaintext3d plain_filters(
      filter_n,
      types::Plaintext2d(in_channel, std::vector<seal::Plaintext>(
                                         filter_height * filter_width)));
  std::vector<seal::Plaintext> plain_biases(filter_n);

  // std::vector<double> bias_values_in_slot(seal_tool->slot_count(), 0);
  // plain_biases
#ifdef _OPENMP
#pragma omp parallel for private(fused_bias)
#endif
  for (std::size_t fn = 0; fn < filter_n; ++fn) {
    std::vector<double> bias_values_in_slot(seal_tool->slot_count(), 0);
    bn_weights[fn] = bn_gammas[fn] / std::sqrt(bn_running_vars[fn] + eps);
    bn_biases[fn] = bn_betas[fn] - (bn_weights[fn] * bn_running_means[fn]);
    fused_bias = conv_biases[fn] * bn_weights[fn] + bn_biases[fn];
    for (int i = 0; i < OUTPUT_H; ++i) {
      for (int j = 0; j < OUTPUT_W; ++j) {
        bias_values_in_slot[OUTPUT_HW_SLOT_IDX[i][j]] = fused_bias;
      }
    }
    seal_tool->encoder().encode(bias_values_in_slot, seal_tool->scale(),
                                plain_biases[fn]);
    for (std::size_t lv = 0; lv < CONSUMED_LEVEL + 1; ++lv) {
      seal_tool->evaluator().mod_switch_to_next_inplace(plain_biases[fn]);
    }
  }

  // std::vector<double> weight_values_in_slot(seal_tool->slot_count(), 0);
  std::size_t pos;
  // plain_weights
#ifdef _OPENMP
#pragma omp parallel for collapse(2) private(fused_weight, pos)
#endif
  for (std::size_t fn = 0; fn < filter_n; ++fn) {
    for (std::size_t ic = 0; ic < in_channel; ++ic) {
      for (std::size_t fh = 0; fh < filter_height; ++fh) {
        for (std::size_t fw = 0; fw < filter_width; ++fw) {
          std::vector<double> weight_values_in_slot(seal_tool->slot_count(), 0);
          pos = fh * filter_width + fw;
          fused_weight =
              folding_value *
              conv_flattened_filters[fn * (in_channel * filter_height *
                                           filter_w) +
                                     ic * (filter_height * filter_w) +
                                     fh * filter_w + fw] *
              bn_weights[fn];
          if (fabs(fused_weight) < ROUND_THRESHOLD) {
            // std::cout << "fused_weight: " << fused_weight << std::endl;
            round_value(fused_weight);
          }
          if (INPUT_H == OUTPUT_H && INPUT_W == OUTPUT_W) {
            int rotation_step = KERNEL_HW_ROTATION_STEP[fh][fw];
            for (int i = std::max(0, 1 - static_cast<int>(fh));
                 i < OUTPUT_H - std::max(0, static_cast<int>(fh) - 1); ++i) {
              for (int j = std::max(0, 1 - static_cast<int>(fw));
                   j < OUTPUT_W - std::max(0, static_cast<int>(fw) - 1); ++j) {
                weight_values_in_slot[OUTPUT_HW_SLOT_IDX[i][j]] = fused_weight;
              }
            }
          } else {
            for (int i = 0; i < OUTPUT_H; ++i) {
              for (int j = 0; j < OUTPUT_W; ++j) {
                weight_values_in_slot[OUTPUT_HW_SLOT_IDX[i][j]] = fused_weight;
              }
            }
          }
          seal_tool->encoder().encode(weight_values_in_slot, seal_tool->scale(),
                                      plain_filters[fn][ic][pos]);
          for (std::size_t lv = 0; lv < CONSUMED_LEVEL; ++lv) {
            seal_tool->evaluator().mod_switch_to_next_inplace(
                plain_filters[fn][ic][pos]);
          }
        }
      }
    }
  }

  {
    INPUT_C = OUTPUT_C;
    INPUT_H = OUTPUT_H;
    INPUT_W = OUTPUT_W;
    INPUT_HW_SLOT_IDX.resize(OUTPUT_H);
    for (int i = 0; i < OUTPUT_H; ++i) {
      INPUT_HW_SLOT_IDX[i].resize(OUTPUT_W);
      for (int j = 0; j < OUTPUT_W; ++j) {
        INPUT_HW_SLOT_IDX[i][j] = OUTPUT_HW_SLOT_IDX[i][j];
      }
    }
  }

  return std::make_shared<Conv2d>(layer_name, plain_filters, plain_biases,
                                  flatten_2d_vector(KERNEL_HW_ROTATION_STEP),
                                  seal_tool);
}

std::shared_ptr<Layer> build_linear_fused_batch_norm(
    picojson::object& linear_layer_info,
    picojson::object& bn_layer_info,
    const std::string& layer_name,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool> seal_tool) {
  /* Read Linear info */
  const std::string linear_layer_name =
      linear_layer_info["name"].get<std::string>();
  const std::size_t unit_size = linear_layer_info["units"].get<double>();

  /* Read params of Linear */
  H5::H5File params_file(model_params_path, H5F_ACC_RDONLY);
  H5::Group group = params_file.openGroup("/" + linear_layer_name);
  H5::DataSet linear_weight_data = group.openDataSet("weight");
  H5::DataSet linear_bias_data = group.openDataSet("bias");

  H5::DataSpace linear_weight_space = linear_weight_data.getSpace();
  int linear_weight_rank = linear_weight_space.getSimpleExtentNdims();
  hsize_t linear_weight_shape[linear_weight_rank];
  int linear_ndims =
      linear_weight_space.getSimpleExtentDims(linear_weight_shape);

  const std::size_t out_channel = linear_weight_shape[0],
                    in_channel = linear_weight_shape[1];
  std::vector<float> linear_flattened_weights(out_channel * in_channel);
  std::vector<float> linear_biases(out_channel);

  linear_weight_data.read(linear_flattened_weights.data(),
                          H5::PredType::NATIVE_FLOAT);
  linear_bias_data.read(linear_biases.data(), H5::PredType::NATIVE_FLOAT);

  /* Read BatchNorm info */
  const std::string bn_layer_name = bn_layer_info["name"].get<std::string>();
  const std::string eps_str = bn_layer_info["eps"].get<std::string>();
  const double eps = std::atof(eps_str.c_str());

  // Read params of BatchNorm
  H5::Group bn_group = params_file.openGroup("/" + bn_layer_name);
  H5::DataSet bn_gamma_data = bn_group.openDataSet("weight");
  H5::DataSet bn_beta_data = bn_group.openDataSet("bias");
  H5::DataSet bn_running_mean_data = bn_group.openDataSet("running_mean");
  H5::DataSet bn_running_var_data = bn_group.openDataSet("running_var");

  H5::DataSpace bn_gamma_space = bn_gamma_data.getSpace();
  int bn_gamma_rank = bn_gamma_space.getSimpleExtentNdims();
  hsize_t bn_gamma_shape[bn_gamma_rank];
  int bn_ndims = bn_gamma_space.getSimpleExtentDims(bn_gamma_shape);

  const std::size_t num_features = bn_gamma_shape[0];

  std::vector<float> bn_gammas(num_features), bn_betas(num_features),
      bn_running_means(num_features), bn_running_vars(num_features);

  bn_gamma_data.read(bn_gammas.data(), H5::PredType::NATIVE_FLOAT);
  bn_beta_data.read(bn_betas.data(), H5::PredType::NATIVE_FLOAT);
  bn_running_mean_data.read(bn_running_means.data(),
                            H5::PredType::NATIVE_FLOAT);
  bn_running_var_data.read(bn_running_vars.data(), H5::PredType::NATIVE_FLOAT);

  /* Calculate fused weights & biases */
  assert(out_channel == num_features);

  double folding_value = 1;
  if (OPT_OPTION.enable_fold_act_coeff && SHOULD_MUL_ACT_COEFF &&
      OPT_OPTION.enable_fold_pool_coeff && SHOULD_MUL_POOL_COEFF) {
    folding_value = POLY_ACT_HIGHEST_DEG_COEFF * CURRENT_POOL_MUL_COEFF;
    SHOULD_MUL_ACT_COEFF = false;
    SHOULD_MUL_POOL_COEFF = false;
  } else if (OPT_OPTION.enable_fold_act_coeff && SHOULD_MUL_ACT_COEFF) {
    folding_value = POLY_ACT_HIGHEST_DEG_COEFF;
    SHOULD_MUL_ACT_COEFF = false;
  } else if (OPT_OPTION.enable_fold_pool_coeff && SHOULD_MUL_POOL_COEFF) {
    folding_value = CURRENT_POOL_MUL_COEFF;
    SHOULD_MUL_POOL_COEFF = false;
  }

  std::vector<double> bn_weights(num_features), bn_biases(num_features);
  double fused_weight, fused_bias;
  std::vector<double> weight_values_in_slot(seal_tool->slot_count(), 0);
  std::vector<seal::Plaintext> plain_weights(out_channel),
      plain_biases(out_channel);

  // plain_biases
#ifdef _OPENMP
#pragma omp parallel for private(fused_bias)
#endif
  for (std::size_t oc = 0; oc < out_channel; ++oc) {
    bn_weights[oc] = bn_gammas[oc] / std::sqrt(bn_running_vars[oc] + eps);
    bn_biases[oc] = bn_betas[oc] - (bn_weights[oc] * bn_running_means[oc]);
    fused_bias = linear_biases[oc] * bn_weights[oc] + bn_biases[oc];
    seal_tool->encoder().encode(fused_bias, seal_tool->scale(),
                                plain_biases[oc]);
    for (std::size_t lv = 0; lv < CONSUMED_LEVEL + 1; ++lv) {
      seal_tool->evaluator().mod_switch_to_next_inplace(plain_biases[oc]);
    }
  }

  // plain_weights
  int slot_idx, counter;
#ifdef _OPENMP
#pragma omp parallel for private(fused_weight, slot_idx, counter)
#endif
  for (std::size_t oc = 0; oc < out_channel; ++oc) {
    counter = 0;
    for (std::size_t c = 0; c < INPUT_C; ++c) {
      for (std::size_t h = 0; h < INPUT_H; ++h) {
        for (std::size_t w = 0; w < INPUT_W; ++w) {
          slot_idx = INPUT_HW_SLOT_IDX[h][w] + (-FLATTEN_ROTATION_STEP[c]);
          fused_weight = folding_value *
                         linear_flattened_weights[oc * in_channel + counter] *
                         bn_weights[oc];
          if (fabs(fused_weight) < ROUND_THRESHOLD) {
            // std::cout << "fused_weight: " << fused_weight << std::endl;
            round_value(fused_weight);
          }
          weight_values_in_slot[slot_idx] = fused_weight;
          counter++;
        }
      }
    }
    seal_tool->encoder().encode(weight_values_in_slot, seal_tool->scale(),
                                plain_weights[oc]);
    for (std::size_t lv = 0; lv < CONSUMED_LEVEL; ++lv) {
      seal_tool->evaluator().mod_switch_to_next_inplace(plain_weights[oc]);
    }
  }

  return std::make_shared<Linear>(layer_name, plain_weights, plain_biases,
                                  seal_tool);
}

const std::unordered_map<std::string,
                         std::function<std::shared_ptr<Layer>(
                             picojson::object&,
                             const std::string&,
                             const std::shared_ptr<helper::he::SealTool>)>>
    BUILD_LAYER_MAP{{CONV_2D_CLASS_NAME, build_conv_2d},
                    {AVG_POOL_2D_CLASS_NAME, build_avg_pool_2d},
                    {ACTIVATION_CLASS_NAME, build_activation},
                    {BATCH_NORM_CLASS_NAME, build_batch_norm},
                    {LINEAR_CLASS_NAME, build_linear},
                    {FLATTEN_CLASS_NAME, build_flatten}};

std::shared_ptr<Layer> LayerBuilder::build(
    picojson::object& layer,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool> seal_tool) {
  const std::string layer_class_name = layer["class_name"].get<std::string>();
  if (auto map_iter = BUILD_LAYER_MAP.find(layer_class_name);
      map_iter != BUILD_LAYER_MAP.end()) {
    picojson::object layer_info = layer["info"].get<picojson::object>();

    return map_iter->second(layer_info, model_params_path, seal_tool);
  } else {
    throw std::runtime_error("\"" + layer_class_name +
                             "\" is not registered as layer class");
  }
}

std::shared_ptr<Layer> LayerBuilder::build(
    picojson::object& layer,
    picojson::object& next_layer,
    picojson::array::const_iterator& layers_iterator,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool> seal_tool) {
  const std::string layer_class_name = layer["class_name"].get<std::string>();
  const std::string next_layer_class_name =
      next_layer["class_name"].get<std::string>();

  if (layer_class_name == CONV_2D_CLASS_NAME &&
      next_layer_class_name == BATCH_NORM_CLASS_NAME) {
    picojson::object conv_layer_info = layer["info"].get<picojson::object>();
    picojson::object bn_layer_info = next_layer["info"].get<picojson::object>();
    const std::string conv_layer_name =
        conv_layer_info["name"].get<std::string>();
    const std::string bn_layer_name = bn_layer_info["name"].get<std::string>();
    const std::string fused_layer_name =
        conv_layer_name + "-fused-with-" + bn_layer_name;
    layers_iterator++;
    std::cout << "  Building " << fused_layer_name << "..." << std::endl;

    return build_conv_2d_fused_batch_norm(conv_layer_info, bn_layer_info,
                                          fused_layer_name, model_params_path,
                                          seal_tool);
  }
  if (layer_class_name == LINEAR_CLASS_NAME &&
      next_layer_class_name == BATCH_NORM_CLASS_NAME) {
    picojson::object linear_layer_info = layer["info"].get<picojson::object>();
    picojson::object bn_layer_info = next_layer["info"].get<picojson::object>();
    const std::string linear_layer_name =
        linear_layer_info["name"].get<std::string>();
    const std::string bn_layer_name = bn_layer_info["name"].get<std::string>();
    const std::string fused_layer_name =
        linear_layer_name + "-fused-with-" + bn_layer_name;
    layers_iterator++;
    std::cout << "  Building " << fused_layer_name << "..." << std::endl;

    return build_linear_fused_batch_norm(linear_layer_info, bn_layer_info,
                                         fused_layer_name, model_params_path,
                                         seal_tool);
  }

  return LayerBuilder::build(layer, model_params_path, seal_tool);
}

}  // namespace cnn::encrypted

namespace cnn::encrypted::batch {

std::shared_ptr<Layer> build_conv_2d(
    picojson::object& layer_info,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool> seal_tool) {
  /* Read structure info */
  const std::string layer_name = layer_info["name"].get<std::string>();
  std::cout << "  Building " << layer_name << "..." << std::endl;
  const std::size_t filter_size = layer_info["filters"].get<double>();
  const picojson::array filter_hw =
      layer_info["kernel_size"].get<picojson::array>();
  const std::size_t filter_h = filter_hw[0].get<double>();
  const std::size_t filter_w = filter_hw[1].get<double>();
  const picojson::array stride_hw = layer_info["stride"].get<picojson::array>();
  const std::size_t stride_h = stride_hw[0].get<double>();
  const std::size_t stride_w = stride_hw[1].get<double>();
  std::pair<std::string, std::pair<std::size_t, std::size_t>> padding;
  if (layer_info["padding"].is<std::string>()) {
    const std::string pad_str = layer_info["padding"].get<std::string>();
    padding = {pad_str, {0, 0}};
  } else if (layer_info["padding"].is<picojson::array>()) {
    const picojson::array padding_hw =
        layer_info["padding"].get<picojson::array>();
    const std::size_t padding_h = padding_hw[0].get<double>();
    const std::size_t padding_w = padding_hw[1].get<double>();
    padding = {"", {padding_h, padding_w}};
  }

  /* Read params data */
  H5::H5File params_file(model_params_path, H5F_ACC_RDONLY);
  H5::Group group = params_file.openGroup("/" + layer_name);
  H5::DataSet weight_data = group.openDataSet("weight");
  H5::DataSet bias_data = group.openDataSet("bias");

  H5::DataSpace weight_space = weight_data.getSpace();
  int weight_rank = weight_space.getSimpleExtentNdims();
  hsize_t weight_shape[weight_rank];
  int ndims = weight_space.getSimpleExtentDims(weight_shape);

  const std::size_t filter_n = weight_shape[0], in_channel = weight_shape[1],
                    filter_height = weight_shape[2],
                    filter_width = weight_shape[3];
  std::vector<float> flattened_filters(filter_n * in_channel * filter_height *
                                       filter_width);
  std::vector<float> biases(filter_n);

  weight_data.read(flattened_filters.data(), H5::PredType::NATIVE_FLOAT);
  bias_data.read(biases.data(), H5::PredType::NATIVE_FLOAT);

  double folding_value = 1, weight;
  if (OPT_OPTION.enable_fold_act_coeff && SHOULD_MUL_ACT_COEFF &&
      OPT_OPTION.enable_fold_pool_coeff && SHOULD_MUL_POOL_COEFF) {
    folding_value = POLY_ACT_HIGHEST_DEG_COEFF * CURRENT_POOL_MUL_COEFF;
    SHOULD_MUL_ACT_COEFF = false;
    SHOULD_MUL_POOL_COEFF = false;
  } else if (OPT_OPTION.enable_fold_act_coeff && SHOULD_MUL_ACT_COEFF) {
    folding_value = POLY_ACT_HIGHEST_DEG_COEFF;
    SHOULD_MUL_ACT_COEFF = false;
  } else if (OPT_OPTION.enable_fold_pool_coeff && SHOULD_MUL_POOL_COEFF) {
    folding_value = CURRENT_POOL_MUL_COEFF;
    SHOULD_MUL_POOL_COEFF = false;
  }

  // types::float4d filters(
  //     filter_n,
  //     types::float3d(
  //         in_channel,
  //         types::float2d(filter_height, std::vector<float>(filter_width))));
  types::Plaintext4d plain_filters(
      filter_n,
      types::Plaintext3d(
          in_channel,
          types::Plaintext2d(filter_height,
                             std::vector<seal::Plaintext>(filter_width))));
  std::vector<seal::Plaintext> plain_biases(filter_n);

#ifdef _OPENMP
#pragma omp parallel for collapse(4) private(weight)
#endif
  for (std::size_t fn = 0; fn < filter_n; ++fn) {
    for (std::size_t ic = 0; ic < in_channel; ++ic) {
      for (std::size_t fh = 0; fh < filter_height; ++fh) {
        for (std::size_t fw = 0; fw < filter_width; ++fw) {
          weight =
              folding_value *
              flattened_filters[fn * (in_channel * filter_height * filter_w) +
                                ic * (filter_height * filter_w) +
                                fh * filter_w + fw];
          if (fabs(weight) < ROUND_THRESHOLD) {
            round_value(weight);
          }
          seal_tool->encoder().encode(weight, seal_tool->scale(),
                                      plain_filters[fn][ic][fh][fw]);
          for (std::size_t lv = 0; lv < CONSUMED_LEVEL; ++lv) {
            seal_tool->evaluator().mod_switch_to_next_inplace(
                plain_filters[fn][ic][fh][fw]);
          }
        }
      }
    }
  }

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (std::size_t fn = 0; fn < filter_n; ++fn) {
    seal_tool->encoder().encode(biases[fn], seal_tool->scale(),
                                plain_biases[fn]);
    for (std::size_t lv = 0; lv < CONSUMED_LEVEL + 1; ++lv) {
      seal_tool->evaluator().mod_switch_to_next_inplace(plain_biases[fn]);
    }
  }

  return std::make_shared<Conv2d>(layer_name, plain_filters, plain_biases,
                                  seal_tool, stride_h, stride_w, padding);
}

std::shared_ptr<Layer> build_avg_pool_2d(
    picojson::object& layer_info,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool> seal_tool) {
  /* Read structure info */
  const std::string layer_name = layer_info["name"].get<std::string>();
  std::cout << "  Building " << layer_name << "..." << std::endl;
  const picojson::array pool_hw =
      layer_info["kernel_size"].get<picojson::array>();
  const std::size_t pool_height = pool_hw[0].get<double>();
  const std::size_t pool_width = pool_hw[1].get<double>();
  const picojson::array stride_hw = layer_info["stride"].get<picojson::array>();
  const std::size_t stride_height = stride_hw[0].get<double>();
  const std::size_t stride_width = stride_hw[1].get<double>();
  const picojson::array padding_hw =
      layer_info["padding"].get<picojson::array>();
  const std::size_t padding_height = padding_hw[0].get<double>();
  const std::size_t padding_width = padding_hw[1].get<double>();
  const std::pair<std::size_t, std::size_t> padding = {padding_height,
                                                       padding_width};

  seal::Plaintext plain_mul_factor;

  if (OPT_OPTION.enable_fold_pool_coeff) {
    CURRENT_POOL_MUL_COEFF = 1.0 / (pool_height * pool_width);
    SHOULD_MUL_POOL_COEFF = true;
  } else if (OPT_OPTION.enable_fold_act_coeff && SHOULD_MUL_ACT_COEFF) {
    seal_tool->encoder().encode(
        POLY_ACT_HIGHEST_DEG_COEFF / (pool_height * pool_width),
        seal_tool->scale(), plain_mul_factor);
    SHOULD_MUL_ACT_COEFF = false;
    for (std::size_t lv = 0; lv < CONSUMED_LEVEL; ++lv) {
      seal_tool->evaluator().mod_switch_to_next_inplace(plain_mul_factor);
    }
  } else {
    seal_tool->encoder().encode(1.0 / (pool_height * pool_width),
                                seal_tool->scale(), plain_mul_factor);
    for (std::size_t lv = 0; lv < CONSUMED_LEVEL; ++lv) {
      seal_tool->evaluator().mod_switch_to_next_inplace(plain_mul_factor);
    }
  }

  return std::make_shared<AvgPool2d>(layer_name, plain_mul_factor, pool_height,
                                     pool_width, stride_height, stride_width,
                                     padding, seal_tool);
}

std::shared_ptr<Layer> build_activation(
    picojson::object& layer_info,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool> seal_tool) {
  /* Read structure info */
  const std::string layer_name = layer_info["name"].get<std::string>();
  const std::string function_name = layer_info["function"].get<std::string>();
  std::cout << "  Building " << layer_name << " (" << function_name << ")..."
            << std::endl;

  if ((ACTIVATION_TYPE == EActivationType::DEG2_POLY_APPROX ||
       ACTIVATION_TYPE == EActivationType::DEG4_POLY_APPROX) &&
      OPT_OPTION.enable_fold_act_coeff) {
    SHOULD_MUL_ACT_COEFF = true;
  }

  return std::make_shared<Activation>(layer_name, ACTIVATION_TYPE, seal_tool);
}

std::shared_ptr<Layer> build_batch_norm(
    picojson::object& layer_info,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool> seal_tool) {
  /* Read structure info */
  const std::string layer_name = layer_info["name"].get<std::string>();
  std::cout << "  Building " << layer_name << "..." << std::endl;
  const std::string eps_str = layer_info["eps"].get<std::string>();
  const double eps = std::atof(eps_str.c_str());

  /* Read params data */
  H5::H5File params_file(model_params_path, H5F_ACC_RDONLY);
  H5::Group group = params_file.openGroup("/" + layer_name);
  H5::DataSet gamma_data = group.openDataSet("weight");
  H5::DataSet beta_data = group.openDataSet("bias");
  H5::DataSet running_mean_data = group.openDataSet("running_mean");
  H5::DataSet running_var_data = group.openDataSet("running_var");

  H5::DataSpace gamma_space = gamma_data.getSpace();
  int gamma_rank = gamma_space.getSimpleExtentNdims();
  hsize_t gamma_shape[gamma_rank];
  int ndims = gamma_space.getSimpleExtentDims(gamma_shape);

  const std::size_t num_features = gamma_shape[0];

  std::vector<float> gammas(num_features), betas(num_features),
      running_means(num_features), running_vars(num_features);

  gamma_data.read(gammas.data(), H5::PredType::NATIVE_FLOAT);
  beta_data.read(betas.data(), H5::PredType::NATIVE_FLOAT);
  running_mean_data.read(running_means.data(), H5::PredType::NATIVE_FLOAT);
  running_var_data.read(running_vars.data(), H5::PredType::NATIVE_FLOAT);

  double weight, bias;
  std::vector<seal::Plaintext> plain_weights(num_features),
      plain_biases(num_features);

#ifdef _OPENMP
#pragma omp parallel for private(weight, bias)
#endif
  for (size_t i = 0; i < num_features; ++i) {
    weight = gammas[i] / std::sqrt(running_vars[i] + eps);
    bias = betas[i] - (weight * running_means[i]);

    seal_tool->encoder().encode(weight, seal_tool->scale(), plain_weights[i]);
    seal_tool->encoder().encode(bias, seal_tool->scale(), plain_biases[i]);
    for (size_t lv = 0; lv < CONSUMED_LEVEL; ++lv) {
      seal_tool->evaluator().mod_switch_to_next_inplace(plain_weights[i]);
      seal_tool->evaluator().mod_switch_to_next_inplace(plain_biases[i]);
    }
    seal_tool->evaluator().mod_switch_to_next_inplace(plain_biases[i]);
  }

  return std::make_shared<BatchNorm>(layer_name, plain_weights, plain_biases,
                                     seal_tool);
}

std::shared_ptr<Layer> build_linear(
    picojson::object& layer_info,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool> seal_tool) {
  /* Read structure info */
  const std::string layer_name = layer_info["name"].get<std::string>();
  std::cout << "  Building " << layer_name << "..." << std::endl;
  const std::size_t unit_size = layer_info["units"].get<double>();

  /* Read params data */
  H5::H5File params_file(model_params_path, H5F_ACC_RDONLY);
  H5::Group group = params_file.openGroup("/" + layer_name);
  H5::DataSet weight_data = group.openDataSet("weight");
  H5::DataSet bias_data = group.openDataSet("bias");

  H5::DataSpace weight_space = weight_data.getSpace();
  int weight_rank = weight_space.getSimpleExtentNdims();
  hsize_t weight_shape[weight_rank];
  int ndims = weight_space.getSimpleExtentDims(weight_shape);

  const std::size_t out_channel = weight_shape[0], in_channel = weight_shape[1];
  std::vector<float> flattened_weights(out_channel * in_channel);
  std::vector<float> biases(out_channel);

  weight_data.read(flattened_weights.data(), H5::PredType::NATIVE_FLOAT);
  bias_data.read(biases.data(), H5::PredType::NATIVE_FLOAT);

  double folding_value = 1, weight;
  if (OPT_OPTION.enable_fold_act_coeff && SHOULD_MUL_ACT_COEFF &&
      OPT_OPTION.enable_fold_pool_coeff && SHOULD_MUL_POOL_COEFF) {
    folding_value = POLY_ACT_HIGHEST_DEG_COEFF * CURRENT_POOL_MUL_COEFF;
    SHOULD_MUL_ACT_COEFF = false;
    SHOULD_MUL_POOL_COEFF = false;
  } else if (OPT_OPTION.enable_fold_act_coeff && SHOULD_MUL_ACT_COEFF) {
    folding_value = POLY_ACT_HIGHEST_DEG_COEFF;
    SHOULD_MUL_ACT_COEFF = false;
  } else if (OPT_OPTION.enable_fold_pool_coeff && SHOULD_MUL_POOL_COEFF) {
    folding_value = CURRENT_POOL_MUL_COEFF;
    SHOULD_MUL_POOL_COEFF = false;
  }

  types::Plaintext2d plain_weights(out_channel,
                                   std::vector<seal::Plaintext>(in_channel));
  std::vector<seal::Plaintext> plain_biases(out_channel);

#ifdef _OPENMP
#pragma omp parallel for collapse(2) private(weight)
#endif
  for (std::size_t oc = 0; oc < out_channel; ++oc) {
    for (std::size_t ic = 0; ic < in_channel; ++ic) {
      weight = folding_value * flattened_weights[oc * in_channel + ic];
      if (fabs(weight) < ROUND_THRESHOLD) {
        round_value(weight);
      }
      seal_tool->encoder().encode(weight, seal_tool->scale(),
                                  plain_weights[oc][ic]);
      for (std::size_t lv = 0; lv < CONSUMED_LEVEL; ++lv) {
        seal_tool->evaluator().mod_switch_to_next_inplace(
            plain_weights[oc][ic]);
      }
    }
  }

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (std::size_t oc = 0; oc < out_channel; ++oc) {
    seal_tool->encoder().encode(biases[oc], seal_tool->scale(),
                                plain_biases[oc]);
    for (size_t lv = 0; lv < CONSUMED_LEVEL + 1; ++lv) {
      seal_tool->evaluator().mod_switch_to_next_inplace(plain_biases[oc]);
    }
  }

  return std::make_shared<Linear>(layer_name, plain_weights, plain_biases,
                                  seal_tool);
}

std::shared_ptr<Layer> build_flatten(
    picojson::object& layer_info,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool> seal_tool) {
  /* Read structure info */
  const std::string layer_name = layer_info["name"].get<std::string>();
  std::cout << "  Building " << layer_name << "..." << std::endl;

  return std::make_shared<Flatten>(layer_name, seal_tool);
}

std::shared_ptr<Layer> build_conv_2d_fused_batch_norm(
    picojson::object& conv_layer_info,
    picojson::object& bn_layer_info,
    const std::string& layer_name,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool> seal_tool) {
  /* Read Conv2d info */
  const std::string conv_layer_name =
      conv_layer_info["name"].get<std::string>();
  const std::size_t filter_size = conv_layer_info["filters"].get<double>();
  const picojson::array filter_hw =
      conv_layer_info["kernel_size"].get<picojson::array>();
  const std::size_t filter_h = filter_hw[0].get<double>();
  const std::size_t filter_w = filter_hw[1].get<double>();
  const picojson::array stride_hw =
      conv_layer_info["stride"].get<picojson::array>();
  const std::size_t stride_h = stride_hw[0].get<double>();
  const std::size_t stride_w = stride_hw[1].get<double>();
  std::pair<std::string, std::pair<std::size_t, std::size_t>> padding;
  if (conv_layer_info["padding"].is<std::string>()) {
    const std::string pad_str = conv_layer_info["padding"].get<std::string>();
    padding = {pad_str, {0, 0}};
  } else if (conv_layer_info["padding"].is<picojson::array>()) {
    const picojson::array padding_hw =
        conv_layer_info["padding"].get<picojson::array>();
    const std::size_t padding_h = padding_hw[0].get<double>();
    const std::size_t padding_w = padding_hw[1].get<double>();
    padding = {"", {padding_h, padding_w}};
  }

  // Read params of Conv2d
  H5::H5File params_file(model_params_path, H5F_ACC_RDONLY);
  H5::Group conv_group = params_file.openGroup("/" + conv_layer_name);
  H5::DataSet conv_weight_data = conv_group.openDataSet("weight");
  H5::DataSet conv_bias_data = conv_group.openDataSet("bias");

  H5::DataSpace conv_weight_space = conv_weight_data.getSpace();
  int conv_weight_rank = conv_weight_space.getSimpleExtentNdims();
  hsize_t conv_weight_shape[conv_weight_rank];
  int conv_ndims = conv_weight_space.getSimpleExtentDims(conv_weight_shape);

  const std::size_t filter_n = conv_weight_shape[0],
                    in_channel = conv_weight_shape[1],
                    filter_height = conv_weight_shape[2],
                    filter_width = conv_weight_shape[3];
  std::vector<float> conv_flattened_filters(filter_n * in_channel *
                                            filter_height * filter_width);
  std::vector<float> conv_biases(filter_n);

  conv_weight_data.read(conv_flattened_filters.data(),
                        H5::PredType::NATIVE_FLOAT);
  conv_bias_data.read(conv_biases.data(), H5::PredType::NATIVE_FLOAT);

  /* Read BatchNorm info */
  const std::string bn_layer_name = bn_layer_info["name"].get<std::string>();
  const std::string eps_str = bn_layer_info["eps"].get<std::string>();
  const double eps = std::atof(eps_str.c_str());

  // Read params of BatchNorm
  H5::Group bn_group = params_file.openGroup("/" + bn_layer_name);
  H5::DataSet bn_gamma_data = bn_group.openDataSet("weight");
  H5::DataSet bn_beta_data = bn_group.openDataSet("bias");
  H5::DataSet bn_running_mean_data = bn_group.openDataSet("running_mean");
  H5::DataSet bn_running_var_data = bn_group.openDataSet("running_var");

  H5::DataSpace bn_gamma_space = bn_gamma_data.getSpace();
  int bn_gamma_rank = bn_gamma_space.getSimpleExtentNdims();
  hsize_t bn_gamma_shape[bn_gamma_rank];
  int bn_ndims = bn_gamma_space.getSimpleExtentDims(bn_gamma_shape);

  const std::size_t num_features = bn_gamma_shape[0];

  std::vector<float> bn_gammas(num_features), bn_betas(num_features),
      bn_running_means(num_features), bn_running_vars(num_features);

  bn_gamma_data.read(bn_gammas.data(), H5::PredType::NATIVE_FLOAT);
  bn_beta_data.read(bn_betas.data(), H5::PredType::NATIVE_FLOAT);
  bn_running_mean_data.read(bn_running_means.data(),
                            H5::PredType::NATIVE_FLOAT);
  bn_running_var_data.read(bn_running_vars.data(), H5::PredType::NATIVE_FLOAT);

  /* Calculate fused weights & biases */
  assert(filter_n == num_features);

  double folding_value = 1;
  if (OPT_OPTION.enable_fold_act_coeff && SHOULD_MUL_ACT_COEFF &&
      OPT_OPTION.enable_fold_pool_coeff && SHOULD_MUL_POOL_COEFF) {
    folding_value = POLY_ACT_HIGHEST_DEG_COEFF * CURRENT_POOL_MUL_COEFF;
    SHOULD_MUL_ACT_COEFF = false;
    SHOULD_MUL_POOL_COEFF = false;
  } else if (OPT_OPTION.enable_fold_act_coeff && SHOULD_MUL_ACT_COEFF) {
    folding_value = POLY_ACT_HIGHEST_DEG_COEFF;
    SHOULD_MUL_ACT_COEFF = false;
  } else if (OPT_OPTION.enable_fold_pool_coeff && SHOULD_MUL_POOL_COEFF) {
    folding_value = CURRENT_POOL_MUL_COEFF;
    SHOULD_MUL_POOL_COEFF = false;
  }

  std::vector<double> bn_weights(filter_n), bn_biases(filter_n);
  double fused_weight, fused_bias;
  types::Plaintext4d plain_filters(
      filter_n,
      types::Plaintext3d(
          in_channel,
          types::Plaintext2d(filter_height,
                             std::vector<seal::Plaintext>(filter_width))));
  std::vector<seal::Plaintext> plain_biases(filter_n);

  // plain_biases
#ifdef _OPENMP
#pragma omp parallel for private(fused_bias)
#endif
  for (std::size_t fn = 0; fn < filter_n; ++fn) {
    bn_weights[fn] = bn_gammas[fn] / std::sqrt(bn_running_vars[fn] + eps);
    bn_biases[fn] = bn_betas[fn] - (bn_weights[fn] * bn_running_means[fn]);
    fused_bias = conv_biases[fn] * bn_weights[fn] + bn_biases[fn];
    seal_tool->encoder().encode(fused_bias, seal_tool->scale(),
                                plain_biases[fn]);
    for (std::size_t lv = 0; lv < CONSUMED_LEVEL + 1; ++lv) {
      seal_tool->evaluator().mod_switch_to_next_inplace(plain_biases[fn]);
    }
  }

  // plain_weights
#ifdef _OPENMP
#pragma omp parallel for collapse(4) private(fused_weight)
#endif
  for (std::size_t fn = 0; fn < filter_n; ++fn) {
    for (std::size_t ic = 0; ic < in_channel; ++ic) {
      for (std::size_t fh = 0; fh < filter_height; ++fh) {
        for (std::size_t fw = 0; fw < filter_width; ++fw) {
          fused_weight =
              folding_value *
              conv_flattened_filters[fn * (in_channel * filter_height *
                                           filter_w) +
                                     ic * (filter_height * filter_w) +
                                     fh * filter_w + fw] *
              bn_weights[fn];
          if (fabs(fused_weight) < ROUND_THRESHOLD) {
            // std::cout << "fused_weight: " << fused_weight << std::endl;
            round_value(fused_weight);
          }
          seal_tool->encoder().encode(fused_weight, seal_tool->scale(),
                                      plain_filters[fn][ic][fh][fw]);
          for (std::size_t lv = 0; lv < CONSUMED_LEVEL; ++lv) {
            seal_tool->evaluator().mod_switch_to_next_inplace(
                plain_filters[fn][ic][fh][fw]);
          }
        }
      }
    }
  }

  return std::make_shared<Conv2d>(layer_name, plain_filters, plain_biases,
                                  seal_tool, stride_h, stride_w, padding);
}

std::shared_ptr<Layer> build_linear_fused_batch_norm(
    picojson::object& linear_layer_info,
    picojson::object& bn_layer_info,
    const std::string& layer_name,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool> seal_tool) {
  /* Read Linear info */
  const std::string linear_layer_name =
      linear_layer_info["name"].get<std::string>();
  const std::size_t unit_size = linear_layer_info["units"].get<double>();

  /* Read params of Linear */
  H5::H5File params_file(model_params_path, H5F_ACC_RDONLY);
  H5::Group group = params_file.openGroup("/" + linear_layer_name);
  H5::DataSet linear_weight_data = group.openDataSet("weight");
  H5::DataSet linear_bias_data = group.openDataSet("bias");

  H5::DataSpace linear_weight_space = linear_weight_data.getSpace();
  int linear_weight_rank = linear_weight_space.getSimpleExtentNdims();
  hsize_t linear_weight_shape[linear_weight_rank];
  int linear_ndims =
      linear_weight_space.getSimpleExtentDims(linear_weight_shape);

  const std::size_t out_channel = linear_weight_shape[0],
                    in_channel = linear_weight_shape[1];
  std::vector<float> linear_flattened_weights(out_channel * in_channel);
  std::vector<float> linear_biases(out_channel);

  linear_weight_data.read(linear_flattened_weights.data(),
                          H5::PredType::NATIVE_FLOAT);
  linear_bias_data.read(linear_biases.data(), H5::PredType::NATIVE_FLOAT);

  /* Read BatchNorm info */
  const std::string bn_layer_name = bn_layer_info["name"].get<std::string>();
  const std::string eps_str = bn_layer_info["eps"].get<std::string>();
  const double eps = std::atof(eps_str.c_str());

  // Read params of BatchNorm
  H5::Group bn_group = params_file.openGroup("/" + bn_layer_name);
  H5::DataSet bn_gamma_data = bn_group.openDataSet("weight");
  H5::DataSet bn_beta_data = bn_group.openDataSet("bias");
  H5::DataSet bn_running_mean_data = bn_group.openDataSet("running_mean");
  H5::DataSet bn_running_var_data = bn_group.openDataSet("running_var");

  H5::DataSpace bn_gamma_space = bn_gamma_data.getSpace();
  int bn_gamma_rank = bn_gamma_space.getSimpleExtentNdims();
  hsize_t bn_gamma_shape[bn_gamma_rank];
  int bn_ndims = bn_gamma_space.getSimpleExtentDims(bn_gamma_shape);

  const std::size_t num_features = bn_gamma_shape[0];

  std::vector<float> bn_gammas(num_features), bn_betas(num_features),
      bn_running_means(num_features), bn_running_vars(num_features);

  bn_gamma_data.read(bn_gammas.data(), H5::PredType::NATIVE_FLOAT);
  bn_beta_data.read(bn_betas.data(), H5::PredType::NATIVE_FLOAT);
  bn_running_mean_data.read(bn_running_means.data(),
                            H5::PredType::NATIVE_FLOAT);
  bn_running_var_data.read(bn_running_vars.data(), H5::PredType::NATIVE_FLOAT);

  /* Calculate fused weights & biases */
  assert(out_channel == num_features);

  double folding_value = 1;
  if (OPT_OPTION.enable_fold_act_coeff && SHOULD_MUL_ACT_COEFF &&
      OPT_OPTION.enable_fold_pool_coeff && SHOULD_MUL_POOL_COEFF) {
    folding_value = POLY_ACT_HIGHEST_DEG_COEFF * CURRENT_POOL_MUL_COEFF;
    SHOULD_MUL_ACT_COEFF = false;
    SHOULD_MUL_POOL_COEFF = false;
  } else if (OPT_OPTION.enable_fold_act_coeff && SHOULD_MUL_ACT_COEFF) {
    folding_value = POLY_ACT_HIGHEST_DEG_COEFF;
    SHOULD_MUL_ACT_COEFF = false;
  } else if (OPT_OPTION.enable_fold_pool_coeff && SHOULD_MUL_POOL_COEFF) {
    folding_value = CURRENT_POOL_MUL_COEFF;
    SHOULD_MUL_POOL_COEFF = false;
  }

  std::vector<double> bn_weights(num_features), bn_biases(num_features);
  double fused_weight, fused_bias;
  types::Plaintext2d plain_weights(out_channel,
                                   std::vector<seal::Plaintext>(in_channel));
  std::vector<seal::Plaintext> plain_biases(num_features);

  // plain_biases
#ifdef _OPENMP
#pragma omp parallel for private(fused_bias)
#endif
  for (std::size_t oc = 0; oc < out_channel; ++oc) {
    bn_weights[oc] = bn_gammas[oc] / std::sqrt(bn_running_vars[oc] + eps);
    bn_biases[oc] = bn_betas[oc] - (bn_weights[oc] * bn_running_means[oc]);
    fused_bias = linear_biases[oc] * bn_weights[oc] + bn_biases[oc];
    seal_tool->encoder().encode(fused_bias, seal_tool->scale(),
                                plain_biases[oc]);
    for (std::size_t lv = 0; lv < CONSUMED_LEVEL + 1; ++lv) {
      seal_tool->evaluator().mod_switch_to_next_inplace(plain_biases[oc]);
    }
  }

  // plain_weights
#ifdef _OPENMP
#pragma omp parallel for collapse(2) private(fused_weight)
#endif
  for (std::size_t oc = 0; oc < out_channel; ++oc) {
    for (std::size_t ic = 0; ic < in_channel; ++ic) {
      fused_weight = folding_value *
                     linear_flattened_weights[oc * in_channel + ic] *
                     bn_weights[oc];
      if (fabs(fused_weight) < ROUND_THRESHOLD) {
        // std::cout << "fused_weight: " << fused_weight << std::endl;
        round_value(fused_weight);
      }
      seal_tool->encoder().encode(fused_weight, seal_tool->scale(),
                                  plain_weights[oc][ic]);
      for (std::size_t lv = 0; lv < CONSUMED_LEVEL; ++lv) {
        seal_tool->evaluator().mod_switch_to_next_inplace(
            plain_weights[oc][ic]);
      }
    }
  }

  return std::make_shared<Linear>(layer_name, plain_weights, plain_biases,
                                  seal_tool);
}

const std::unordered_map<std::string,
                         std::function<std::shared_ptr<Layer>(
                             picojson::object&,
                             const std::string&,
                             const std::shared_ptr<helper::he::SealTool>)>>
    BUILD_LAYER_MAP{{CONV_2D_CLASS_NAME, build_conv_2d},
                    {AVG_POOL_2D_CLASS_NAME, build_avg_pool_2d},
                    {ACTIVATION_CLASS_NAME, build_activation},
                    {BATCH_NORM_CLASS_NAME, build_batch_norm},
                    {LINEAR_CLASS_NAME, build_linear},
                    {FLATTEN_CLASS_NAME, build_flatten}};

std::shared_ptr<Layer> LayerBuilder::build(
    picojson::object& layer,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool> seal_tool) {
  const std::string layer_class_name = layer["class_name"].get<std::string>();
  if (auto map_iter = BUILD_LAYER_MAP.find(layer_class_name);
      map_iter != BUILD_LAYER_MAP.end()) {
    picojson::object layer_info = layer["info"].get<picojson::object>();

    return map_iter->second(layer_info, model_params_path, seal_tool);
  } else {
    throw std::runtime_error("\"" + layer_class_name +
                             "\" is not registered as layer class");
  }
}

std::shared_ptr<Layer> LayerBuilder::build(
    picojson::object& layer,
    picojson::object& next_layer,
    picojson::array::const_iterator& layers_iterator,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool> seal_tool) {
  const std::string layer_class_name = layer["class_name"].get<std::string>();
  const std::string next_layer_class_name =
      next_layer["class_name"].get<std::string>();

  if (layer_class_name == CONV_2D_CLASS_NAME &&
      next_layer_class_name == BATCH_NORM_CLASS_NAME) {
    picojson::object conv_layer_info = layer["info"].get<picojson::object>();
    picojson::object bn_layer_info = next_layer["info"].get<picojson::object>();
    const std::string conv_layer_name =
        conv_layer_info["name"].get<std::string>();
    const std::string bn_layer_name = bn_layer_info["name"].get<std::string>();
    const std::string fused_layer_name =
        conv_layer_name + "-fused-with-" + bn_layer_name;
    layers_iterator++;
    std::cout << "  Building " << fused_layer_name << "..." << std::endl;

    return build_conv_2d_fused_batch_norm(conv_layer_info, bn_layer_info,
                                          fused_layer_name, model_params_path,
                                          seal_tool);
  }
  if (layer_class_name == LINEAR_CLASS_NAME &&
      next_layer_class_name == BATCH_NORM_CLASS_NAME) {
    picojson::object linear_layer_info = layer["info"].get<picojson::object>();
    picojson::object bn_layer_info = next_layer["info"].get<picojson::object>();
    const std::string linear_layer_name =
        linear_layer_info["name"].get<std::string>();
    const std::string bn_layer_name = bn_layer_info["name"].get<std::string>();
    const std::string fused_layer_name =
        linear_layer_name + "-fused-with-" + bn_layer_name;
    layers_iterator++;
    std::cout << "  Building " << fused_layer_name << "..." << std::endl;

    return build_linear_fused_batch_norm(linear_layer_info, bn_layer_info,
                                         fused_layer_name, model_params_path,
                                         seal_tool);
  }

  return LayerBuilder::build(layer, model_params_path, seal_tool);
}

}  // namespace cnn::encrypted::batch
