#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace constants::fname {

extern const std::string DATASETS_DIR;
extern const std::string TRAIN_MODEL_DIR;
extern const std::string SECRETS_DIR;
extern const std::string PARAMS_SUFFIX;
extern const std::string SECRET_KEY_SUFFIX;
extern const std::string PUBLIC_KEY_SUFFIX;
extern const std::string RELIN_KEYS_SUFFIX;
extern const std::string GALOIS_KEYS_SUFFIX;

}  // namespace constants::fname

namespace constants::dataset {

extern const std::string MNIST;
extern const std::string CIFAR10;

}  // namespace constants::dataset

namespace constants::mode {

extern const std::string SINGLE;
extern const std::string BATCH;

}  // namespace constants::mode

namespace constants::activation {

extern const std::string SQUARE;
extern const std::string RELU_RG5_DEG2;
extern const std::string RELU_RG7_DEG2;
extern const std::string RELU_RG5_DEG4;
extern const std::string RELU_RG7_DEG4;
extern const std::string SWISH_RG5_DEG2;
extern const std::string SWISH_RG7_DEG2;
extern const std::string SWISH_RG5_DEG4;
extern const std::string SWISH_RG7_DEG4;
extern const std::string MISH_RG5_DEG2;
extern const std::string MISH_RG7_DEG2;
extern const std::string MISH_RG5_DEG4;
extern const std::string MISH_RG7_DEG4;

/* Coefficients of polynomial approximated ReLU */
// 2 degree
extern std::vector<double> RELU_RG5_DEG2_COEFFS;
extern std::vector<double> RELU_RG7_DEG2_COEFFS;
// 2 degree (dividid by highest degree coefficient)
extern std::vector<double> RELU_RG5_DEG2_OPT_COEFFS;
extern std::vector<double> RELU_RG7_DEG2_OPT_COEFFS;
// 4 degree
extern std::vector<double> RELU_RG5_DEG4_COEFFS;
extern std::vector<double> RELU_RG7_DEG4_COEFFS;
// 4 degree (dividid by highest degree coefficient)
extern std::vector<double> RELU_RG5_DEG4_OPT_COEFFS;
extern std::vector<double> RELU_RG7_DEG4_OPT_COEFFS;

/* Coefficients of polynomial approximated Swish */
// 2 degree
extern std::vector<double> SWISH_RG5_DEG2_COEFFS;
extern std::vector<double> SWISH_RG7_DEG2_COEFFS;
// 2 degree (dividid by highest degree coefficient)
extern std::vector<double> SWISH_RG5_DEG2_OPT_COEFFS;
extern std::vector<double> SWISH_RG7_DEG2_OPT_COEFFS;
// 4 degree
extern std::vector<double> SWISH_RG5_DEG4_COEFFS;
extern std::vector<double> SWISH_RG7_DEG4_COEFFS;
// 4 degree (dividid by highest degree coefficient)
extern std::vector<double> SWISH_RG5_DEG4_OPT_COEFFS;
extern std::vector<double> SWISH_RG7_DEG4_OPT_COEFFS;

/* Coefficients of polynomial approximated Mish */
// 2 degree
extern std::vector<double> MISH_RG5_DEG2_COEFFS;
extern std::vector<double> MISH_RG7_DEG2_COEFFS;
// 2 degree (dividid by highest degree coefficient)
extern std::vector<double> MISH_RG5_DEG2_OPT_COEFFS;
extern std::vector<double> MISH_RG7_DEG2_OPT_COEFFS;
// 4 degree
extern std::vector<double> MISH_RG5_DEG4_COEFFS;
extern std::vector<double> MISH_RG7_DEG4_COEFFS;
// 4 degree (dividid by highest degree coefficient)
extern std::vector<double> MISH_RG5_DEG4_OPT_COEFFS;
extern std::vector<double> MISH_RG7_DEG4_OPT_COEFFS;

/* key: activation function name, value: Pair<coeffs, highest_deg_coeff> */
extern std::unordered_map<std::string, std::vector<double>> POLY_ACT_MAP;
extern std::unordered_map<std::string, std::pair<std::vector<double>, double>>
    OPT_POLY_ACT_MAP;
// std::unordered_map<std::string, std::vector<double>> POLY_ACT_MAP{
//     {RELU_RG5_DEG2, RELU_RG5_DEG2_COEFFS},
//     {RELU_RG7_DEG2, RELU_RG7_DEG2_COEFFS},
//     {SWISH_RG5_DEG2, SWISH_RG5_DEG2_COEFFS},
//     {SWISH_RG7_DEG2, SWISH_RG7_DEG2_COEFFS},
//     {MISH_RG5_DEG2, MISH_RG5_DEG2_COEFFS},
//     {MISH_RG7_DEG2, MISH_RG7_DEG2_COEFFS},
//     {RELU_RG5_DEG4, RELU_RG5_DEG4_COEFFS},
//     {RELU_RG7_DEG4, RELU_RG7_DEG4_COEFFS},
//     {SWISH_RG5_DEG4, SWISH_RG5_DEG4_COEFFS},
//     {SWISH_RG7_DEG4, SWISH_RG7_DEG4_COEFFS},
//     {MISH_RG5_DEG4, MISH_RG5_DEG4_COEFFS},
//     {MISH_RG7_DEG4, MISH_RG7_DEG4_COEFFS},
// };
// std::unordered_map<std::string, std::pair<std::vector<double>, double>>
//     OPT_POLY_ACT_MAP{
//         {RELU_RG5_DEG2,
//          {RELU_RG5_DEG2_OPT_COEFFS, RELU_RG5_DEG2_COEFFS.front()}},
//         {RELU_RG7_DEG2,
//          {RELU_RG7_DEG2_OPT_COEFFS, RELU_RG7_DEG2_COEFFS.front()}},
//         {SWISH_RG5_DEG2,
//          {SWISH_RG5_DEG2_OPT_COEFFS, SWISH_RG5_DEG2_COEFFS.front()}},
//         {SWISH_RG7_DEG2,
//          {SWISH_RG7_DEG2_OPT_COEFFS, SWISH_RG7_DEG2_COEFFS.front()}},
//         {MISH_RG5_DEG2,
//          {MISH_RG5_DEG2_OPT_COEFFS, MISH_RG5_DEG2_COEFFS.front()}},
//         {MISH_RG7_DEG2,
//          {MISH_RG7_DEG2_OPT_COEFFS, MISH_RG7_DEG2_COEFFS.front()}},
//         {RELU_RG5_DEG4,
//          {RELU_RG5_DEG4_OPT_COEFFS, RELU_RG5_DEG4_COEFFS.front()}},
//         {RELU_RG7_DEG4,
//          {RELU_RG7_DEG4_OPT_COEFFS, RELU_RG7_DEG4_COEFFS.front()}},
//         {SWISH_RG5_DEG4,
//          {SWISH_RG5_DEG4_OPT_COEFFS, SWISH_RG5_DEG4_COEFFS.front()}},
//         {SWISH_RG7_DEG4,
//          {SWISH_RG7_DEG4_OPT_COEFFS, SWISH_RG7_DEG4_COEFFS.front()}},
//         {MISH_RG5_DEG4,
//          {MISH_RG5_DEG4_OPT_COEFFS, MISH_RG5_DEG4_COEFFS.front()}},
//         {MISH_RG7_DEG4,
//          {MISH_RG7_DEG4_OPT_COEFFS, MISH_RG7_DEG4_COEFFS.front()}},
//     };

}  // namespace constants::activation
