#include "constants.hpp"

namespace constants::fname {

const std::string DATASETS_DIR = "datasets";
const std::string TRAIN_MODEL_DIR = "train_model";
const std::string SECRETS_DIR = "secrets";
const std::string PARAMS_SUFFIX = "_params";
const std::string SECRET_KEY_SUFFIX = "_secretKey";
const std::string PUBLIC_KEY_SUFFIX = "_publicKey";
const std::string RELIN_KEYS_SUFFIX = "_relinKeys";
const std::string GALOIS_KEYS_SUFFIX = "_galoisKeys";

}  // namespace constants::fname

namespace constants::dataset {

const std::string MNIST = "mnist";
const std::string CIFAR10 = "cifar-10";

}  // namespace constants::dataset

namespace constants::mode {

const std::string SINGLE = "single";
const std::string BATCH = "batch";

}  // namespace constants::mode

namespace constants::activation {

const std::string SQUARE = "square";
const std::string RELU_RG5_DEG2 = "relu_rg5_deg2";
const std::string RELU_RG7_DEG2 = "relu_rg7_deg2";
const std::string RELU_RG5_DEG4 = "relu_rg5_deg4";
const std::string RELU_RG7_DEG4 = "relu_rg7_deg4";
const std::string SWISH_RG5_DEG2 = "swish_rg5_deg2";
const std::string SWISH_RG7_DEG2 = "swish_rg7_deg2";
const std::string SWISH_RG5_DEG4 = "swish_rg5_deg4";
const std::string SWISH_RG7_DEG4 = "swish_rg7_deg4";
const std::string MISH_RG5_DEG2 = "mish_rg5_deg2";
const std::string MISH_RG7_DEG2 = "mish_rg7_deg2";
const std::string MISH_RG5_DEG4 = "mish_rg5_deg4";
const std::string MISH_RG7_DEG4 = "mish_rg7_deg4";

/***********************
 * ReLU approximation
 ***********************/
/* 2 Degree */
// ax^2 + bx + c
std::vector<double> RELU_RG5_DEG2_COEFFS = {0.09, 0.5, 0.47};
std::vector<double> RELU_RG7_DEG2_COEFFS = {0.0669, 0.5, 0.6569};
// x^2 + b'x + c'
std::vector<double> RELU_RG5_DEG2_OPT_COEFFS = {5.5555, 5.2222};
std::vector<double> RELU_RG7_DEG2_OPT_COEFFS = {7.4738, 9.8191};
/* 4 Degree */
// ax^4 + bx^2 + cx + d
std::vector<double> RELU_RG5_DEG4_COEFFS = {-0.00327, 0.1639, 0.5, 0.2932};
std::vector<double> RELU_RG7_DEG4_COEFFS = {-0.00119, 0.11707, 0.5, 0.41056};
// x^4 + b'x^2 + c'x + d'
std::vector<double> RELU_RG5_DEG4_OPT_COEFFS = {-50.1223, -152.9052, -89.6636};
std::vector<double> RELU_RG7_DEG4_OPT_COEFFS = {-98.37815, -420.168067,
                                                -345.0084};

/***********************
 * Swish approximation
 ***********************/
/* 2 Degree */
// ax^2 + bx + c
std::vector<double> SWISH_RG5_DEG2_COEFFS = {0.1, 0.5, 0.24};
std::vector<double> SWISH_RG7_DEG2_COEFFS = {0.0723, 0.5, 0.4517};
// x^2 + b'x + c'
std::vector<double> SWISH_RG5_DEG2_OPT_COEFFS = {5.0, 2.4};
std::vector<double> SWISH_RG7_DEG2_OPT_COEFFS = {6.91563, 6.24758};
/* 4 Degree */
// ax^4 + bx^2 + cx + d
std::vector<double> SWISH_RG5_DEG4_COEFFS = {-0.00315, 0.17, 0.5, 0.07066};
std::vector<double> SWISH_RG7_DEG4_COEFFS = {-0.001328, 0.128, 0.5, 0.1773};
// x^4 + b'x^2 + c'x + d'
std::vector<double> SWISH_RG5_DEG4_OPT_COEFFS = {-53.968254, -158.73,
                                                 -22.431746};
std::vector<double> SWISH_RG7_DEG4_OPT_COEFFS = {-96.38554, -376.506, -133.509};

/***********************
 * Mish approximation
 ***********************/
/* 2 Degree */
// ax^2 + bx + c
std::vector<double> MISH_RG5_DEG2_COEFFS = {0.1, 0.516, 0.2967};
std::vector<double> MISH_RG7_DEG2_COEFFS = {0.071, 0.5066, 0.5095};
// x^2 + b'x + c'
std::vector<double> MISH_RG5_DEG2_OPT_COEFFS = {5.16, 2.967};
std::vector<double> MISH_RG7_DEG2_OPT_COEFFS = {7.1352, 7.17606};
/* 4 Degree */
// ax^4 + bx^2 + cx + d
std::vector<double> MISH_RG5_DEG4_COEFFS = {-0.00346, 0.17573, 0.5495, 0.1104};
std::vector<double> MISH_RG7_DEG4_COEFFS = {-0.00134, 0.12748, 0.527, 0.232356};
// x^4 + b'x^2 + c'x + d'
std::vector<double> MISH_RG5_DEG4_OPT_COEFFS = {-50.789, -158.815, -31.9075};
std::vector<double> MISH_RG7_DEG4_OPT_COEFFS = {-95.13433, -393.28358, -173.4};

std::unordered_map<std::string, std::vector<double>> POLY_ACT_MAP = {};
std::unordered_map<std::string, std::pair<std::vector<double>, double>>
    OPT_POLY_ACT_MAP = {};

// std::unordered_map<const std::string, std::vector<double>> POLY_ACT_MAP = {
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

// std::unordered_map<const std::string, std::pair<std::vector<double>, double>>
//     OPT_POLY_ACT_MAP = {
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
// };

}  // namespace constants::activation
