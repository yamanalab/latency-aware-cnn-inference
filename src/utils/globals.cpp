#include "globals.hpp"

OptOption OPT_OPTION;
std::size_t CONSUMED_LEVEL;
std::vector<double> POLY_ACT_COEFFS;
double POLY_ACT_HIGHEST_DEG_COEFF;
double CURRENT_POOL_MUL_COEFF;
bool SHOULD_MUL_ACT_COEFF;
bool SHOULD_MUL_POOL_COEFF;
EActivationType ACTIVATION_TYPE;
double ROUND_THRESHOLD;
seal::GaloisKeys GALOIS_KEYS;
std::set<int> USE_ROTATION_STEPS;
std::size_t INPUT_C, INPUT_H, INPUT_W, OUTPUT_C, OUTPUT_H, OUTPUT_W,
    INPUT_UNITS, OUTPUT_UNITS;
std::vector<std::vector<int>> INPUT_HW_SLOT_IDX;
std::vector<std::vector<int>> OUTPUT_HW_SLOT_IDX;
std::vector<std::vector<int>> KERNEL_HW_ROTATION_STEP;
std::vector<int> FLATTEN_ROTATION_STEP;
std::vector<int> INPUT_UNITS_SLOT_IDX;
std::vector<int> OUTPUT_UNITS_SLOT_IDX;
