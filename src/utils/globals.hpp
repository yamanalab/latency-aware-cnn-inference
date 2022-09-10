#pragma once

#include <seal/seal.h>
#include <set>
#include <utils/opt_option.hpp>
#include <vector>

extern OptOption OPT_OPTION;

/* memo of consumed multiplicative level that is used when load trained model */
extern std::size_t CONSUMED_LEVEL;

/* coefficients of polynomial activation */
extern std::vector<double> POLY_ACT_COEFFS;

/* highest degree coefficient of polynomial activation */
extern double POLY_ACT_HIGHEST_DEG_COEFF;

/* pooling multiply coefficient */
extern double CURRENT_POOL_MUL_COEFF;

/* if 'opt-act' option is true,
   multiply highest degree coeff of polynomial activation function to weight
   parameter of next linear layer of activation layer */
extern bool SHOULD_MUL_ACT_COEFF;

/* if 'opt-pool' option is true,
   multiply 1/(pool_height * pool_width) to parameter of next linear layer of
   average pooling layer */
extern bool SHOULD_MUL_POOL_COEFF;

/* supported types of activation function for inference on ciphertext */
enum EActivationType { SQUARE, DEG2_POLY_APPROX, DEG4_POLY_APPROX };
extern EActivationType ACTIVATION_TYPE;

// Rounding value for when encode too small value (depending on SEAL parameter)
// if fabs(target_encode_value) < ROUND_THRESHOLD, we change target_encode_value
// = ROUND_THRESHOLD * (target_encode_value/fabs(target_encode_value))
extern double ROUND_THRESHOLD;

extern seal::GaloisKeys GALOIS_KEYS;
extern std::set<int> USE_ROTATION_STEPS;

extern std::size_t INPUT_C, INPUT_H, INPUT_W, OUTPUT_C, OUTPUT_H, OUTPUT_W,
    INPUT_UNITS, OUTPUT_UNITS;
extern std::vector<std::vector<int>> INPUT_HW_SLOT_IDX;
extern std::vector<std::vector<int>> OUTPUT_HW_SLOT_IDX;
extern std::vector<std::vector<int>> KERNEL_HW_ROTATION_STEP;
extern std::vector<int> FLATTEN_ROTATION_STEP;
extern std::vector<int> INPUT_UNITS_SLOT_IDX;
extern std::vector<int> OUTPUT_UNITS_SLOT_IDX;
