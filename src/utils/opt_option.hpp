#pragma once

struct OptOption {
  bool enable_fuse_linear_layers;
  bool enable_fold_act_coeff;
  bool enable_fold_pool_coeff;

  OptOption()
      : enable_fuse_linear_layers(false),
        enable_fold_act_coeff(false),
        enable_fold_pool_coeff(false) {}
  OptOption(bool enable_fuse_linear_layers,
            bool enable_fold_act_coeff,
            bool enable_fold_pool_coeff)
      : enable_fuse_linear_layers(enable_fuse_linear_layers),
        enable_fold_act_coeff(enable_fold_act_coeff),
        enable_fold_pool_coeff(enable_fold_pool_coeff) {}
};
