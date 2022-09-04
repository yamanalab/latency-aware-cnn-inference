#pragma once

namespace cnn {

enum ELayerType {
  CONV_2D,
  AVG_POOL_2D,
  ACTIVATION,
  BATCH_NORM,
  LINEAR,
  FLATTEN
};

}  // namespace cnn

namespace cnn::encrypted {

enum ELayerType {
  CONV_2D,
  AVG_POOL_2D,
  ACTIVATION,
  BATCH_NORM,
  LINEAR,
  FLATTEN
};

}  // namespace cnn::encrypted
