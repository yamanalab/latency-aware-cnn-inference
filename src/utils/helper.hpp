#pragma once

#include <seal/seal.h>

#include "picojson.h"
#include "types.hpp"

namespace helper::he {

class SealTool {
public:
  SealTool(seal::Evaluator& evaluator,
           seal::CKKSEncoder& encoder,
           seal::Decryptor& decryptor,
           seal::RelinKeys& relin_keys,
           //  seal::GaloisKeys& galois_keys,
           const std::size_t slot_count,
           const double scale);
  ~SealTool() = default;

  seal::Evaluator& evaluator() const { return evaluator_; };
  seal::CKKSEncoder& encoder() const { return encoder_; };
  seal::Decryptor& decryptor() const { return decryptor_; };
  seal::RelinKeys& relin_keys() const { return relin_keys_; };
  // seal::GaloisKeys& galois_keys() const { return galois_keys_; };
  const size_t slot_count() const { return slot_count_; };
  const double scale() const { return scale_; };

private:
  seal::Evaluator& evaluator_;
  seal::CKKSEncoder& encoder_;
  seal::Decryptor& decryptor_;
  seal::RelinKeys& relin_keys_;
  // seal::GaloisKeys& galois_keys_;
  std::size_t slot_count_;
  double scale_;
};

/*
Helper function: Prints the parameters in a SEALContext.
*/
void print_parameters(const std::shared_ptr<seal::SEALContext>& context);

/**
 * @brief Calculate total sum within slots of a ciphertext
 *
 * @param target_ct
 * @param dest_ct
 * @param slot_count
 * @param evaluator
 * @param galois_keys
 */
void total_sum(seal::Ciphertext& target_ct,
               seal::Ciphertext& dest_ct,
               std::size_t slot_count,
               seal::Evaluator& evaluator,
               seal::GaloisKeys& galois_keys);

void encrypt_image(const std::vector<float>& origin_image,
                   std::vector<seal::Ciphertext>& target_cts,
                   const std::size_t image_c,
                   const std::size_t image_h,
                   const std::size_t image_w,
                   const std::size_t slot_count,
                   seal::CKKSEncoder& encoder,
                   seal::Encryptor& encryptor,
                   const double scale);

namespace batch {
void encrypt_images(const types::float2d& origin_images,
                    types::Ciphertext3d& target_ct_3d,
                    const std::size_t slot_count,
                    const std::size_t begin_idx,
                    const std::size_t end_idx,
                    seal::CKKSEncoder& encoder,
                    seal::Encryptor& encryptor,
                    const double scale);
}  // namespace batch

}  // namespace helper::he

namespace helper::json {

picojson::object read_json(const std::string& file_path);

}  // namespace helper::json
