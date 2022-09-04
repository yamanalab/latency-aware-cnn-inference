#include <fstream>

#include "cmdline.h"
#include "utils/constants.hpp"
#include "utils/helper.hpp"

using std::ifstream;
using std::ios;
using std::shared_ptr;
using std::string;
using std::unique_ptr;
using std::vector;

int main(int argc, char* argv[]) {
  cmdline::parser parser;

  parser.add<string>("prefix", 0, "Prefix of filename of keys");
  parser.parse_check(argc, argv);
  const string fname_prefix = parser.get<string>("prefix");

  unique_ptr<ifstream> ifs_ptr;
  auto secrets_ifs =
      [&](const string& fname_suffix) -> const unique_ptr<ifstream>& {
    ifs_ptr.reset(new ifstream(
        constants::fname::SECRETS_DIR + "/" + fname_prefix + fname_suffix,
        ios::binary));
    return ifs_ptr;
  };

  seal::EncryptionParameters params;
  params.load(*secrets_ifs(constants::fname::PARAMS_SUFFIX));

  shared_ptr<seal::SEALContext> context(new seal::SEALContext(params));
  helper::he::print_parameters(context);
  std::cout << std::endl;

  shared_ptr<seal::SecretKey> secret_key(new seal::SecretKey);
  secret_key->load(*context, *secrets_ifs(constants::fname::SECRET_KEY_SUFFIX));
  shared_ptr<seal::PublicKey> public_key(new seal::PublicKey);
  public_key->load(*context, *secrets_ifs(constants::fname::PUBLIC_KEY_SUFFIX));
  shared_ptr<seal::RelinKeys> relin_keys(new seal::RelinKeys);
  relin_keys->load(*context, *secrets_ifs(constants::fname::RELIN_KEYS_SUFFIX));
  shared_ptr<seal::GaloisKeys> galois_keys(new seal::GaloisKeys);
  galois_keys->load(*context,
                    *secrets_ifs(constants::fname::GALOIS_KEYS_SUFFIX));

  seal::CKKSEncoder encoder(*context);
  seal::Encryptor encryptor(*context, *public_key);
  seal::Decryptor decryptor(*context, *secret_key);

  // size_t poly_modulus_degree = params.poly_modulus_degree();
  size_t log2_f = std::log2(params.coeff_modulus()[1].value() - 1) + 1;
  double scale = static_cast<double>(static_cast<uint64_t>(1) << log2_f);
  std::cout << "scale: 2^" << log2_f << "(" << scale << ")" << std::endl;
  size_t slot_count = encoder.slot_count();
  std::cout << "# of slots: " << slot_count << std::endl;

  return 0;
}
