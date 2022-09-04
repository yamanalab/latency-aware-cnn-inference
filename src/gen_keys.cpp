#include <seal/seal.h>
#include <experimental/filesystem>
#include <fstream>

#include "cmdline.h"
#include "utils/constants.hpp"

using std::ios;
using std::ofstream;
using std::size_t;
using std::string;
using std::unique_ptr;
using std::vector;
namespace fs = std::experimental::filesystem;

vector<int> generate_log2_modulus(size_t level,
                                  size_t log2_q0,
                                  size_t log2_qi,
                                  size_t log2_ql) {
  vector<int> modulus(level + 2);
  modulus[0] = log2_q0;
  for (size_t i = 1; i <= level; ++i) {
    modulus[i] = log2_qi;
  }
  modulus[level + 1] = log2_ql;

  return modulus;
}

int main(int argc, char* argv[]) {
  cmdline::parser parser;

  parser.add<size_t>("poly-deg", 'N', "Degree of polynomial ring");
  parser.add<size_t>("level", 'L',
                     "Initial level of ciphertext (Multiplicative depth)");
  parser.add<size_t>("q0", 0, "Bit number of the first prime in coeff_modulus");
  parser.add<size_t>("qi", 0,
                     "Bit number of intermediate primes in coeff_modulus");
  parser.add<size_t>("ql", 0, "Bit number of the last prime in coeff_modulus");
  parser.add<string>("prefix", 0, "Prefix of the generating file name");
  parser.add<string>("dataset", 'D', "Dataset name", false, "unknown");

  parser.parse_check(argc, argv);
  const size_t poly_modulus_degree = parser.get<size_t>("poly-deg");
  const size_t level = parser.get<size_t>("level");
  const size_t log2_q0 = parser.get<size_t>("q0");
  const size_t log2_qi = parser.get<size_t>("qi");
  const size_t log2_ql = parser.get<size_t>("ql");
  const string fname_prefix = parser.get<string>("prefix");
  const string dataset_name = parser.get<string>("dataset");

  vector<int> prime_bit_sizes =
      generate_log2_modulus(level, log2_q0, log2_qi, log2_ql);

  seal::EncryptionParameters params(seal::scheme_type::ckks);
  params.set_poly_modulus_degree(poly_modulus_degree);
  params.set_coeff_modulus(
      seal::CoeffModulus::Create(poly_modulus_degree, prime_bit_sizes));

  seal::SEALContext context(params);
  seal::KeyGenerator keygen(context);
  auto secret_key = keygen.secret_key();
  seal::PublicKey public_key;
  keygen.create_public_key(public_key);
  seal::RelinKeys relin_keys;
  keygen.create_relin_keys(relin_keys);
  seal::GaloisKeys galois_keys;
  if (dataset_name == "mnist") {
    const vector<int> rotation_steps{
        -529, -528, -515, -514, -513, -512, -499, -498, -497, -496, -483, -482,
        -481, -480, -467, -466, -465, -464, -451, -450, -449, -448, -99,  -98,
        -97,  -96,  -83,  -82,  -81,  -80,  -67,  -66,  -65,  -64,  -51,  -50,
        -49,  -48,  -35,  -34,  -33,  -32,  -19,  -18,  -17,  -16,  -3,   -2,
        -1,   1,    2,    3,    4,    6,    8,    16,   28,   29,   30,   31,
        32,   56,   57,   58,   59,   60,   62,   64,   84,   85,   86,   87,
        88,   112,  113,  114,  115,  116,  118,  120,  128,  168,  170,  172,
        174,  176,  224,  226,  228,  230,  232,  256,  512,  1024, 2048, 4096};
    keygen.create_galois_keys(rotation_steps, galois_keys);
  } else if (dataset_name == "cifar-10") {
    const vector<int> rotation_steps{
        -132, -128, -127, -126, -125, -124, -123, -122, -121, -120, -119, -118,
        -117, -116, -115, -114, -113, -112, -111, -110, -109, -108, -107, -106,
        -105, -104, -103, -102, -101, -100, -99,  -98,  -97,  -96,  -95,  -94,
        -93,  -92,  -91,  -90,  -89,  -88,  -87,  -86,  -85,  -84,  -83,  -82,
        -81,  -80,  -79,  -78,  -77,  -76,  -75,  -74,  -73,  -72,  -71,  -70,
        -69,  -68,  -67,  -66,  -65,  -64,  -63,  -62,  -61,  -60,  -59,  -58,
        -57,  -56,  -55,  -54,  -53,  -52,  -51,  -50,  -49,  -48,  -47,  -46,
        -45,  -44,  -43,  -42,  -41,  -40,  -39,  -38,  -37,  -36,  -35,  -34,
        -33,  -32,  -31,  -30,  -29,  -28,  -27,  -26,  -25,  -24,  -23,  -22,
        -21,  -20,  -19,  -18,  -17,  -16,  -15,  -14,  -13,  -12,  -11,  -10,
        -9,   -8,   -7,   -6,   -5,   -4,   -3,   -2,   -1,   1,    2,    4,
        8,    16,   24,   31,   32,   33,   62,   64,   66,   124,  128,  132,
        256,  264,  272,  280,  512,  520,  528,  536,  768,  776,  784,  792,
        1024, 2048, 4096};
    keygen.create_galois_keys(rotation_steps, galois_keys);
  } else {
    keygen.create_galois_keys(galois_keys);
  }

  unique_ptr<ofstream> ofs_ptr;
  auto secrets_ofs =
      [&](const string& fname_suffix) -> const unique_ptr<ofstream>& {
    string fname;
    if (dataset_name == "mnist" || dataset_name == "cifar-10") {
      fname = dataset_name + "_" + fname_prefix + fname_suffix;
    } else {
      fname = fname_prefix + fname_suffix;
    }
    ofs_ptr.reset(
        new ofstream(constants::fname::SECRETS_DIR + "/" + fname, ios::binary));
    return ofs_ptr;
  };

  if (!fs::exists(constants::fname::SECRETS_DIR))
    fs::create_directory(constants::fname::SECRETS_DIR);

  params.save(*secrets_ofs(constants::fname::PARAMS_SUFFIX));
  secret_key.save(*secrets_ofs(constants::fname::SECRET_KEY_SUFFIX));
  public_key.save(*secrets_ofs(constants::fname::PUBLIC_KEY_SUFFIX));
  relin_keys.save(*secrets_ofs(constants::fname::RELIN_KEYS_SUFFIX));
  galois_keys.save(*secrets_ofs(constants::fname::GALOIS_KEYS_SUFFIX));

  return 0;
}
