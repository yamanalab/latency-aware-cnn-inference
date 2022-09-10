#include <chrono>
#include <fstream>
#include <unordered_map>

#include "cmdline.h"
#include "cnn/network.hpp"
#include "cnn/network_builder.hpp"
#include "utils/constants.hpp"
#include "utils/globals.hpp"
#include "utils/helper.hpp"
#include "utils/load_dataset.hpp"
#include "utils/opt_option.hpp"

using std::cerr;
using std::cout;
using std::endl;
using std::ifstream;
using std::ios;
using std::shared_ptr;
using std::string;
using std::unique_ptr;
using std::vector;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

std::unordered_map<string, std::unordered_map<char, size_t>> CHW_MAP;

void initialize_map_variables() {
  CHW_MAP[constants::dataset::MNIST] = {{'C', 1}, {'H', 28}, {'W', 28}};
  CHW_MAP[constants::dataset::CIFAR10] = {{'C', 3}, {'H', 32}, {'W', 32}};
  constants::activation::POLY_ACT_MAP[constants::activation::RELU_RG5_DEG2] =
      constants::activation::RELU_RG5_DEG2_COEFFS;
  constants::activation::POLY_ACT_MAP[constants::activation::RELU_RG7_DEG2] =
      constants::activation::RELU_RG7_DEG2_COEFFS;
  constants::activation::POLY_ACT_MAP[constants::activation::SWISH_RG5_DEG2] =
      constants::activation::SWISH_RG5_DEG2_COEFFS;
  constants::activation::POLY_ACT_MAP[constants::activation::SWISH_RG7_DEG2] =
      constants::activation::SWISH_RG7_DEG2_COEFFS;
  constants::activation::POLY_ACT_MAP[constants::activation::MISH_RG5_DEG2] =
      constants::activation::MISH_RG5_DEG2_COEFFS;
  constants::activation::POLY_ACT_MAP[constants::activation::MISH_RG7_DEG2] =
      constants::activation::MISH_RG7_DEG2_COEFFS;
  constants::activation::POLY_ACT_MAP[constants::activation::RELU_RG5_DEG4] =
      constants::activation::RELU_RG5_DEG4_COEFFS;
  constants::activation::POLY_ACT_MAP[constants::activation::RELU_RG7_DEG4] =
      constants::activation::RELU_RG7_DEG4_COEFFS;
  constants::activation::POLY_ACT_MAP[constants::activation::SWISH_RG5_DEG4] =
      constants::activation::SWISH_RG5_DEG4_COEFFS;
  constants::activation::POLY_ACT_MAP[constants::activation::SWISH_RG7_DEG4] =
      constants::activation::SWISH_RG7_DEG4_COEFFS;
  constants::activation::POLY_ACT_MAP[constants::activation::MISH_RG5_DEG4] =
      constants::activation::MISH_RG5_DEG4_COEFFS;
  constants::activation::POLY_ACT_MAP[constants::activation::MISH_RG7_DEG4] =
      constants::activation::MISH_RG7_DEG4_COEFFS;
  constants::activation::OPT_POLY_ACT_MAP
      [constants::activation::RELU_RG5_DEG2] = {
          constants::activation::RELU_RG5_DEG2_OPT_COEFFS,
          constants::activation::RELU_RG5_DEG2_COEFFS.front()};
  constants::activation::OPT_POLY_ACT_MAP
      [constants::activation::RELU_RG7_DEG2] = {
          constants::activation::RELU_RG7_DEG2_OPT_COEFFS,
          constants::activation::RELU_RG7_DEG2_COEFFS.front()};
  constants::activation::OPT_POLY_ACT_MAP
      [constants::activation::SWISH_RG5_DEG2] = {
          constants::activation::SWISH_RG5_DEG2_OPT_COEFFS,
          constants::activation::SWISH_RG5_DEG2_COEFFS.front()};
  constants::activation::OPT_POLY_ACT_MAP
      [constants::activation::SWISH_RG7_DEG2] = {
          constants::activation::SWISH_RG7_DEG2_OPT_COEFFS,
          constants::activation::SWISH_RG7_DEG2_COEFFS.front()};
  constants::activation::OPT_POLY_ACT_MAP
      [constants::activation::MISH_RG5_DEG2] = {
          constants::activation::MISH_RG5_DEG2_OPT_COEFFS,
          constants::activation::MISH_RG5_DEG2_COEFFS.front()};
  constants::activation::OPT_POLY_ACT_MAP
      [constants::activation::MISH_RG7_DEG2] = {
          constants::activation::MISH_RG7_DEG2_OPT_COEFFS,
          constants::activation::MISH_RG7_DEG2_COEFFS.front()};
  constants::activation::OPT_POLY_ACT_MAP
      [constants::activation::RELU_RG5_DEG4] = {
          constants::activation::RELU_RG5_DEG4_OPT_COEFFS,
          constants::activation::RELU_RG5_DEG4_COEFFS.front()};
  constants::activation::OPT_POLY_ACT_MAP
      [constants::activation::RELU_RG7_DEG4] = {
          constants::activation::RELU_RG7_DEG4_OPT_COEFFS,
          constants::activation::RELU_RG7_DEG4_COEFFS.front()};
  constants::activation::OPT_POLY_ACT_MAP
      [constants::activation::SWISH_RG5_DEG4] = {
          constants::activation::SWISH_RG5_DEG4_OPT_COEFFS,
          constants::activation::SWISH_RG5_DEG4_COEFFS.front()};
  constants::activation::OPT_POLY_ACT_MAP
      [constants::activation::SWISH_RG7_DEG4] = {
          constants::activation::SWISH_RG7_DEG4_OPT_COEFFS,
          constants::activation::SWISH_RG7_DEG4_COEFFS.front()};
  constants::activation::OPT_POLY_ACT_MAP
      [constants::activation::MISH_RG5_DEG4] = {
          constants::activation::MISH_RG5_DEG4_OPT_COEFFS,
          constants::activation::MISH_RG5_DEG4_COEFFS.front()};
  constants::activation::OPT_POLY_ACT_MAP
      [constants::activation::MISH_RG7_DEG4] = {
          constants::activation::MISH_RG7_DEG4_OPT_COEFFS,
          constants::activation::MISH_RG7_DEG4_COEFFS.front()};
}

template <typename T>
types::vector3d<double> flatten_images_per_channel(
    const types::vector4d<T>& images) {
  size_t n = images.size(), c = images.at(0).size(),
         h = images.at(0).at(0).size(), w = images.at(0).at(0).at(0).size();
  types::double3d flattened(n, types::double2d(c, vector<double>(h * w)));

  for (size_t n_i = 0; n_i < n; ++n_i) {
    for (size_t c_i = 0; c_i < c; ++c_i) {
      for (size_t h_i = 0; h_i < h; ++h_i) {
        for (size_t w_i = 0; w_i < w; ++w_i) {
          flattened[n_i][c_i][h_i * h + w_i] = images[n_i][c_i][h_i][w_i];
        }
      }
    }
  }

  return flattened;
}

int main(int argc, char* argv[]) {
  /* Initialize map */
  initialize_map_variables();

  /* Parse command line args */
  cmdline::parser parser;
  parser.add<string>("secret-prefix", 'P', "Prefix of filename of keys");
  parser.add<string>("dataset", 'D', "Dataset name", false,
                     constants::dataset::MNIST,
                     cmdline::oneof<string>(constants::dataset::MNIST,
                                            constants::dataset::CIFAR10));
  parser.add<string>("model", 'M', "Model name trained with PyTorch");
  parser.add<string>("model-structure", 0, "Model structure file name (json)",
                     false, "default");
  parser.add<string>("model-params", 0, "Model params file name (h5)");
  parser.add<string>("activation", 'A', "Activation function name");
  parser.add("fuse-layer", 0,
             "Whether to apply optimization which fuse linear layers (e.g. "
             "Conv2d + BatchNormalization)");
  parser.add("opt-act", 0,
             "Whether to apply optimization which fold coefficient of "
             "polynomial approx. activation function");
  parser.add("opt-pool", 0,
             "Whether to apply optimization which fold coefficient of average "
             "pooling");
  parser.add<string>(
      "mode", 0,
      "Inference mode "
      "(single: channel-wise packing, batch: batch-axis packing)",
      false, constants::mode::SINGLE,
      cmdline::oneof<string>(constants::mode::SINGLE, constants::mode::BATCH));
  parser.add<size_t>("images", 0, "How many images to execute inference", false,
                     0);
  parser.add<size_t>("trials", 'N',
                     "Number of trials to evaluate inference on test dataset",
                     false, 1);

  parser.parse_check(argc, argv);
  const string secret_fname_prefix = parser.get<string>("secret-prefix");
  const string dataset_name = parser.get<string>("dataset");
  const string model_name = parser.get<string>("model");
  const string model_structure_fname = parser.get<string>("model-structure");
  const string model_params_fname = parser.get<string>("model-params");
  const string activation_str = parser.get<string>("activation");
  const bool enable_fuse_linear_layer = parser.exist("fuse-layer");
  const bool enable_fold_act_coeff = parser.exist("opt-act");
  const bool enable_fold_pool_coeff = parser.exist("opt-pool");
  OPT_OPTION = OptOption(enable_fuse_linear_layer, enable_fold_act_coeff,
                         enable_fold_pool_coeff);
  const string inference_mode = parser.get<string>("mode");
  size_t inference_image_count = parser.get<size_t>("images");
  size_t inference_trial_count = parser.get<size_t>("trials");

  if (inference_mode == constants::mode::SINGLE &&
      activation_str == "swish_rg7_deg4" &&
      dataset_name == constants::dataset::MNIST) {
    ROUND_THRESHOLD = 1e-6;
  } else {
    ROUND_THRESHOLD = 1e-7;
  }

  const string saved_models_path =
      constants::fname::TRAIN_MODEL_DIR + "/" + dataset_name + "/saved_models/";
  string model_structure_path;
  if (model_structure_fname == "default") {
    model_structure_path = saved_models_path + model_name + "-structure.json";
  } else {
    model_structure_path = saved_models_path + model_structure_fname;
  }
  const string model_params_path = saved_models_path + model_params_fname;

  unique_ptr<ifstream> ifs_ptr;
  auto secrets_ifs =
      [&](const string& fname_suffix) -> const unique_ptr<ifstream>& {
    string fname;
    if (inference_mode == constants::mode::SINGLE) {
      fname = dataset_name + "_" + secret_fname_prefix + fname_suffix;
    } else {
      fname = secret_fname_prefix + fname_suffix;
    }
    ifs_ptr.reset(
        new ifstream(constants::fname::SECRETS_DIR + "/" + fname, ios::binary));
    return ifs_ptr;
  };

  /* Initialize global variables */
  CONSUMED_LEVEL = 0;
  SHOULD_MUL_ACT_COEFF = false;
  SHOULD_MUL_POOL_COEFF = false;
  if (activation_str.compare(constants::activation::SQUARE) == 0) {
    ACTIVATION_TYPE = EActivationType::SQUARE;
  } else if (activation_str.find("deg2") != string::npos) {
    ACTIVATION_TYPE = EActivationType::DEG2_POLY_APPROX;
  } else if (activation_str.find("deg4") != string::npos) {
    ACTIVATION_TYPE = EActivationType::DEG4_POLY_APPROX;
  } else {
    throw std::logic_error("Unsupported activation function: " +
                           activation_str);
  }
  if (ACTIVATION_TYPE == EActivationType::DEG2_POLY_APPROX ||
      ACTIVATION_TYPE == EActivationType::DEG4_POLY_APPROX) {
    if (OPT_OPTION.enable_fold_act_coeff) {
      if (auto map_iter =
              constants::activation::OPT_POLY_ACT_MAP.find(activation_str);
          map_iter != constants::activation::OPT_POLY_ACT_MAP.end()) {
        POLY_ACT_COEFFS = map_iter->second.first;
        POLY_ACT_HIGHEST_DEG_COEFF = map_iter->second.second;
      } else {
        throw std::logic_error("Unsupported activation function: " +
                               activation_str);
      }
    } else {
      if (auto map_iter =
              constants::activation::POLY_ACT_MAP.find(activation_str);
          map_iter != constants::activation::POLY_ACT_MAP.end()) {
        POLY_ACT_COEFFS = map_iter->second;
      } else {
        throw std::logic_error("Unsupported activation function: " +
                               activation_str);
      }
    }
  }

  /* Load seal params */
  seal::EncryptionParameters params;
  params.load(*secrets_ifs(constants::fname::PARAMS_SUFFIX));

  shared_ptr<seal::SEALContext> context(new seal::SEALContext(params));
  helper::he::print_parameters(context);
  cout << endl;

  /* Create seal keys */
  // seal::KeyGenerator keygen(*context);
  // auto secret_key = keygen.secret_key();
  // seal::PublicKey public_key;
  // keygen.create_public_key(public_key);
  // seal::RelinKeys relin_keys;
  // keygen.create_relin_keys(relin_keys);

  /* Load seal keys */
  cout << "Loading seal keys..." << endl;
  shared_ptr<seal::SecretKey> secret_key(new seal::SecretKey);
  secret_key->load(*context, *secrets_ifs(constants::fname::SECRET_KEY_SUFFIX));
  shared_ptr<seal::PublicKey> public_key(new seal::PublicKey);
  public_key->load(*context, *secrets_ifs(constants::fname::PUBLIC_KEY_SUFFIX));
  shared_ptr<seal::RelinKeys> relin_keys(new seal::RelinKeys);
  relin_keys->load(*context, *secrets_ifs(constants::fname::RELIN_KEYS_SUFFIX));
  shared_ptr<seal::GaloisKeys> galois_keys(new seal::GaloisKeys);
  galois_keys->load(*context,
                    *secrets_ifs(constants::fname::GALOIS_KEYS_SUFFIX));
  GALOIS_KEYS.load(*context,
                   *secrets_ifs(constants::fname::GALOIS_KEYS_SUFFIX));
  cout << "Finish loading keys!\n" << endl;

  seal::Evaluator evaluator(*context);
  seal::CKKSEncoder encoder(*context);
  seal::Encryptor encryptor(*context, *public_key);
  seal::Decryptor decryptor(*context, *secret_key);

  const size_t log2_f = std::log2(params.coeff_modulus()[1].value() - 1) + 1;
  const double scale = static_cast<double>(static_cast<uint64_t>(1) << log2_f);
  cout << "scale: 2^" << log2_f << "(" << scale << ")" << endl;
  const size_t slot_count = encoder.slot_count();
  cout << "# of slots: " << slot_count << endl;
  cout << endl;

  // shared_ptr<helper::he::SealTool> seal_tool =
  //     std::make_shared<helper::he::SealTool>(evaluator, encoder, *relin_keys,
  //                                            *galois_keys, slot_count,
  //                                            scale);
  shared_ptr<helper::he::SealTool> seal_tool =
      std::make_shared<helper::he::SealTool>(evaluator, encoder, decryptor,
                                             *relin_keys, slot_count, scale);

  /* Load test dataset for inference */
  cout << "Loading test images & labels..." << endl;
  types::float2d test_images = utils::load_test_images(dataset_name);
  vector<unsigned char> test_labels = utils::load_test_labels(dataset_name);

  const size_t input_n = test_images.size(),
               input_c = CHW_MAP[dataset_name]['C'],
               input_h = CHW_MAP[dataset_name]['H'],
               input_w = CHW_MAP[dataset_name]['W'], label_size = 10;

  cout << "Finish loading!" << endl;
  cout << "Shape of test images: [" << input_n << ", " << input_c << ", "
       << input_h << ", " << input_w << "]" << endl;
  cout << endl;

  /* Channel-wise packed ciphertext inference */
  if (inference_mode == constants::mode::SINGLE) {
    {
      INPUT_C = input_c, INPUT_H = input_h, INPUT_W = input_w;
      INPUT_HW_SLOT_IDX.resize(INPUT_H);
      int counter = 0;
      for (size_t i = 0; i < INPUT_H; ++i) {
        INPUT_HW_SLOT_IDX[i].resize(INPUT_W);
        for (size_t j = 0; j < INPUT_W; ++j) {
          INPUT_HW_SLOT_IDX[i][j] = counter++;
        }
      }
      // register rotation steps for total sum (log2(slot_count))
      size_t total_sum_rotate_count =
          std::ceil(std::log2(seal_tool->slot_count()));
      for (std::size_t i = 0; i < total_sum_rotate_count; ++i) {
        USE_ROTATION_STEPS.insert(static_cast<int>(std::pow(2, i)));
      }
    }
    using namespace cnn::encrypted;
    /* Build network */
    cout << "Building network from trained model..." << endl;
    Network network;
    auto build_network_begin_time = high_resolution_clock::now();
    try {
      network = NetworkBuilder::build(model_structure_path, model_params_path,
                                      seal_tool);
    } catch (const std::runtime_error& re) {
      cerr << "Error has occurred in building network." << endl;
      cerr << "RuntimeError: " << re.what() << endl;
      return 1;
    } catch (const std::exception& e) {
      cerr << "Error has occurred in building network." << endl;
      cerr << "Exception: " << e.what() << endl;
      return 1;
    }
    auto build_network_end_time = high_resolution_clock::now();
    duration<double> build_network_sec =
        build_network_end_time - build_network_begin_time;
    cout << "Finish building! (" << build_network_sec.count() << " sec)\n"
         << endl;

    /* Create galois keys */
    // USE_ROTATION_STEPS.erase(0);
    // cout << "Creating " << USE_ROTATION_STEPS.size() << " galois keys..."
    //      << endl;
    // const vector<int> rotation_steps(USE_ROTATION_STEPS.begin(),
    //                                  USE_ROTATION_STEPS.end());
    // for (const int step : rotation_steps) {
    //   cout << step << ", ";
    // }
    // cout << endl;
    // keygen.create_galois_keys(rotation_steps, GALOIS_KEYS);
    // cout << "Finish creating!\n" << endl;

    /* Execute inference on test data */
    inference_image_count =
        inference_image_count == 0 ? input_n : inference_image_count;
    vector<seal::Ciphertext> enc_results(label_size);
    seal::Plaintext plain_result;
    vector<double> tmp_results(slot_count);
    vector<vector<double>> results(inference_image_count,
                                   vector<double>(label_size));
    size_t correct_prediction_count = 0;
    duration<double> sum_encryption_sec = duration<double>::zero(),
                     sum_inference_sec = duration<double>::zero(),
                     sum_decryption_sec = duration<double>::zero();
    for (size_t i = 0; i < inference_image_count; ++i) {
      cout << "\t<Image " << i + 1 << "/" << inference_image_count << ">\n"
           << "\tEncrypting image per channel..." << endl;
      /* Encrypt image */
      vector<seal::Ciphertext> enc_channel_wise_packed_image(input_c);
      auto encrypt_image_begin_time = high_resolution_clock::now();
      helper::he::encrypt_image(test_images[i], enc_channel_wise_packed_image,
                                input_c, input_h, input_w, slot_count, encoder,
                                encryptor, scale);
      auto encrypt_image_end_time = high_resolution_clock::now();
      duration<double> encrypt_image_sec =
          encrypt_image_end_time - encrypt_image_begin_time;
      sum_encryption_sec += encrypt_image_sec;
      cout << "\tFinish encrypting! (" << encrypt_image_sec.count() << " sec)\n"
           << "\t  encrypted image has " << enc_channel_wise_packed_image.size()
           << " channels\n"
           << endl;
      // {
      //   cout << "Image [" << i << "]:" << endl;
      //   // size_t counter = 0;
      //   size_t pos;
      //   for (size_t c = 0; c < input_c; ++c) {
      //     cout << "Channel " << c << endl;
      //     for (size_t h = 0; h < input_h; ++h) {
      //       for (size_t w = 0; w < input_w; ++w) {
      //         pos = c * (input_h * input_w) + h * input_w + w;
      //         cout << test_images[i][pos] << ", ";
      //       }
      //       cout << endl;
      //     }
      //   }
      //   // seal::Ciphertext enc_image_channel =
      //   // enc_channel_wise_packed_image[0]; seal::Plaintext plain_y;
      //   // std::vector<double> y_values(seal_tool->slot_count());
      //   // seal_tool->decryptor().decrypt(enc_image_channel, plain_y);
      //   // seal_tool->encoder().decode(plain_y, y_values);
      //   // cout << "Decrypted image [" << i << "]: " << endl;
      //   // counter = 0;
      //   // for (size_t h = 0; h < input_h; ++h) {
      //   //   for (size_t w = 0; w < input_w; ++w) {
      //   //     cout << y_values[counter++] << ", ";
      //   //   }
      //   //   cout << endl;
      //   // }
      // }

      /* Execute inference */
      cout << "\tExecuting inference..." << endl;
      auto inference_begin_time = high_resolution_clock::now();
      enc_results = network.predict(enc_channel_wise_packed_image);
      auto inference_end_time = high_resolution_clock::now();

      duration<double> inference_sec =
          inference_end_time - inference_begin_time;
      sum_inference_sec += inference_sec;
      cout << "\tFinish executing inference! (" << inference_sec.count()
           << " sec)\n"
           << endl;

      /* Decrypt inference results */
      cout << "\tDecrypting inference result..." << endl;
      auto decrypt_result_begin_time = high_resolution_clock::now();
      for (size_t label = 0; label < label_size; label++) {
        decryptor.decrypt(enc_results[label], plain_result);
        encoder.decode(plain_result, tmp_results);
        // {
        //   cout << "\tLabel: " << label << endl;
        //   cout << "\tresults[0]: " << tmp_results[0] << endl;
        // }
        results[i][label] = tmp_results[0];
      }
      auto decrypt_result_end_time = high_resolution_clock::now();

      duration<double> decrypt_result_sec =
          decrypt_result_end_time - decrypt_result_begin_time;
      sum_decryption_sec += decrypt_result_sec;
      cout << "\tFinish decrypting! (" << decrypt_result_sec.count() << " sec)"
           << endl;

      vector<double>::iterator begin_iter = results[i].begin();
      vector<double>::iterator max_iter =
          std::max_element(begin_iter, results[i].end());
      size_t predicted_label = std::distance(begin_iter, max_iter);
      size_t correct_label = static_cast<size_t>(test_labels[i]);
      cout << "\tPredicted: " << predicted_label
           << ", Correct: " << correct_label << "\n"
           << endl;
      if (predicted_label == correct_label) {
        correct_prediction_count++;
      }
    }

    /* Calculate accuracy */
    cout << "Calculating accuracy..." << endl;
    const double accuracy =
        static_cast<double>(correct_prediction_count) / inference_image_count;
    cout << "Finish calculating!\n" << endl;
    cout << "Accuracy: " << accuracy << " (" << correct_prediction_count << "/"
         << inference_image_count << ")\n"
         << endl;

    /* Output average execution time */
    duration<double> average_encryption_sec =
        sum_encryption_sec / inference_image_count;
    duration<double> average_prediction_sec =
        sum_inference_sec / inference_image_count;
    duration<double> average_decryption_sec =
        sum_decryption_sec / inference_image_count;
    cout << "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
         << "Average of " << inference_image_count
         << " encryption: " << average_encryption_sec.count() << " sec\n"
         << "Average of " << inference_image_count
         << " prediction: " << average_prediction_sec.count() << " sec\n"
         << "Average of " << inference_image_count
         << " decryption: " << average_decryption_sec.count() << " sec\n"
         << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
         << endl;
  }

  /* Batch-axis packed ciphertext inference */
  if (inference_mode == constants::mode::BATCH) {
    using namespace cnn::encrypted::batch;
    /* Build network */
    cout << "Building network from trained model..." << endl;
    Network network;
    auto build_network_begin_time = high_resolution_clock::now();
    try {
      network = NetworkBuilder::build(model_structure_path, model_params_path,
                                      seal_tool);
    } catch (const std::runtime_error& re) {
      cerr << "Error has occurred in building network." << endl;
      cerr << "RuntimeError: " << re.what() << endl;
      return 1;
    } catch (const std::exception& e) {
      cerr << "Error has occurred in building network." << endl;
      cerr << "Exception: " << e.what() << endl;
      return 1;
    }
    auto build_network_end_time = high_resolution_clock::now();
    duration<double> build_network_sec =
        build_network_end_time - build_network_begin_time;
    cout << "Finish building! (" << build_network_sec.count() << " sec)" << endl
         << endl;

    /* Execute inference on test data */
    inference_image_count =
        inference_image_count == 0 ? input_n : inference_image_count;
    size_t remain_image_count = inference_image_count;
    vector<seal::Ciphertext> enc_results(label_size);
    vector<seal::Plaintext> plain_results(label_size);
    vector<vector<double>> results(inference_image_count,
                                   vector<double>(label_size));
    vector<double> tmp_results(slot_count);
    duration<double> sum_encryption_trials_sec, sum_inference_trials_sec,
        sum_decryption_trials_sec;

    const size_t step_count =
        std::ceil(static_cast<double>(inference_image_count) / slot_count);
    cout << "# of inference trials: " << inference_trial_count << endl;
    cout << "# of steps: " << step_count << endl << endl;
    for (size_t step = 0, image_count_in_step; step < step_count; ++step) {
      cout << "Step " << step + 1 << ":\n"
           << "\t--------------------------------------------------" << endl;

      sum_encryption_trials_sec = duration<double>::zero();
      sum_inference_trials_sec = duration<double>::zero();
      sum_decryption_trials_sec = duration<double>::zero();
      image_count_in_step =
          (slot_count < remain_image_count) ? slot_count : remain_image_count;
      const size_t begin_idx = step * slot_count;
      const size_t end_idx = begin_idx + image_count_in_step;

      for (size_t n = 0; n < inference_trial_count; ++n) {
        cout << "\t<Trial " << n + 1 << ">\n"
             << "\tEncrypting " << image_count_in_step << " images..." << endl;

        /* Encrypt images in step */
        types::Ciphertext3d enc_packed_images(
            input_c,
            types::Ciphertext2d(input_h, vector<seal::Ciphertext>(input_w)));
        auto step_encrypt_images_begin_time = high_resolution_clock::now();
        helper::he::batch::encrypt_images(test_images, enc_packed_images,
                                          slot_count, begin_idx, end_idx,
                                          encoder, encryptor, scale);
        auto step_encrypt_images_end_time = high_resolution_clock::now();
        duration<double> step_encrypt_images_sec =
            step_encrypt_images_end_time - step_encrypt_images_begin_time;
        sum_encryption_trials_sec += step_encrypt_images_sec;
        cout << "\tFinish encrypting! (" << step_encrypt_images_sec.count()
             << " sec)\n"
             << "\t  encrypted packed images shape: "
             << enc_packed_images.size() << "x"
             << enc_packed_images.at(0).size() << "x"
             << enc_packed_images.at(0).at(0).size() << "\n"
             << endl;

        /* Execute inference */
        cout << "\tExecuting inference..." << endl;
        auto step_inference_begin_time = high_resolution_clock::now();
        enc_results = network.predict(enc_packed_images);
        auto step_inference_end_time = high_resolution_clock::now();

        duration<double> step_inference_sec =
            step_inference_end_time - step_inference_begin_time;
        sum_inference_trials_sec += step_inference_sec;
        cout << "\tFinish executing inference! (" << step_inference_sec.count()
             << " sec)\n"
             << "\t  encrypted results size: " << enc_results.size() << "\n"
             << endl;

        /* Decrypt inference results */
        cout << "\tDecrypting inference results..." << endl;
        auto step_decrypt_results_begin_time = high_resolution_clock::now();
        for (size_t label = 0; label < label_size; ++label) {
          decryptor.decrypt(enc_results[label], plain_results[label]);
          encoder.decode(plain_results[label], tmp_results);

          for (size_t image_idx = begin_idx, counter = 0; image_idx < end_idx;
               ++image_idx) {
            results[image_idx][label] = tmp_results[counter++];
          }
        }
        auto step_decrypt_results_end_time = high_resolution_clock::now();

        duration<double> step_decrypt_results_sec =
            step_decrypt_results_end_time - step_decrypt_results_begin_time;
        sum_decryption_trials_sec += step_decrypt_results_sec;
        cout << "\tFinish decrypting! (" << step_decrypt_results_sec.count()
             << " sec)\n"
             << "\t--------------------------------------------------" << endl;
      }
      duration<double> average_encryption_trials_sec =
          sum_encryption_trials_sec / inference_trial_count;
      duration<double> average_prediction_trials_sec =
          sum_inference_trials_sec / inference_trial_count;
      duration<double> average_decryption_trials_sec =
          sum_decryption_trials_sec / inference_trial_count;
      cout << "\n\t~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
           << "\tAverage of " << inference_trial_count
           << " encryption trials: " << average_encryption_trials_sec.count()
           << " sec\n"
           << "\tAverage of " << inference_trial_count
           << " prediction trials: " << average_prediction_trials_sec.count()
           << " sec\n"
           << "\tAverage of " << inference_trial_count
           << " decryption trials: " << average_decryption_trials_sec.count()
           << " sec\n"
           << "\t~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
           << endl;
      remain_image_count -= image_count_in_step;
    }

    /* Calculate accuracy */
    vector<double>::iterator begin_iter, max_iter;
    size_t predicted_label, correct_prediction_count = 0;

    cout << "Calculating accuracy..." << endl;
    for (size_t image_idx = 0; image_idx < inference_image_count; ++image_idx) {
      vector<double> predict_outputs = results[image_idx];
      begin_iter = predict_outputs.begin();
      max_iter = std::max_element(begin_iter, predict_outputs.end());
      predicted_label = distance(begin_iter, max_iter);

      if (predicted_label == static_cast<size_t>(test_labels[image_idx])) {
        correct_prediction_count++;
      }
    }

    const double accuracy =
        static_cast<double>(correct_prediction_count) / inference_image_count;
    cout << "Finish calculating!\n" << endl;
    cout << "Accuracy: " << accuracy << " (" << correct_prediction_count << "/"
         << inference_image_count << ")\n"
         << endl;
  }

  return 0;
}
