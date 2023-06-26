// sherpa/csrc/online-zipformer-transducer-ctc-bs-model.h
//
// Copyright (c)  2022  Xiaomi Corporation
#ifndef SHERPA_CSRC_ONLINE_ZIPFORMER_TRANSDUCER_CTC_BS_MODEL_H_
#define SHERPA_CSRC_ONLINE_ZIPFORMER_TRANSDUCER_CTC_BS_MODEL_H_
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "sherpa/csrc/online-zipformer-transducer-model.h"

namespace sherpa {
/** This class implements models from pruned_transducer_stateless7_streaming
 * from icefall.
 *
 * See
 * https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless7_ctc_bs/zipformer.py
 * for an instance.
 *
 * You can find the interface and implementation details of the
 * encoder, decoder, and joiner network in the above Python code.
 */
class OnlineZipformerTransducerCtcBsModel : public OnlineZipformerTransducerModel {
 public:
  /** Constructor.
   *
   * @param filename Path to the torchscript model. See
   *                 https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless7_ctc_bs/jit_trace_export.py
   *                 for how to export a model.
   * @param decode_chunk_size  Number of frames before subsampling
   * @param device  Move the model to this device on loading.
   */
  explicit OnlineZipformerTransducerCtcBsModel(const std::string &filename,
                                               torch::Device device = torch::kCPU);

 private:

  torch::jit::Module ctc_output_;
  torch::jit::Module lconv_;
  torch::jit::Module frame_reducer_;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_ONLINE_ZIPFORMER_TRANSDUCER_CTC_BS_MODEL_H_
