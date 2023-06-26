// sherpa/csrc/online-zipformer-transducer-ctc-bs-model.cc
//
// Copyright (c)  2022  Xiaomi Corporation

#include "sherpa/csrc/online-zipformer-transducer-ctc-bs-model.h"

#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace sherpa {

OnlineZipformerTransducerCtcBsModel::OnlineZipformerTransducerCtcBsModel(
    const std::string &filename, torch::Device device /*= torch::kCPU*/)
    : device_(device) {
  model_ = torch::jit::load(filename, device);
  model_.eval();

  encoder_ = model_.attr("encoder").toModule();
  decoder_ = model_.attr("decoder").toModule();
  joiner_ = model_.attr("joiner").toModule();

  lconv_ = model_.attr("lconv").toModule();
  frame_reducer_ = model_.attr("frame_reducer").toModule();
  ctc_output_ = model_.attr("ctc_output").toModule();

  context_size_ =
      decoder_.attr("conv").toModule().attr("weight").toTensor().size(2);

  // Use 7 here since the subsampling is ((len - 7) // 2 + 1) // 2.
  int32_t pad_length = 7;

  chunk_shift_ = encoder_.attr("decode_chunk_size").toInt() * 2;
  chunk_size_ = chunk_shift_ + pad_length;

  from_torch_jit_trace_ = false;
}

std::tuple<torch::Tensor, torch::Tensor, torch::IValue>
OnlineZipformerTransducerModel::RunEncoder(
    const torch::Tensor &features, const torch::Tensor &features_length,
    const torch::Tensor &num_processed_frames, torch::IValue states) {
  torch::NoGradGuard no_grad;

  // It returns [torch.Tensor, torch.Tensor, Pair[torch.Tensor, torch.Tensor]
  // which are [encoder_out, encoder_out_len, states]
  //
  // We skip the second entry `encoder_out_len` since we assume the
  // feature input is of fixed chunk size and there are no paddings.
  // We can figure out `encoder_out_len` from `encoder_out`.
  torch::List<torch::Tensor> s_list =
      c10::impl::toTypedList<torch::Tensor>(states.toList());
  torch::IValue ivalue =
      encoder_.run_method("forward", features, features_length, states);
  auto tuple_ptr = ivalue.toTuple();
  torch::Tensor encoder_out = tuple_ptr->elements()[0].toTensor();

  torch::Tensor encoder_out_length = tuple_ptr->elements()[1].toTensor();

  auto next_states = tuple_ptr->elements()[2];

  torch::Tensor ctc_output =
      ctc_output_.run_method("forward", encoder_out);

  encoder_out = lconv_.run_method("forward", encoder_out, encoder_out_length)

  ivalue = frame_reducer_.run_method("forward", encoder_out, encoder_out_length, ctc_output, 0)

  tuple_ptr = ivalue.toTuple();
  encoder_out = tuple_ptr->elements()[0].toTensor();
  encoder_out_length = tuple_ptr->elements()[1].toTensor();

  return std::make_tuple(encoder_out, encoder_out_length, next_states);
}

}  // namespace sherpa
