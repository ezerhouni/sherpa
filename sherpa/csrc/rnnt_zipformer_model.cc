/**
 * Copyright (c)  2022  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "sherpa/csrc/rnnt_zipformer_model.h"

#include <tuple>

namespace sherpa {


RnntZipformerModel::RnntZipformerModel(
    const std::string &filename, torch::Device device /*= torch::kCPU*/)
    : device_(device) {
  model_ = torch::jit::load(filename, device);
  model_.eval();

  encoder_ = model_.attr("encoder").toModule();
  decoder_ = model_.attr("decoder").toModule();
  joiner_ = model_.attr("joiner").toModule();

  context_size_ =
      decoder_.attr("conv").toModule().attr("weight").toTensor().size(2);

  // Use 7 here since the subsampling is ((len - 7) // 2 + 1) // 2.
  int32_t pad_length = 7;

  chunk_shift_ = encoder_.attr("decode_chunk_size").toInt() * 2;
  chunk_size_ = chunk_shift_ + pad_length;

  from_torch_jit_trace_ = false;
}


torch::IValue RnntZipformerModel::StackStates(
    const std::vector<torch::IValue> &states) const {
  int32_t batch_size = states.size();

  // mod_states.size() == num_elements == 7 * num_encoders
  // mod_states[i].size() == batch_size
  std::vector<std::vector<torch::Tensor>> mod_states;
  int32_t num_elements = 0;

  for (auto &s : states) {
    torch::List<torch::Tensor> s_list =
        c10::impl::toTypedList<torch::Tensor>(s.toList());

    num_elements = s_list.size();
    if (mod_states.empty()) {
      mod_states.resize(num_elements);
    }

    for (int32_t i = 0; i != num_elements; ++i) {
      mod_states[i].push_back(s_list[i]);
    }
  }

  int32_t num_encoders = num_elements / 7;
  std::vector<torch::Tensor> stacked_states(num_elements);

  for (int32_t i = 0; i != num_encoders; ++i) {
    // cached_len: (num_layers, batch_size)
    stacked_states[i] = torch::cat(mod_states[i], /*dim*/ 1);
  }

  for (int32_t i = num_encoders; i != 2 * num_encoders; ++i) {
    // cached_avg: (num_layers, batch_size, D)
    stacked_states[i] = torch::cat(mod_states[i], /*dim*/ 1);
  }

  for (int32_t i = 2 * num_encoders; i != 3 * num_encoders; ++i) {
    // cached_key: (num_layers, left_context_size, batch_size, D)
    stacked_states[i] = torch::cat(mod_states[i], /*dim*/ 2);
  }

  for (int32_t i = 3 * num_encoders; i != 4 * num_encoders; ++i) {
    // cached_val: (num_layers, left_context_size, batch_size, D)
    stacked_states[i] = torch::cat(mod_states[i], /*dim*/ 2);
  }

  for (int32_t i = 4 * num_encoders; i != 5 * num_encoders; ++i) {
    // cached_val2: (num_layers, left_context_size, batch_size, D)
    stacked_states[i] = torch::cat(mod_states[i], /*dim*/ 2);
  }

  for (int32_t i = 5 * num_encoders; i != 6 * num_encoders; ++i) {
    // cached_conv1: (num_layers, batch_size, D, kernel-1)
    stacked_states[i] = torch::cat(mod_states[i], /*dim*/ 1);
  }

  for (int32_t i = 6 * num_encoders; i != 7 * num_encoders; ++i) {
    // cached_conv2: (num_layers, batch_size, D, kernel-1)
    stacked_states[i] = torch::cat(mod_states[i], /*dim*/ 1);
  }

  return stacked_states;
}

std::vector<torch::IValue> RnntZipformerModel::UnStackStates(
    torch::IValue ivalue) const {
  // ivalue is a list
  auto list_ptr = ivalue.toList();
  int32_t num_elements = list_ptr.size();

  // states.size() == num_elements = 7 * num_encoders
  std::vector<torch::Tensor> states;
  states.reserve(num_elements);
  for (int32_t i = 0; i != num_elements; ++i) {
    states.emplace_back(list_ptr.get(i).toTensor());
  }

  int32_t num_encoders = num_elements / 7;
  int32_t batch_size = states[0].size(1);

  // unstacked_states.size() == batch_size
  // unstacked_states[n].size() == num_elements
  std::vector<std::vector<torch::Tensor>> unstacked_states(batch_size);

  for (int32_t i = 0; i != num_encoders; ++i) {
    // cached_len: (num_layers, batch_size)
    std::vector<torch::Tensor> cached_len =
        torch::chunk(states[i], /*chunks*/ batch_size, /*dim*/ 1);
    for (int32_t n = 0; n != batch_size; ++n) {
      unstacked_states[n].push_back(cached_len[n]);
    }
  }

  for (int32_t i = num_encoders; i != 2 * num_encoders; ++i) {
    // cached_avg: (num_layers, batch_size, D)
    std::vector<torch::Tensor> cached_avg =
        torch::chunk(states[i], /*chunks*/ batch_size, /*dim*/ 1);
    for (int32_t n = 0; n != batch_size; ++n) {
      unstacked_states[n].push_back(cached_avg[n]);
    }
  }

  for (int32_t i = 2 * num_encoders; i != 3 * num_encoders; ++i) {
    // cached_key: (num_layers, left_context_size, batch_size, D)
    std::vector<torch::Tensor> cached_key =
        torch::chunk(states[i], /*chunks*/ batch_size, /*dim*/ 2);
    for (int32_t n = 0; n != batch_size; ++n) {
      unstacked_states[n].push_back(cached_key[n]);
    }
  }

  for (int32_t i = 3 * num_encoders; i != 4 * num_encoders; ++i) {
    // cached_val: (num_layers, left_context_size, batch_size, D)
    std::vector<torch::Tensor> cached_val =
        torch::chunk(states[i], /*chunks*/ batch_size, /*dim*/ 2);
    for (int32_t n = 0; n != batch_size; ++n) {
      unstacked_states[n].push_back(cached_val[n]);
    }
  }

  for (int32_t i = 4 * num_encoders; i != 5 * num_encoders; ++i) {
    // cached_val2: (num_layers, left_context_size, batch_size, D)
    std::vector<torch::Tensor> cached_val2 =
        torch::chunk(states[i], /*chunks*/ batch_size, /*dim*/ 2);
    for (int32_t n = 0; n != batch_size; ++n) {
      unstacked_states[n].push_back(cached_val2[n]);
    }
  }

  for (int32_t i = 5 * num_encoders; i != 6 * num_encoders; ++i) {
    // cached_conv1: (num_layers, batch_size, D, kernel-1)
    std::vector<torch::Tensor> cached_conv1 =
        torch::chunk(states[i], /*chunks*/ batch_size, /*dim*/ 1);
    for (int32_t n = 0; n != batch_size; ++n) {
      unstacked_states[n].push_back(cached_conv1[n]);
    }
  }

  for (int32_t i = 6 * num_encoders; i != 7 * num_encoders; ++i) {
    // cached_conv2: (num_layers, batch_size, D, kernel-1)
    std::vector<torch::Tensor> cached_conv2 =
        torch::chunk(states[i], /*chunks*/ batch_size, /*dim*/ 1);
    for (int32_t n = 0; n != batch_size; ++n) {
      unstacked_states[n].push_back(cached_conv2[n]);
    }
  }

  std::vector<torch::IValue> ans(batch_size);
  for (int32_t n = 0; n != batch_size; ++n) {
    // unstacked_states[n] is std::vector<torch::Tensor>
    ans[n] = unstacked_states[n];
  }

  return ans;
}

torch::IValue RnntZipformerModel::StateToIValue(const State &states) const {
  torch::List<torch::List<torch::Tensor>> ans;
  ans.reserve(states.size());
  for (const auto &s : states) {
    ans.push_back(torch::List<torch::Tensor>{s});
  }
  return ans;
}

RnntZipformerModel::State RnntZipformerModel::StateFromIValue(
    torch::IValue ivalue) const {
  torch::List<torch::IValue> list = ivalue.toList();

  int32_t num_layers = list.size();
  State ans;
  ans.reserve(num_layers);
  for (int32_t i = 0; i != num_layers; ++i) {
    ans.push_back(
        c10::impl::toTypedList<list.get(i).toTensor());
  }

  return ans
}

torch::IValue RnntZipformerModel::GetEncoderInitStates(
    int32_t batch_size /*=1*/) {
  torch::NoGradGuard no_grad;
  return encoder_.run_method("get_init_state", device_);
}

std::tuple<torch::Tensor, torch::Tensor, torch::IValue>
RnntZipformerModel::StreamingForwardEncoder(
    const torch::Tensor &features, const torch::Tensor &features_length,
    torch::IValue states) {
  torch::NoGradGuard no_grad;
  audo stack_states = StackStates(states);
  auto outputs =
      encoder_
          .run_method("streaming_forward", features, features_length, states)
          .toTuple();
  auto encoder_out = outputs->elements()[0].toTensor();
  auto encoder_out_length = outputs->elements()[1].toTensor();

  auto next_states = UnStackStates(outputs->elements()[2]);

  return std::make_tuple(encoder_out, encoder_out_length, next_states);
}

torch::Tensor RnntZipformerModel::ForwardDecoder(
    const torch::Tensor &decoder_input) {
  torch::NoGradGuard no_grad;
  return decoder_.run_method("forward", decoder_input, /*need_pad*/ false)
      .toTensor();
}

torch::Tensor RnntZipformerModel::ForwardJoiner(
    const torch::Tensor &projected_encoder_out,
    const torch::Tensor &projected_decoder_out) {
  torch::NoGradGuard no_grad;
  return joiner_
      .run_method("forward", projected_encoder_out, projected_decoder_out,
                  /*project_input*/ false)
      .toTensor();
}

torch::Tensor RnntZipformerModel::ForwardEncoderProj(
    const torch::Tensor &encoder_out) {
  torch::NoGradGuard no_grad;
  return encoder_proj_.run_method("forward", encoder_out).toTensor();
}

torch::Tensor RnntZipformerModel::ForwardDecoderProj(
    const torch::Tensor &decoder_out) {
  torch::NoGradGuard no_grad;
  return decoder_proj_.run_method("forward", decoder_out).toTensor();
}

}  // namespace sherpa
