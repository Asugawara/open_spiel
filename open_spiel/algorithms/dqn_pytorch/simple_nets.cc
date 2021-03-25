// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/algorithms/dqn_pytorch/simple_nets.h"

#include <torch/torch.h>

#include <iostream>

namespace open_spiel {
namespace algorithms {
namespace torch_dqn {

std::istream& operator>>(std::istream& stream, MLPConfig& config) {
  stream >> config.input_size >> config.hidden_size >> config.out_size;
  return stream;
};

std::ostream& operator<<(std::ostream& stream, const MLPConfig& config) {
  stream << config.input_size;
  return stream;
};

SonnetLinearImpl::SonnetLinearImpl(int input_size, int output_size, bool activate_relu=false)
   : sonnet_linear(torch::nn::Linear(
     /*input_size*/input_size,
     /*output_size*/output_size)) {
  activate_relu_ = activate_relu;
  register_module("sonnet_linear", sonnet_linear);
};

torch::Tensor SonnetLinearImpl::forward(torch::Tensor x) {
  if (activate_relu_) {
    return torch::relu(sonnet_linear(x));
  } else {
    return sonnet_linear(x);
  };
};

MLPImpl::MLPImpl(const MLPConfig& config) {
  int input_size = config.input_size;
  for (auto h_size: config.hidden_size) {
    SonnetLinear sonnet_linear(/*input_size*/input_size,
                               /*output_size*/h_size);
    layers_->push_back(sonnet_linear);
    input_size = h_size;
  };
  SonnetLinear sonnet_linear(/*input_size*/input_size,
                            /*output_size*/config.output_size,
                            /*activate_final*/config.activate_final);
  layers_->push_back(sonnet_linear);
  register_module("layers", layers_);
};

torch::Tensor MLPImpl::forward(torch::Tensor x) {
  return this->forward_(x);
}

}  // namespace torch_dqn
}  // namespace algorithms
}  // namespace open_spiel