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

#include "open_spiel/algorithms/dqn_torch/simple_nets.h"

#include <torch/torch.h>

#include <iostream>

namespace open_spiel {
namespace algorithms {
namespace torch_dqn {

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

MLPImpl::MLPImpl(int input_size,
                 std::vector<int> hidden_layers_sizes,
                 int output_size,
                 bool activate_final,
                 std::string loss_str) 
    : input_size_(input_size),
      hidden_layers_sizes_(hidden_layers_sizes),
      output_size_(output_size_),
      activate_final_(activate_final),
      loss_str_(loss_str) {
  for (auto h_size: hidden_layers_sizes_) {
    SonnetLinear sonnet_linear(/*input_size*/input_size_,
                               /*output_size*/h_size);
    layers_->push_back(sonnet_linear);
    input_size_ = h_size;
  };
  SonnetLinear sonnet_linear(/*input_size*/input_size_,
                            /*output_size*/output_size,
                            /*activate_final*/activate_final);
  layers_->push_back(sonnet_linear);
  register_module("layers", layers_);
};

torch::Tensor MLPImpl::forward(torch::Tensor x) {
  return this->forward_(x);
};

torch::Tensor MLPImpl::losses(torch::Tensor input, torch::Tensor target) {
  if (loss_str_ == "mse") {
    torch::nn::MSELoss loss;
    torch::Tensor value_loss = loss(input, target);
  } else if(loss_str_ == "huber") {
    torch::nn::SmoothL1Loss loss;
    torch::Tensor value_loss = loss(input, target);
  } else {
    SpielFatalError("Not implemented, choose from 'mse', 'huber'.");
  };

};

}  // namespace torch_dqn
}  // namespace algorithms
}  // namespace open_spiel