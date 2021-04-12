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

SonnetLinearImpl::SonnetLinearImpl(const int& input_size, const int& output_size, bool activate_relu=false)
   : sonnet_linear_(torch::nn::LinearOptions(
     /*in_features*/input_size,
     /*out_features*/output_size)),
     activate_relu_(activate_relu) {
  double stddev = 1.0 / std::sqrt(input_size);
  double mean = 0;
  double lower = (-2 * stddev - mean) / stddev;
  double upper = (2 * stddev - mean) / stddev;
  for (auto& named_parameter : sonnet_linear_->named_parameters()) {
    if (named_parameter.key().find("weight") != std::string::npos) {
      named_parameter.value().data() = torch::nn::functional::normalize(named_parameter.value().data());
    };
    if (named_parameter.key().find("bias") != std::string::npos) {
      named_parameter.value().data() = torch::zeros({output_size});
    };
    // std::cout << named_parameter.value() << std::endl;
  };
  register_module("sonnet_linear", sonnet_linear_);
};

torch::Tensor SonnetLinearImpl::forward(torch::Tensor x) {
  // std::cout << x << std::endl;
  if (activate_relu_) {
    return torch::relu(sonnet_linear_(x));
  } else {
    return sonnet_linear_(x);
  };
};

MLPImpl::MLPImpl(const int& input_size,
                 std::vector<int> hidden_layers_sizes,
                 int output_size,
                 bool activate_final,
                 std::string loss_str) 
    : input_size_(input_size),
      hidden_layers_sizes_(hidden_layers_sizes),
      output_size_(output_size_),
      activate_final_(activate_final),
      loss_str_(loss_str) {
  int layer_size = input_size_;
  for (auto h_size: hidden_layers_sizes_) {
    std::cout << layer_size << std::endl;
    layers_->push_back(SonnetLinear(/*input_size*/layer_size,
                                    /*output_size*/h_size));
    layer_size = h_size;
    std::cout << layer_size << std::endl;
  };
  layers_->push_back(SonnetLinear(/*input_size*/input_size_,
                                  /*output_size*/output_size,
                                  /*activate_final*/activate_final));
  register_module("layers", layers_);
};

torch::Tensor MLPImpl::forward(torch::Tensor x) {
  std::cout << "start MLP forward" << std::endl;
  for (int i;i<hidden_layers_sizes_.size() + 1;i++) {
    x = layers_[i]->as<SonnetLinear>()->forward(x);
  }
  std::cout << "end MLP forward" << std::endl;
  return x;
};


}  // namespace torch_dqn
}  // namespace algorithms
}  // namespace open_spiel