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
#include <cmath>

namespace open_spiel {
namespace algorithms {
namespace torch_dqn {

static constexpr double kSqrt2 = 1.4142135623730950488;

SonnetLinearImpl::SonnetLinearImpl(const int& input_size, const int& output_size, bool activate_relu=false)
   : sonnet_linear_(torch::nn::LinearOptions(/*in_features*/input_size,
                                             /*out_features*/output_size)),
     activate_relu_(activate_relu) {
  double stddev = 1.0 / std::sqrt(input_size);
  double lower = -2.0 * stddev;
  double upper = 2.0 * stddev;

  std::cout << "lower" << lower << "upper" << upper << std::endl;
  for (auto& named_parameter : sonnet_linear_->named_parameters()) {
    if (named_parameter.key().find("weight") != std::string::npos) {
      torch::Tensor uniform_param = torch::nn::init::uniform_(named_parameter.value()).to(torch::kFloat64);
      double clip_lower = 0.5 * (1.0 + std::erf(lower / kSqrt2));
      double clip_upper = 0.5 * (1.0 + std::erf(upper / kSqrt2));
      torch::Tensor new_param = kSqrt2 * torch::erfinv(2.0 * ((clip_upper - clip_lower) * uniform_param + clip_lower) - 1.0);
      named_parameter.value().data() = new_param;
    };
    if (named_parameter.key().find("bias") != std::string::npos) {
      named_parameter.value().data() = torch::zeros({output_size});
    };
  };
  
  register_module("sonnet_linear_", sonnet_linear_);
};

torch::Tensor SonnetLinearImpl::forward(torch::Tensor x) {
  if (activate_relu_) {
    return torch::relu(sonnet_linear_->forward(x));
  } else {
    return sonnet_linear_->forward(x);;
  };
};

MLPImpl::MLPImpl(const int& input_size,
                 std::vector<int> hidden_layers_sizes,
                 const int& output_size,
                 bool activate_final) 
    : input_size_(input_size),
      hidden_layers_sizes_(hidden_layers_sizes),
      output_size_(output_size),
      activate_final_(activate_final){
  int layer_size = input_size_;
  for (auto h_size: hidden_layers_sizes_) {
    layers_->push_back(SonnetLinear(/*input_size*/layer_size,
                                    /*output_size*/h_size));
    layer_size = h_size;
  };
  layers_->push_back(SonnetLinear(/*input_size*/layer_size,
                                  /*output_size*/output_size,
                                  /*activate_final*/activate_final));
  register_module("layers_", layers_);
};

torch::Tensor MLPImpl::forward(torch::Tensor x) {
  for (int i=0;i<hidden_layers_sizes_.size() + 1;i++) {
    x = layers_[i]->as<SonnetLinear>()->forward(x);
  };
  return x;
};


}  // namespace torch_dqn
}  // namespace algorithms
}  // namespace open_spiel