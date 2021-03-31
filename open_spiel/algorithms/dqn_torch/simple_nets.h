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

#ifndef OPEN_SPIEL_ALGORITHMS_DQN_TORCH_SIMPLE_NETS_H_
#define OPEN_SPIEL_ALGORITHMS_DQN_TORCH_SIMPLE_NETS_H_

#include <torch/torch.h>

#include <iostream>
#include <string>
#include <vector>


#include "open_spiel/spiel.h"

namespace open_spiel {
namespace algorithms {
namespace torch_dqn {


class SonnetLinearImpl : public torch::nn::Module {
  public :
    SonnetLinearImpl(int input_size, int output_size, bool activate_relu);
    torch::Tensor forward(torch::Tensor x);
  
  private:
    bool activate_relu_;
    torch::nn::Linear sonnet_linear;
};
TORCH_MODULE(SonnetLinear);

class MLPImpl : public torch::nn::Module {
  public:
    MLPImpl(int input_size,
            std::vector<int> hidden_size,
            int output_size,
            bool activate_final=false,
            std::string loss_str="mse");
    torch::Tensor forward(torch::Tensor x);
    torch::Tensor losses(torch::Tensor input, torch::Tensor target);

  private:
    int input_size_;
    std::vector<int> hidden_layers_sizes_;
    int output_size_;
    bool activate_final_;
    std::string loss_str_;
    double learning_rate_;
    torch::Tensor forward_(torch::Tensor x);
    torch::nn::ModuleList layers_;

};
TORCH_MODULE(MLP);

}  // namespace torch_dqn
}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_DQN_TORCH_SIMPLE_NETS_H_