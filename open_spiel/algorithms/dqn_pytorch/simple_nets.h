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

#ifndef OPEN_SPIEL_ALGORITHMS_SIMPLE_NETS_H_
#define OPEN_SPIEL_ALGORITHMS_SIMPLE_NETS_H_

#include <torch/torch.h>

#include <iostream>
#include <vector>

#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace algorithms {
namespace torch_dqn {

struct MLPConfig {
  int input_size;
  std::vector<int> hidden_size;
  int output_size;
  bool activate_final;
};

std::istream& operator>>(std::istream& stream, MLPConfig& config);
std::ostream& operator<<(std::ostream& stream, const MLPConfig& config);

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
    MLPImpl(const MLPConfig& config);
    torch::Tensor forward(torch::Tensor x);

  private:
    torch::Tensor forward_(torch::Tensor x);
    torch::nn::ModuleList layers_;

};
TORCH_MODULE(MLP);

}  // namespace torch_dqn
}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_SIMPLE_NETS_H_