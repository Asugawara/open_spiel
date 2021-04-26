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

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"


namespace open_spiel {
namespace algorithms {
namespace torch_dqn {
namespace {

void TestModelCreation() {
  std::shared_ptr<const Game> game = LoadGame("clobber");
  std::vector<int> layer = {128, 128};
  MLP mlp(/*input_size=*/game->ObservationTensorSize(),
          /*hidden_size=*/layer,
          /*output_size=*/game->NumDistinctActions());
  std::cout << mlp << std::endl;
}

void TestLossReduction() {
  std::vector<int> layer = {16};
  MLP mlp(/*input_size=*/4,
          /*hidden_size=*/layer,
          /*output_size=*/2,
          /*activate_final*/true);
  torch::Tensor x = torch::rand({4,4});
  torch::Tensor y = torch::rand({4});
  std::cout << x << y << std::endl;
  torch::optim::Adam optimizer(mlp->parameters(), torch::optim::AdamOptions(0.01));
  torch::nn::MSELoss mse_loss;
  for (int i=0;i<20;i++) {
    torch::Tensor output = mlp->forward(x);
    optimizer.zero_grad();
    torch::Tensor loss = mse_loss(std::get<0>(output.max(1)), y);
    std::cout << "loss:" << loss << std::endl;
    loss.backward();
    optimizer.step();
  };

};

}  // namespace
}  // namespace torch_dqn
}  // namespace algorithms
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::algorithms::torch_dqn::TestModelCreation();
  open_spiel::algorithms::torch_dqn::TestLossReduction();
  return 0;
}