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
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {
namespace torch_dqn {
namespace {

void TestModelCreation() {
  std::cout << "\n~-~-~-~- TestModelCreation -~-~-~-~" << std::endl;

  std::shared_ptr<const Game> game = LoadGame("clobber");

  MLPImpl mlp(/*input_size=*/game->ObservationTensorSize(),
              /*hidden_size=*/{128,128},
              /*output_size=*/game->NumDistinctActions(),
              /*activate_final*/true);
};

}
}  // namespace torch_dqn
}  // namespace algorithms
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::algorithms::torch_dqn::TestModelCreation();
  return 0;
}