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

#include "open_spiel/algorithms/dqn_torch/dqn_torch.h"

#include <torch/torch.h>

#include <iostream>
#include <string>
#include <vector>

#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/circular_buffer.h"
#include "open_spiel/games/efg_game.h"
#include "open_spiel/games/efg_game_data.h"

namespace open_spiel {
namespace algorithms {
namespace torch_dqn {
namespace {

void TestSimpleGame() {
  std::shared_ptr<const Game> game = efg_game::LoadEFGGame(efg_game::GetSampleEFGData());
  SPIEL_CHECK_TRUE(game != nullptr);
  MLPConfig mlp_config = {};
  DQN agent(game, 0, mlp_config);

  std::unique_ptr<State> state = game->NewInitialState();
  for (int i=0;i<100;i++) {
    Action action = agent.Step(game->NewInitialState());
  }

  SPIEL_CHECK_TRUE(game != nullptr);

}
  
}  // namespace
}  // namespace torch_dqn
}  // namespace algorithms
}  // namespace open_spiel

int main(int args, char** argv) {
  open_spiel::algorithms::torch_dqn::TestSimpleGame();
  return 0;
}