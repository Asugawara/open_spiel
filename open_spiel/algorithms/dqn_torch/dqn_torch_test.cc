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
  std::shared_ptr<const Game> game = efg_game::LoadEFGGame(efg_game::GetSimpleForkEFGData());
  SPIEL_CHECK_TRUE(game != nullptr);
  DQN dqn(/*use_observation*/game->GetType().provides_observation_tensor,
          /*player_id*/0,
          /*state_representation_size*/game->InformationStateTensorSize(),
          /*num_actions*/game->NumDistinctActions(),
          /*hidden_layers_sizes*/{16},
          /*replay_buffer_capacity*/100,
          /*batch_size*/5,
          /*learning_rate*/0.01,
          /*update_target_network_every*/20,
          /*learn_every*/10,
          /*discount_factor*/1.0,
          /*min_buffer_size_to_learn*/5,
          /*epsilon_start*/0.02,
          /*epsilon_end*/0.01);
  std::unique_ptr<State> state = game->NewInitialState();
  int total_reward = 0;
  for (int i=0;i<100;i++) {
    std::cout << "Episode: " << i << std::endl;
    std::cout << "total_reward: " << total_reward << std::endl;
    std::unique_ptr<open_spiel::State> state = game->NewInitialState();
    while (!state->IsTerminal()) {
      open_spiel::Action action = dqn.Step(state);
      std::cout << "action: " << action << std::endl;
      state->ApplyAction(action);
      total_reward += state->Rewards()[0];
    };
    dqn.Step(state);
  };

  SPIEL_CHECK_GE(total_reward, 75);

}
  
}  // namespace
}  // namespace torch_dqn
}  // namespace algorithms
}  // namespace open_spiel

int main(int args, char** argv) {
  open_spiel::algorithms::torch_dqn::TestSimpleGame();
  return 0;
}