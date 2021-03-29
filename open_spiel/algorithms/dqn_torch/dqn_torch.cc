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

namespace open_spiel {
namespace algorithms {
namespace torch_dqn {

// CircularBuffer<hogehoge> replay_buffer(replay_buffer_size);

DQN::DQN(std::shared_ptr<const Game> game, Player player_id, MLPConfig mlp_config)
    : game_(game),
      player_id_(player_id),
      q_network_(MLP(mlp_config)),
      target_q_network_(MLP(mlp_config)),
      optimizer_(q_network_->parameters(), 
                 torch::optim::AdamOptions(mlp_config.learning_rate)),
      step_counter_(0) {
  std::cout << q_network_ << std::endl;
};

Action DQN::Step(std::unique_ptr<State> state, bool is_evaluate) {
  // if (state->IsTerminal()) {
  //   return;
  // }
  if (state->IsChanceNode()) {
    for (const auto& action_prob : state->ChanceOutcomes()) {
      state->ApplyAction(action_prob.first);
    }
  }
  else if (state->CurrentPlayer() == player_id_) {
    std::vector<float> info_state = state->InformationStateTensor(player_id_);
    std::vector<Action> legal_actions = state->LegalActions(player_id_);
    // double epsilon = this->GetEpsilon(is_evaluation);
    // ActionsAndProbs action_prob = this->EpsilonGreedy(info_state, legal_actions, epsilon);
  }
  Action action;
  if (!is_evaluate) {
    step_counter_++;

    if (step_counter_ % learn_every_ == 0) {
      void last_loss_value = Learn()
    };
    if (step_counter_ % update_target_network_every_ == 0) {
      torch::save(q_network_, 'q_network.pt');
      torch::load(target_q_network_, 'q_network.pt');
    };
    if (prev_timestep && add_transition_record) {
      AddTransition();
    };
    if 

  }

  return action;
};
  

}  // namespace torch_dqn
}  // namespace algorithms
}  // namespace open_spiel