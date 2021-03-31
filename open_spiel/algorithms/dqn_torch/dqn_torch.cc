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
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace algorithms {
namespace torch_dqn {

// CircularBuffer<hogehoge> replay_buffer(replay_buffer_size);

DQN::DQN(Player player_id,
         int state_representation_size,
         int num_actions,
         std::vector<int> hidden_layers_sizes,
         int replay_buffer_capacity,
         int batch_size,
         double learning_rate,
         int update_target_network_every,
         int learn_every,
         double discount_factor,
         int min_buffer_size_to_learn,
         double epsilon_start,
         double epsilon_end,
         int epsilon_decay_duration,
         std::string optimizer_str,
         std::string loss_str)
    : player_id_(player_id),
      input_size_(state_representation_size),
      num_actions_(num_actions),
      hidden_layers_sizes_(hidden_layers_sizes),
      batch_size_(batch_size),
      update_target_network_every_(update_target_network_every),
      learn_every_(learn_every),
      min_buffer_size_to_learn_(min_buffer_size_to_learn),
      discount_factor_(discount_factor),
      epsilon_start_(epsilon_start),
      epsilon_end_(epsilon_end_),
      epsilon_decay_duration_(epsilon_decay_duration),
      replay_buffer_(replay_buffer_capacity),
      q_network_(input_size_, hidden_layers_sizes_, output_size_),
      target_q_network_(input_size_, hidden_layers_sizes_, output_size_),
      optimizer_(q_network_->parameters(), torch::optim::AdamOptions(learning_rate_)),
      exists_prev_(false),
      prev_state_(nullptr),
      step_counter_(0) {
};

Action DQN::Step(std::unique_ptr<State> state, bool is_evaluation, bool add_transition_record) {
  // if (state->IsTerminal()) {
  //   return;
  // }
  Action action;
  if (state->IsChanceNode()) {
    for (const auto& action_prob : state->ChanceOutcomes()) {
      state->ApplyAction(action_prob.first);
    }
  }
  if (!state->IsTerminal() && state->CurrentPlayer() == player_id_) {
    std::vector<float> info_state = state->InformationStateTensor(player_id_);
    std::vector<Action> legal_actions = state->LegalActions(player_id_);
    double epsilon = this->GetEpsilon(is_evaluation);
    ActionsAndProbs action_prob = this->EpsilonGreedy(info_state, legal_actions, epsilon);
    action = action_prob[0].first;
    std::vector<double> probs;
  } else {
    action = 0;
    std::vector<double> probs;
  }
  
  if (!is_evaluation) {
    step_counter_++;

    if (step_counter_ % learn_every_ == 0) {
      int last_loss_value = Learn();
    };
    if (step_counter_ % update_target_network_every_ == 0) {
      torch::save(q_network_, "q_network.pt");
      torch::load(target_q_network_, "q_network.pt");
    };
    if (exists_prev_ && add_transition_record) {
      AddTransition(prev_state_, prev_action_, state);
    };
    if (state->IsTerminal()) {
      exists_prev_=false;
      prev_action_=0;
      return;
    } else {
      exists_prev_ = true;
      prev_state_ = state->Clone();
      prev_action_ = action;
    };

  };

  return action;
};
  
void DQN::AddTransition(const std::unique_ptr<State>& prev_state, Action prev_action, const std::unique_ptr<State>& state) {
  if (prev_state == nullptr) {
    SpielFatalError("prev_state_ doesn't exist");
  }
  std::vector<int> legal_actions_mask = state->LegalActionsMask(player_id_);
  Transition transition = {
    /*info_state=*/prev_state->InformationStateTensor(player_id_),
    /*action=*/prev_action_,
    /*reward=*/state->PlayerReward(player_id_),
    /*next_info_state=*/state->InformationStateTensor(player_id_),
    /*is_final_step=*/state->IsTerminal(),
    /*legal_actions_mask=*/legal_actions_mask};
  replay_buffer_.Add(transition); 
};

}  // namespace torch_dqn
}  // namespace algorithms
}  // namespace open_spiel