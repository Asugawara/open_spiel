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
#include <algorithm>

#include "open_spiel/abseil-cpp/absl/random/random.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"


namespace open_spiel {
namespace algorithms {
namespace torch_dqn {

constexpr const int kIllegalActionLogitsPenalty = -1e9;

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
      q_network_(input_size_, hidden_layers_sizes_, num_actions_),
      target_q_network_(input_size_, hidden_layers_sizes_, num_actions_),
      optimizer_(q_network_->parameters(), torch::optim::AdamOptions(learning_rate_)),
      loss_str_(loss_str),
      exists_prev_(false),
      prev_state_(nullptr),
      step_counter_(0) {
        std::cout<<"construct"<<input_size_<<std::endl;
};

Action DQN::Step(const std::unique_ptr<State>& state, bool is_evaluation, bool add_transition_record) {
  Action action;
  std::cout << "start step" << std::endl;
  if (state->IsChanceNode()) {
    for (const auto& action_prob : state->ChanceOutcomes()) {
      state->ApplyAction(action_prob.first);
    }
  }
  if (!state->IsTerminal() && state->CurrentPlayer() == player_id_) {
    std::vector<float> info_state = state->InformationStateTensor(player_id_);
    std::vector<Action> legal_actions = state->LegalActions(player_id_);
    double epsilon = this->GetEpsilon(is_evaluation);
    action = this->EpsilonGreedy(info_state, legal_actions, epsilon);
    std::vector<double> probs;
  } else {
    action = 0;
    std::vector<double> probs;
  }

  if (!is_evaluation) {
    step_counter_++;

    if (step_counter_ % learn_every_ == 0) {
      this->Learn();
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

Action DQN::EpsilonGreedy(std::vector<float> info_state, std::vector<Action> legal_actions, double epsilon) {
  std::cout<<"start epsilon greedy" << std::endl;
  Action action;
  ActionsAndProbs actions_probs;
  if (absl::Uniform(rng_, 0.0, 1.0) < epsilon) {
    std::cout<<"calcurate probs" << std::endl;
    std::vector<double> probs(legal_actions.size(), 1.0/legal_actions.size());
    for (int i=0;i<legal_actions.size();i++){
      actions_probs.push_back({legal_actions[i], probs[i]});
    };
    std::cout<<"end push back" << std::endl;
    action = SampleAction(actions_probs, rng_).first;
    std::cout<<"action" << SampleAction(actions_probs, rng_).first << std::endl;
  } else {
    torch::Tensor info_state_tensor = torch::from_blob(info_state.data(), info_state.size()).view({1, -1});
    torch::Tensor q_value = q_network_->forward(info_state_tensor);
    action = q_value.argmax(1).item().toInt();
  };
  std::cout<<"end epsilon greedy" << std::endl;
  return action;
} ;

double DQN::GetEpsilon(bool is_evaluation, int power) {
  if (is_evaluation) {
    return 0.0;
  };

  double decay_steps = std::min((double)step_counter_, epsilon_decay_duration_);
  double decayed_epsilon = (
    epsilon_end_ + (epsilon_start_ - epsilon_end_) *
    std::pow((1 - decay_steps / epsilon_decay_duration_), power)
  );
  return decayed_epsilon;
};

void DQN::Learn() {
  std::cout << "replay buffer size:" << replay_buffer_.Size() << std::endl;
  if (replay_buffer_.Size() < batch_size_ || replay_buffer_.Size() < min_buffer_size_to_learn_) return;
  std::cout << "start learn" << std::endl;
  std::vector<Transition> transition = replay_buffer_.Sample(&rng_, batch_size_);
  std::vector<std::vector<float>> info_states;
  std::vector<long long> actions;
  std::vector<float> rewards;
  std::vector<std::vector<float>> next_info_states;
  std::vector<int> are_final_steps;
  std::vector<std::vector<int>> legal_actions_mask;
  for (auto t: transition) {
    info_states.push_back(t.info_state);
    actions.push_back(t.action);
    rewards.push_back(t.reward);
    next_info_states.push_back(t.next_info_state);
    are_final_steps.push_back(t.is_final_step);
    legal_actions_mask.push_back(t.legal_actions_mask);
  }
  std::cout << "to vector done" << std::endl;
  torch::Tensor info_states_tensor = torch::from_blob(info_states.data(), {batch_size_, info_states[0].size()});
  torch::Tensor next_info_states_tensor = torch::from_blob(next_info_states.data(), {batch_size_, next_info_states[0].size()});
  torch::Tensor q_values = q_network_->forward(info_states_tensor);
  torch::Tensor target_q_values = target_q_network_->forward(next_info_states_tensor);
  torch::Tensor illegal_actions = torch::sub(
      torch::from_blob(legal_actions_mask.data(), {batch_size_, legal_actions_mask[0].size()}), 1);
  torch::Tensor illegal_logits = torch::mul(illegal_actions, -1 * kIllegalActionLogitsPenalty);
  std::cout << "before add" << std::endl;
  torch::Tensor max_next_q = torch::max(
      torch::add(target_q_values, illegal_logits));
  torch::Tensor are_final_steps_tensor = torch::from_blob(are_final_steps.data(), {batch_size_, 1});
  torch::Tensor rewards_tensor = torch::from_blob(rewards.data(), {batch_size_, 1});
  torch::Tensor target = torch::sub(
      rewards_tensor, torch::mul(torch::sub(are_final_steps_tensor, 1), torch::mul(max_next_q, discount_factor_)));
  torch::Tensor actions_tensor = torch::from_blob(actions.data(), {batch_size_}, torch::TensorOptions().dtype(torch::kInt64));

  torch::Tensor predictions = q_values.index({torch::arange(q_values.size(0)), actions_tensor}).unsqueeze(1);
  torch::Tensor value_loss;
  if (loss_str_ == "mse") {
    torch::nn::MSELoss loss;
    value_loss = loss(predictions, target);
  } else if(loss_str_ == "huber") {
    torch::nn::SmoothL1Loss loss;
    value_loss = loss(predictions, target);
  } else {
    SpielFatalError("Not implemented, choose from 'mse', 'huber'.");
  };
  optimizer_.zero_grad();
  value_loss.backward();
  optimizer_.step();

};

}  // namespace torch_dqn
}  // namespace algorithms
}  // namespace open_spiel