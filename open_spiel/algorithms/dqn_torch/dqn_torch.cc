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

DQN::DQN(bool use_observation,
         Player player_id,
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
    : use_observation_(use_observation),
      player_id_(player_id),
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
};

std::vector<float> DQN::GetInfoState(const std::unique_ptr<State>& state, Player player_id, bool use_observation) {
  if (use_observation){
    return state->ObservationTensor(player_id);
  } else {
    return state->InformationStateTensor(player_id);
  };
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
    std::cout << state->IsTerminal() << ':' << state->CurrentPlayer() << std::endl;
    std::vector<float> info_state = GetInfoState(state, player_id_, use_observation_);
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
    /*info_state=*/GetInfoState(prev_state, player_id_, use_observation_),
    /*action=*/prev_action_,
    /*reward=*/state->PlayerReward(player_id_),
    /*next_info_state=*/GetInfoState(state, player_id_, use_observation_),
    /*is_final_step=*/state->IsTerminal(),
    /*legal_actions_mask=*/legal_actions_mask};
  replay_buffer_.Add(transition); 
};

Action DQN::EpsilonGreedy(std::vector<float> info_state, std::vector<Action> legal_actions, double epsilon) {
  std::cout<<"start epsilon greedy" << std::endl;
  Action action;
  ActionsAndProbs actions_probs;
  if (absl::Uniform(rng_, 0.0, 1.0) < epsilon) {
    std::vector<double> probs(legal_actions.size(), 1.0/legal_actions.size());
    for (int i=0;i<legal_actions.size();i++){
      actions_probs.push_back({legal_actions[i], probs[i]});
    };
    action = SampleAction(actions_probs, rng_).first;
    std::cout<<"sample action" << SampleAction(actions_probs, rng_).first << std::endl;
  } else {
    torch::Tensor info_state_tensor = torch::from_blob(info_state.data(), {info_state.size()}).view({1, -1});
    torch::Tensor q_value = q_network_->forward(info_state_tensor);
    for (auto e: legal_actions){
      std::cout << e << std::endl;
    };
    torch::Tensor legal_actions_tensor = torch::from_blob(legal_actions.data(), {legal_actions.size()}, torch::TensorOptions().dtype(torch::kInt64));
    std::cout << legal_actions_tensor << q_network_->forward(info_state_tensor) << q_value << std::endl;
    action = torch::mul(q_value.detach(), legal_actions_tensor).argmax(1).item().toInt();
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
  std::vector<torch::Tensor> info_states;
  std::vector<torch::Tensor> next_info_states;
  std::vector<torch::Tensor> legal_actions_mask;
  std::vector<Action> actions;
  std::vector<float> rewards;
  std::vector<int> are_final_steps;
  for (auto t: transition) {
    info_states.push_back(
        torch::from_blob(t.info_state.data(), {1, t.info_state.size()}, torch::TensorOptions().dtype(torch::kFloat32)).clone());
    next_info_states.push_back(
        torch::from_blob(t.next_info_state.data(), {1, t.next_info_state.size()}, torch::TensorOptions().dtype(torch::kFloat)).clone());
    legal_actions_mask.push_back(
        torch::from_blob(t.legal_actions_mask.data(), {1, t.legal_actions_mask.size()}, torch::TensorOptions().dtype(torch::kInt32)).to(torch::kInt64).clone());
    actions.push_back(t.action);
    rewards.push_back(t.reward);
    are_final_steps.push_back(t.is_final_step);    
  }
  std::cout << "to vector done" << std::endl;
  torch::Tensor info_states_tensor = torch::cat(torch::TensorList(info_states), 0);
  torch::Tensor next_info_states_tensor = torch::cat(torch::TensorList(next_info_states), 0);
  torch::Tensor q_values = q_network_->forward(info_states_tensor);
  torch::Tensor target_q_values = target_q_network_->forward(next_info_states_tensor);
  std::cout << info_states_tensor << std::endl;
  std::cout << torch::TensorList(legal_actions_mask) << std::endl;
  torch::Tensor legal_action_masks_tensor = torch::cat(torch::TensorList(legal_actions_mask));
  std::cout << legal_action_masks_tensor << std::endl;
  torch::Tensor illegal_actions = torch::sub(legal_action_masks_tensor, 1);
  std::cout << illegal_actions << std::endl;
  torch::Tensor illegal_logits = torch::mul(illegal_actions, -1 * kIllegalActionLogitsPenalty);
  std::cout << "before add" << std::endl;
  torch::Tensor max_next_q = std::get<0>(torch::max(
      torch::add(target_q_values, illegal_logits), 1));
  torch::Tensor are_final_steps_tensor = torch::from_blob(are_final_steps.data(), {batch_size_}, torch::TensorOptions().dtype(torch::kInt32));
  torch::Tensor rewards_tensor = torch::from_blob(rewards.data(), {batch_size_});
  torch::Tensor target = torch::sub(
      rewards_tensor, torch::mul(torch::sub(are_final_steps_tensor, 1), torch::mul(max_next_q, discount_factor_)));
  // std::cout << "sub:" << torch::sub(are_final_steps_tensor, 1) << std::endl;
  // std::cout << "mul:" << torch::mul(max_next_q, discount_factor_) << std::endl;
  torch::Tensor actions_tensor = torch::from_blob(actions.data(), {batch_size_}, torch::TensorOptions().dtype(torch::kInt64));
  torch::Tensor predictions = q_values.index({torch::arange(q_values.size(0)), actions_tensor});
  std::cout << q_values << predictions << target << std::endl;
  std::cout << "rewards:" << rewards_tensor << std::endl;
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