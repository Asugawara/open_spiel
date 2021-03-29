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

#ifndef OPEN_SPIEL_ALGORITHMS_DQN_PYTORCH_H_
#define OPEN_SPIEL_ALGORITHMS_DQN_PYTORCH_H_

#include <torch/torch.h>

#include <memory>
#include <random>
#include <vector>

#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/strings/string_view.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/algorithms/dqn_torch/simple_nets.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {
namespace torch_dqn {

struct Transition {
  int info_state;
  int action;
  int reward;
  int next_info_state;
  bool is_final_step;
  int legal_actions_mask;
};


class DQN {
  public: 
    DQN(std::shared_ptr<const Game> game, Player player_id, MLPConfig mlp_config);
    virtual ~DQN() = default;
    Action Step(std::unique_ptr<State> state, bool is_evaluate);
  protected:
    std::shared_ptr<const Game> game_;
  private:
    int player_id_;
    int update_target_network_every_;
    int learn_every_;
    int replay_buffer_capacity_;
    int batch_size_;
    int step_counter_;
    MLP q_network_;
    MLP target_q_network_;
    torch::optim::Adam optimizer_;
    void AddTransition();
    ActionsAndProbs EpsilonGreedy();
    double GetEpsilon();
    void Learn();
};

}  // namespace torch_dqn
}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_DQN_PYTORCH_H_