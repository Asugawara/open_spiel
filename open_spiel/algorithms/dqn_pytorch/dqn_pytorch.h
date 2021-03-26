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
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"

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
    DQN(const Game& game);
    virtual ~DQN() = default;
  private:
    void step();
    void add_transition();
    void epsilon_greedy();
    void get_epsilon();
    void learn();
};

}  // namespace torch_dqn
}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_DQN_PYTORCH_H_