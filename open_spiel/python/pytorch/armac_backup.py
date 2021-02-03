# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from open_spiel.python import policy

Trajectory = collections.namedtuple(
    "Trajectory",
    "i j util")

class B(nn.Module):
    def __init__(self):
        super(B, self).__init__()
        self.l1 = nn.Linear(input, 128)
        self.activation1 = nn.ReLu()
        self.l2 = nn.Linear(128, 128)
        self.activation2 = nn.ReLu()
        
    def forward(self, x):
        h1 = self.l1(x)
        h2 = self.activation1(h1)
        h3 = h1 + self.l2(h2)
        
        return self.activation2(h3)    

class GlobalCritic(nn.Module):
    def __init__(self):
        super(GlobalCritic, self).__init__()
        self.l1 = nn.Linear(input, 128)
        self.l2 = nn.Linear(input, 128)
        self.B = B(256)

    def forward(self, s0, s1):
        a0 = self.l1(s0) + self.l2(s1)
        a1 = self.l1(s1) + self.l2(s0)
        h1 = torch.cat(a0, a1)
        return self.B(h1)
        
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.l1 = nn.Linear(input, 128)
        self.l2 = nn.Linear(128, 512)
        self.lstm = nn.LSTM(512, 256)
        self.B = B(256)
        
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        _, hidden, _ = self.lstm(x)
        return self.B(hidden)
        
class MLP(nn.Module):
  """A simple network built from nn.linear layers."""

  def __init__(self,
               input_size,
               hidden_sizes,
               output_size,
               activate_final=False):
    """Create the MLP.
    Args:
      input_size: (int) number of inputs
      hidden_sizes: (list) sizes (number of units) of each hidden layer
      output_size: (int) number of outputs
      activate_final: (bool) should final layer should include a ReLU
    """

    super(MLP, self).__init__()
    self._layers = []
    # Hidden layers
    for size in hidden_sizes:
      self._layers.append(nn.Linear(in_features=input_size, out_features=size))
      self._layers.append(nn.ReLU())
      input_size = size
    # Output layer
    self._layers.append(
        nn.Linear(in_features=input_size, out_features=output_size))
    if activate_final:
      self._layers.append(nn.ReLU())

    self.model = nn.ModuleList(self._layers)

  def forward(self, x):
    for layer in self.model:
      x = layer(x)
    return x
        
        
class ArmacSolver(policy.Policy):
    def __init__(self, game):
        all_players = list(range(game.num_players()))
        super(ArmacSolver, self).__init__(game, all_players)
        self._game = game
        if game.get_type().dynamics == pyspiel.GameType.Dynamics.SIMULTANEOUS:
          # `_traverse_game_tree` does not take into account this option.
          raise ValueError("Simulatenous games are not supported.")
        self._session = session
        self._batch_size_advantage = batch_size_advantage
        self._batch_size_strategy = batch_size_strategy
        self._policy_network_train_steps = policy_network_train_steps
        self._advantage_network_train_steps = advantage_network_train_steps
        self._num_players = game.num_players()
        self._root_node = self._game.new_initial_state()
        self._embedding_size = len(self._root_node.information_state_tensor(0))
        self._num_iterations = num_iterations
        self._num_trajectories = num_trajectories
        self._reinitialize_advantage_networks = reinitialize_advantage_networks
        self._num_actions = game.num_distinct_actions()
        
        self._critic_network = GlobalCritic()
        self._optimizer_critic = torch.optim.Adam(
            self._critic_network.parameters(), lr=critic_learning_rate)
        
        self._policy_network = MLP(self._embedding_size,
                                   list(policy_network_layers),
                                   self._num_actions)
        self._optimizer_policy = torch.optim.Adam(self._policy_network.parameters(),
                                                  lr=learning_rate)
        
        self._advantage_networks =[
            MLP(self._embedding_size, list(advantage_network_layers),
                self._num_actions) for _ in range(self._num_players)]
        self._optimizer_advantages = [
            torch.optim.Adam(
                self._advantage_networks[i].parameters(), lr=learning_rate)
            for i in range(self._num_players)]

        self.memory = []
        self.D = ReservoirBuffer(1024)
        
        
    def iteration(self, epoch):
        self.D.clean()
        state = self._game.new_initial_state()
        i = state.current_player()
        self.policy = NomarizedReLU(self._advantage_networks[i](info_state))
        self.value = torch.sum(
            self.W(h, a) * self._critic(h, a))
        behavior_policy = mu
        self.memory[epoch] = {'policy': self.policy, 'value': self.value}
        
        for update_player in range(self._num_players):
            
            self._episode(epoch, state, update_player)
            self._learn()
    
    def _episode(self, epoch, state, update_player):
        cur_player = state.current_player()
        if cur_player == update_player:
            j = random.choice([i for i in range(epoch)], 1)
            self.trajectory = []
            self.trajectory, self.d = self._sample_trajectory(state, 
                                                      update_player, 
                                                      update_player_policy, 
                                                      policy)
            for hist in self.trajectory:
                if hist.current_player() = cur_player:
                    state = hist
                    regret = (self.memory[j]['policy'](h, a_change) 
                              - self.memory[j]['value'](h,a))
                    a = trajectory.action
                    self.d += [hist, state, a, regret, NomarizedReLU(self.W(s))]
            self.D.append(d)
                    
    def _add_regret(self, info_state_key, action_idx, amount):
        self._infostates[info_state_key][_REGRET_INDEX][action_idx] += amount
                    
    def _sample_trajectory(self, state, update_player, update_player_policy, policy):
        trajectory.append(state)
        if state.is_terminal():
            self.d = [i, j , state.player_return(update_player)]
            return trajectory, self.d

        if state.is_chance_node():
            outcomes, _ = zip(*state.chance_outcomes())
            outcome = np.random.choice(outcomes, p=self.policy)
            state.apply_action(outcome)
            return self._sample_trajectory(state, update_player)

        cur_player = state.current_player()
        info_state_key = state.information_state_string(cur_player)
        legal_actions = state.legal_actions()
        num_legal_actions = len(legal_actions)

        #policy = self._regret_matching(infostate_info[_REGRET_INDEX],
        #                               num_legal_actions)
        if cur_player == update_player:
            action = update_player_policy
        else:
            action = np.random.choice(legal_actions, p=policy)
            
        state.apply_action(action)
        return self._sample_trajectory(state, update_player)
          

    def _learn(self, cur_player):
        episode_batch = self.D.sample(32)

        for h, s in episode_batch:
            self._train_tree_backup_algo(_lambda)
            if h.player == cur_player:
                loss_advantage = MSE(advantage, target)
            else:
                loss_policy = MSE
                
    def _train_tree_backup_algo(self, _lambda):
        state = s0
        if self.elig_trace:
            self.elig_trace = self.elig_trace*discount*_lambda*policy
        else:
            self.elig_trace = 1
        
        loss = (regret 
                    + discount
                    * torch.sum(policy(s_t1, a) * self._critic(s_t1, a)) 
                    - self._critic(s_t, a_t))
        
        self.optimizer.zero_grad()
        loss.backword()
        self.optimiser.step()
        
    def _learn_advantage_network():
        
    def _learn_policy_network():
        
        
  


        
class ReservoirBuffer(object):
    def __init__(self, reservoir_buffer_capacity):
        self._reservoir_buffer_capacity = reservoir_buffer_capacity
        self._data = []
        self._add_calls = 0
    
    def add(self, element):
        if len(self._data) < self._reservoir_buffer_capacity:
            self._data.append(element)
        else:
            idx = np.random.randint(0, self._add_calls + 1)
        if idx < self._reservoir_buffer_capacity:
            self._data[idx] = element
        self._add_calls += 1
    
    def sample(self, num_samples):
        if len(self._data) < num_samples:
            raise ValueError("{} elements could not be sampled from size {}".format(
        num_samples, len(self._data)))
        return random.sample(self._data, num_samples)
    
    def clear(self):
        self._data = []
        self._add_calls = 0
    
    def __len__(self):
        return len(self._data)
    
    def __iter__(self):
        return iter(self._data)
    
    @property
    def data(self):
      return self._data

    def shuffle_data(self):
      random.shuffle(self._data)