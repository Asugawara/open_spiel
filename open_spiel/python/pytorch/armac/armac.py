import random
import sys
import pyspiel

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nets import BaseModel, GlobalCritic, RNN, MLP_torso
from armac_utils import ARMACActor, ARMACLearner
from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability

class Buffer(object):
  """A fixed size buffer that keeps the newest values."""

  def __init__(self, max_size):
    self.max_size = max_size
    self.data = []
    self.total_seen = 0  # The number of items that have passed through.

  def __len__(self):
    return len(self.data)

  def __bool__(self):
    return bool(self.data)

  def append(self, val):
    return self.extend([val])

  def extend(self, batch):
    batch = list(batch)
    self.total_seen += len(batch)
    self.data.extend(batch)
    self.data[:-self.max_size] = []

  def sample(self, count):
    return random.sample(self.data, count)





def armac(game, epochs, episodes, learning_steps, eval_interval, use_rnn=False):
  GAME = pyspiel.load_game(game)
  num_actions = GAME.num_distinct_actions()
  num_players = GAME.num_players()
  observation_size = GAME.observation_tensor_size()
  HEAD_HIDDEN_SIZE = 256
  player_head = []
  for _ in range(num_players):
    if use_rnn:
      head = RNN(observation_size, HEAD_HIDDEN_SIZE)
    else:
      head = MLP_torso(observation_size, HEAD_HIDDEN_SIZE)
    player_head.append(head)
    
  global_critic_head = GlobalCritic(player_head, HEAD_HIDDEN_SIZE)
  out_critic = nn.Linear(HEAD_HIDDEN_SIZE, 1)
  global_critic_network = nn.Sequential(global_critic_head, out_critic)

  policy_network_list = []
  B = BaseModel(HEAD_HIDDEN_SIZE)
  for head in player_head:
    out_policy = nn.Linear(HEAD_HIDDEN_SIZE, num_actions)
    policy_network_list.append(nn.Sequential(head, B, out_policy))

  for t in range(epochs):
    # TODO implement adaptive policy
    D = Buffer(1024)
    PLAYER_ID = t%num_players
    if len(D):
      sampled_net_states = D.sample(1)
    else:
      sampled_net_states = [global_critic_network.state_dict(),
          [net.state_dict() for net in policy_network_list]]
    sampled_critic_network = copy.copy(global_critic_network)
    sampled_critic_network.load_state_dict(sampled_net_states[0])

    sampled_policy_network_list = []
    for i in range(num_players):
      sampled_policy_network = copy.copy(policy_network_list[0])
      sampled_policy_network.load_state_dict(sampled_net_states[1][i])
      sampled_policy_network_list.append(sampled_policy_network)

    actor = ARMACActor(GAME, 
                        policy_network_list[PLAYER_ID], 
                        sampled_policy_network_list, 
                        sampled_critic_network, 
                        episodes)
    actor.act()

    BATCH_SIZE = 32
    learner = ARMACLearner(actor.buffer, 
                            BATCH_SIZE, 
                            global_critic_network, 
                            policy_network_list[PLAYER_ID])
    learner.learn(learning_steps)

    if (t+1) % eval_interval == 0:
      print('eval start')
      def action_probabilities(state):
        """Returns action probabilities dict for a single batch."""
        cur_player = state.current_player()
        legal_actions = state.legal_actions(cur_player)
        # TODO confirm difference information_state_tensor or observation_tensor
        info_state = torch.Tensor(state.observation_tensor(cur_player))
        probs = policy_network_list[cur_player](info_state)
        return {action: probs.detach().numpy()[action] for action in legal_actions}
      average_policy = policy.tabular_policy_from_callable(
        GAME, action_probabilities)
      conv = exploitability.nash_conv(GAME, average_policy)
      print(f"Iter: {t+1}, NashConv: {conv}")

    #TODO use dict or namedtuple
    D.append([global_critic_network.state_dict(),
        [net.state_dict() for net in policy_network_list]])
