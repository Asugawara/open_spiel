import random
import sys
import pyspiel

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nets import BaseModel, GlobalCritic, RNN
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





def armac(game, epochs, episodes, learning_steps, eval_interval):
  GAME = pyspiel.load_game(game)
  num_actions = GAME.num_distinct_actions()
  num_players = GAME.num_players()
  observation_size = GAME.observation_tensor_size()
  RNN_HIDDEN_SIZE = 256
  
  player_rnn = []
  for _ in range(num_players):
    rnn = RNN(observation_size, RNN_HIDDEN_SIZE)
    player_rnn.append(rnn)
  
  global_critic_head = GlobalCritic(player_rnn, RNN_HIDDEN_SIZE)
  out_critic = nn.Linear(RNN_HIDDEN_SIZE, 1)
  global_critic_network = nn.Sequential(global_critic_head, out_critic)

  policy_netowork_list = []
  for _ in range(num_players):
    out_policy = nn.Linear(RNN_HIDDEN_SIZE, num_actions)
    policy_netowork_list.append(nn.Sequential(global_critic_head, out_policy))

  for t in range(epochs):
    # TODO implement adaptive policy
    D = Buffer(1024)
    #info_state = 
    #recent_policy = [F.softmax(net(info_state)) for net in policy_netowork_list]
    PLAYER_ID = t%num_players
    for ep in range(episodes):
      if len(D):
        sampled_net_states = D.sample(1)
      else:
        sampled_net_states = [global_critic_network.state_dict(),
            [net.state_dict() for net in policy_netowork_list]]
      sampled_critic_network = copy.copy(global_critic_network)
      sampled_critic_network.load_state_dict(sampled_net_states[0])

      sampled_policy_network_list = []
      for i in range(num_players):
        sampled_policy_network = copy.copy(policy_netowork_list[0])
        sampled_policy_network.load_state_dict(sampled_net_states[1][i])
        sampled_policy_network_list.append(sampled_policy_network)
      print("="*89,f'{game}\n',policy_netowork_list[0].state_dict())
      print("="*89,f'{game}\n',sampled_policy_network_list[0].state_dict())

      actor = ARMACActor(GAME, 
                         policy_netowork_list[PLAYER_ID], 
                         sampled_policy_network_list, 
                         sampled_critic_network, 
                         10)
      actor.act(PLAYER_ID)

    for l_step in range(learning_steps):
      BATCH_SIZE = 4
      learner = ARMACLearner(actor.buffer, 
                             BATCH_SIZE, 
                             global_critic_network, 
                             policy_netowork_list[PLAYER_ID])
      learner.learn()
    
    if (t+1) % eval_interval == 0:
      print('eval start')
      def action_probabilities(state):
        """Returns action probabilities dict for a single batch."""
        cur_player = state.current_player()
        legal_actions = state.legal_actions(cur_player)
        # TODO confirm difference information_state_tensor or observation_tensor
        info_state = torch.Tensor(state.observation_tensor(cur_player))
        print(policy_netowork_list[cur_player].state_dict(), "aaaaaaaaa",
            sampled_policy_network_list[cur_player].state_dict())
        exit()
        probs = policy_netowork_list[cur_player](info_state)
        print(probs)
        return {action: probs.detach().numpy()[action] for action in legal_actions}
      average_policy = policy.tabular_policy_from_callable(
        GAME, action_probabilities)
      conv = exploitability.nash_conv(game, average_policy)
      print(f"Iter: {t}, NashConv: {conv}")

    #TODO use dict or namedtuple
    D.append([global_critic_network.state_dict(),
        [net.state_dict() for net in policy_netowork_list]])
