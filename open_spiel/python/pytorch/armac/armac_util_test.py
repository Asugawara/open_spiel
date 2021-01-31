from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import copy
from torch.testing._internal.common_utils import run_tests, TestCase
import torch
import torch.nn as nn

import pyspiel

from nets import BaseModel, GlobalCritic, RNN
from armac_utils import ARMACActor


def setUp(game):
  GAME = pyspiel.load_game(game)
  num_actions = GAME.num_distinct_actions()
  num_players = GAME.num_players()
  observation_size = GAME.observation_tensor_size()
  RNN_HIDDEN_SIZE = 256
  B = BaseModel(RNN_HIDDEN_SIZE)
  sampled_joint_policy = []
  player_rnn = []
  for _ in range(num_players):
    rnn = RNN(observation_size, RNN_HIDDEN_SIZE)
    player_rnn.append(rnn)
    print('rnn:',id(rnn))
    policy_logits_layer = nn.Linear(RNN_HIDDEN_SIZE, num_actions)
    print('policy_logits_layer: ', id(policy_logits_layer))
    sampled_joint_policy.append(nn.Sequential(rnn, B, policy_logits_layer))
  
  global_critic = GlobalCritic(player_rnn, RNN_HIDDEN_SIZE)
  out_critic = nn.Linear(RNN_HIDDEN_SIZE, 1)
  global_critic_network = nn.Sequential(global_critic, out_critic)

  B_value = BaseModel(observation_size)
  immediate_regret_out = nn.Linear(RNN_HIDDEN_SIZE, num_actions)
  immediate_regret_net = nn.Sequential(B_value, immediate_regret_out)

  return GAME, sampled_joint_policy, global_critic_network, immediate_regret_net


class ARMACTest(parameterized.TestCase, TestCase):
  @parameterized.parameters("tic_tac_toe", "kuhn_poker", "liars_dice")
  def test_play_game(self, game):
    GAME, sampled_joint_policy, _, _ = setUp(game)
    for i in range(GAME.num_players()):
      actor = ARMACActor(GAME, 
                         sampled_joint_policy[i], 
                         sampled_joint_policy, 
                         None, None, None)
      trajectory = actor._play_game(player_id=i)
      print(trajectory.states[0].action)

  @parameterized.parameters("tic_tac_toe", "kuhn_poker", "liars_dice")
  def test_act(self, game):
    GAME, sampled_joint_policy, global_critic_net, immediate_regret_net = setUp(game)
    for i in range(GAME.num_players()):
      actor = ARMACActor(GAME, 
                         sampled_joint_policy[i], 
                         sampled_joint_policy, 
                         global_critic_net, 
                         immediate_regret_net, None)
      actor.act(player_id=i)



if __name__=="__main__":
  run_tests()

