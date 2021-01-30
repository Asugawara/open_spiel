from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from torch.testing._internal.common_utils import run_tests, TestCase

import pyspiel

from armac_utils import ARMACActor


class ARMACTest(parameterized.TestCase, TestCase):

  @parameterized.parameters("tic_tac_toe", "kuhn_poker", "liars_dice")
  def test_play_game(self, game):
    GAME = pyspiel.load_game(game)
    actions = GAME.num_distinct_actions()
    num_palyers = GAME.num_players()
    test_policy = [[1/actions]*actions]*num_palyers
    actor = ARMACActor(GAME, test_policy, None, None, None)
    trajectory = actor._play_game()
    print(trajectory.states[0].action)

  @parameterized.parameters("tic_tac_toe", "kuhn_poker", "liars_dice")
  def test_act(self, game):
    GAME = pyspiel.load_game(game)
    actions = GAME.num_distinct_actions()
    num_palyers = GAME.num_players()
    test_policy = [[1/actions]*actions]*num_palyers
    actor = ARMACActor(GAME, test_policy, None, None, None)
    actor.act(1)



if __name__=="__main__":
  run_tests()

