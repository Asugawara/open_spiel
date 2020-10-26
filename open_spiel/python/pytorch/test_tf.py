import numpy as np
# Note: this import needs to come before Tensorflow to fix a malloc error.
import pyspiel  # pylint: disable=g-bad-import-order
import tensorflow.compat.v1 as tf

from open_spiel.python.algorithms import rcfr

# Temporarily disable TF2 behavior while the code is not updated.
tf.disable_v2_behavior()

tf.enable_eager_execution()

_GAME = pyspiel.load_game('kuhn_poker')
_BOOLEANS = [False, True]

def _new_model():
  return rcfr.DeepRcfrModel(
      _GAME,
      num_hidden_layers=1,
      num_hidden_units=13,
      num_hidden_factors=1,
      use_skip_connections=True)

    
def test_cfr():
  root = rcfr.RootStateWrapper(_GAME.new_initial_state())
  num_half_iterations = 100

  cumulative_regrets = [np.zeros(n) for n in root.num_player_sequences]
  cumulative_reach_weights = [np.zeros(n) for n in root.num_player_sequences]

  average_profile = root.sequence_weights_to_tabular_profile(
      cumulative_reach_weights)
  print(pyspiel.nash_conv(_GAME, average_profile), 0.91)

  regret_player = 0
  for _ in range(num_half_iterations):
    reach_weights_player = 1 if regret_player == 0 else 0

    regrets, reach = root.counterfactual_regrets_and_reach_weights(
        regret_player, reach_weights_player, *rcfr.relu(cumulative_regrets))

    cumulative_regrets[regret_player] += regrets
    cumulative_reach_weights[reach_weights_player] += reach

    regret_player = reach_weights_player

  average_profile = root.sequence_weights_to_tabular_profile(
      cumulative_reach_weights)
  print(pyspiel.nash_conv(_GAME, average_profile), 0.27)
  
  
if __name__ == '__main__':
  test_cfr()