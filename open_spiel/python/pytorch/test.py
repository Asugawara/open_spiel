import numpy as np
import torch
import torch.nn as nn

from open_spiel.python.pytorch import rcfr
import pyspiel

_GAME = pyspiel.load_game('kuhn_poker')


def _new_model():
  return rcfr.DeepRcfrModel(
      _GAME,
      num_hidden_layers=1,
      num_hidden_units=13,
      num_hidden_factors=1,
      use_skip_connections=True)

def test_rcfr_with_buffer():
  buffer_size = 12
  num_epochs = 100
  num_iterations = 100
  models = [_new_model() for _ in range(_GAME.num_players())]

  patient = rcfr.ReservoirRcfrSolver(_GAME, models, buffer_size=buffer_size)

  def _train(model, data):
    data = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)

    loss_fn = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, amsgrad=True)
    for epoch in range(num_epochs):
      for x, y in data:
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

  average_policy = patient.average_policy()
  print(pyspiel.nash_conv(_GAME, average_policy), 0.91)

  for _ in range(num_iterations):
    patient.evaluate_and_update_policy(_train)

    average_policy = patient.average_policy()
    print(pyspiel.nash_conv(_GAME, average_policy), 0.91)
    
    
def test_cfr():
  root = rcfr.RootStateWrapper(_GAME.new_initial_state())
  num_half_iterations = 100

  cumulative_regrets = [np.zeros(n) for n in root.num_player_sequences]
  cumulative_reach_weights = [np.zeros(n) for n in root.num_player_sequences]

  average_profile = root.sequence_weights_to_tabular_profile(
      cumulative_reach_weights)
  # parameterized.TestCase

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
  #test_rcfr_with_buffer()
  test_cfr()