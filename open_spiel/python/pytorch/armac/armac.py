import random
import sys
import pyspiel
from .armac_utils import ARMACActor, ARMACLearner


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





def armac(game, epochs, episodes, learning_steps):
  game = pyspiel.load_game(game)
  policy_storage = []
  global_crtic = Crtic()
  for t in range(epochs):
    D = Buffer()
    for ep in range(episodes):
      policy = random.sample(policy_storage)
      actor = ARMACActor(game.num_players(), policy)
      d = actor.act()
      D.append(d)

    for l_steps in range(learning_steps):
      batch = D.sample()
      learner = ARMACLearner()
      learner.learn(batch)

    policy_storage.append(global_crtic.state_dict())