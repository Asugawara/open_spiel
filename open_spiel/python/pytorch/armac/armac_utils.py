import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple

from nets import RNN

Transition = namedtuple(
    "Transition",
    "player_id reward history regret sampled_policy")


class TrajectoryState(object):
  """A particular point along a trajectory."""

  def __init__(self, observation, current_player, legals_mask, action, policy,
               value):
    self.observation = observation
    self.current_player = current_player
    self.legals_mask = legals_mask
    self.action = action
    self.policy = policy
    self.value = value


class Trajectory(object):
  """A sequence of observations, actions and policies, and the outcomes."""

  def __init__(self):
    self.states = []
    self.returns = None

  def add(self, information_state, action, policy):
    self.states.append((information_state, action, policy))

class ReplayBuffer(object):
  """ReplayBuffer of fixed size with a FIFO replacement policy.

  Stored transitions can be sampled uniformly.

  The underlying datastructure is a ring buffer, allowing 0(1) adding and
  sampling.
  """

  def __init__(self, replay_buffer_capacity):
    self._replay_buffer_capacity = replay_buffer_capacity
    self._data = []
    self._next_entry_index = 0

  def add(self, element):
    """Adds `element` to the buffer.

    If the buffer is full, the oldest element will be replaced.

    Args:
      element: data to be added to the buffer.
    """
    if len(self._data) < self._replay_buffer_capacity:
      self._data.append(element)
    else:
      self._data[self._next_entry_index] = element
      self._next_entry_index += 1
      self._next_entry_index %= self._replay_buffer_capacity

  def sample(self, num_samples):
    """Returns `num_samples` uniformly sampled from the buffer.

    Args:
      num_samples: `int`, number of samples to draw.

    Returns:
      An iterable over `num_samples` random elements of the buffer.

    Raises:
      ValueError: If there are less than `num_samples` elements in the buffer
    """
    if len(self._data) < num_samples:
      raise ValueError("{} elements could not be sampled from size {}".format(
          num_samples, len(self._data)))
    return random.sample(self._data, num_samples)

  def __len__(self):
    return len(self._data)

  def __iter__(self):
    return iter(self._data)


class ARMACActor:
  def __init__(self, game, policy_net, sampled_joint_policy, sampled_critic_net, episodes):
    self.game = game
    self.num_players = game.num_players()
    self.plicy_net = policy_net
    self.sampled_joint_policy = sampled_joint_policy
    self.sampled_critic_net = sampled_critic_net
    self.episodes = episodes
    self.buffer = ReplayBuffer(1024)

  def _play_game(self, player_id):
    trajectory = Trajectory()
    state = self.game.new_initial_state()
    while not state.is_terminal():
      if state.is_chance_node():
        outcomes = state.chance_outcomes()
        action_list, prob_list = zip(*outcomes)
        action = np.random.choice(action_list, p=prob_list)
      else:
        info_state = torch.Tensor(state.observation_tensor(player_id))
        if state.current_player() == player_id:
          policy = self.plicy_net(info_state).detach()
        else:
          policy = self.sampled_joint_policy[state.current_player()](info_state).detach()
        legal_actions_mask = torch.LongTensor(state.legal_actions_mask())
        policy = F.softmax(policy, dim=0).mul(legal_actions_mask)
        action = torch.max(policy, dim=0)[1]
        trajectory.states.append(TrajectoryState(
            state.observation_tensor(), state.current_player(),
            state.legal_actions_mask(), action, policy, value=0))
      state.apply_action(action)

    trajectory.returns = state.returns()
    return trajectory


  def act(self, player_id):
    for _ in range(self.episodes):
      trajectory = self._play_game(player_id)
      # TODO use batch
      for history in trajectory.states:
        if history.current_player == player_id:
          info_state = torch.Tensor(history.observation)

          critic_value = self.sampled_critic_net(info_state).detach()
          action_value = self.sampled_joint_policy[player_id](info_state).detach()
          regret = critic_value - action_value
          transition = Transition(
              player_id=player_id,
              reward=trajectory.returns[player_id],
              history=history,
              regret=regret,
              sampled_policy=self.sampled_joint_policy[player_id]
          )
          self.buffer.add(transition)


class ARMACLearner:
  def __init__(self, buffer, batch_size, critic_net, policy_net):
    self.buffer = buffer
    self.batch_size = batch_size
    self.critic_net = critic_net
    self.policy_net = policy_net
    self.policy_optimzer = optim.SGD(self.policy_net.parameters(), lr=0.01)
    self.eligibility_trace = None

  def learn(self):
    transitions = self.buffer.sample(self.batch_size)
    for i in range(len(transitions) - 1):
      self._critic_update(transitions[i], transitions[i+1], 0.9, 1, 0.9)
      # TODO use batch
      self._policy_update(transitions[i])

  def _critic_update(self, transition, next_trasition, decay, step_size, lambda_):
    history = transition.history
    history_info_state = torch.Tensor(history.observation)
    if torch.is_tensor(self.eligibility_trace):
      self.eligibility_trace *= (decay * lambda_ * transition.sampled_policy(history_info_state))
    else:
      self.eligibility_trace = 1

    next_info_state = torch.Tensor(next_trasition.history.observation)
    next_state_policy = next_trasition.sampled_policy(next_info_state)
    next_expected_value= torch.mul(next_state_policy, self.critic_net(next_info_state))
    td_error = (next_trasition.reward + 
        decay * torch.sum(next_expected_value).item() - 
            self.critic_net(history_info_state).item())
  
    for param in self.critic_net.state_dict():
      self.critic_net.state_dict()[param] += step_size * self.eligibility_trace * td_error

  # TODO use batch transition -> transitions
  def _policy_update(self, transition):
    info_state = torch.Tensor(transition.history.observation)
    regret = transition.regret
    
    estimated = self.policy_net(info_state)
    policy_loss = torch.mean(
          F.mse_loss(estimated, regret))
    self.policy_optimzer.zero_grad()
    policy_loss.backward()
    self.policy_optimzer.step()
