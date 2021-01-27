from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import os
from absl import logging
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from open_spiel.python import rl_agent
from open_spiel.python.pytorch.dqn import SonnetLinear
#from open_spiel.python.pytorch.losses import rl_losses
from losses import rl_losses


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


class RetrospectiveBuffer(object):
  """A fixed size buffer that keeps the newest values."""

  def __init__(self, max_size):
    self.max_size = max_size
    self.data = []
    self.theta = []
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

class MLPTorso(nn.Module):
  """A specialized half-MLP module when constructing multiple heads.

  Note that every layer includes a ReLU non-linearity activation.
  """

  def __init__(self,
               input_size,
               hidden_sizes):
    """Create the MLPTorso.

    Args:
      input_size: (int) number of inputs
      hidden_sizes: (list) sizes (number of units) of each hidden layer
    """

    super(MLPTorso, self).__init__()
    self._layers = []
    # Hidden layers
    for size in hidden_sizes:
      self._layers.append(SonnetLinear(in_size=input_size, out_size=size))
      input_size = size

    self.model = nn.ModuleList(self._layers)

  def forward(self, x):
    for layer in self.model:
      x = layer(x)
    return x

class ARMAC(rl_agent.AbstractAgent):
  """RPG Agent implementation in PyTorch.

  See open_spiel/python/examples/single_agent_catch.py for an usage example.
  """

  def __init__(self,
               num_players,
               player_id,
               info_state_size,
               num_actions,
               loss_str="a2c",
               loss_class=None,
               hidden_layers_sizes=(128,),
               batch_size=16,
               critic_learning_rate=0.01,
               pi_learning_rate=0.001,
               entropy_cost=0.01,
               num_critic_before_pi=8,
               additional_discount_factor=1.0,
               max_global_gradient_norm=None,
               optimizer_str="sgd",max_size=1000):
    """Initialize the PolicyGradient agent.

    Args:
      player_id: int, player identifier. Usually its position in the game.
      info_state_size: int, info_state vector size.
      num_actions: int, number of actions per info state.
      loss_str: string or None. If string, must be one of ["rpg", "qpg", "rm",
        "a2c"] and defined in `_get_loss_class`. If None, a loss class must be
        passed through `loss_class`. Defaults to "a2c".
      loss_class: Class or None. If Class, it must define the policy gradient
        loss. If None a loss class in a string format must be passed through
        `loss_str`. Defaults to None.
      hidden_layers_sizes: iterable, defines the neural network layers. Defaults
          to (128,), which produces a NN: [INPUT] -> [128] -> ReLU -> [OUTPUT].
      batch_size: int, batch size to use for Q and Pi learning. Defaults to 128.
      critic_learning_rate: float, learning rate used for Critic (Q or V).
        Defaults to 0.001.
      pi_learning_rate: float, learning rate used for Pi. Defaults to 0.001.
      entropy_cost: float, entropy cost used to multiply the entropy loss. Can
        be set to None to skip entropy computation. Defaults to 0.001.
      num_critic_before_pi: int, number of Critic (Q or V) updates before each
        Pi update. Defaults to 8 (every 8th critic learning step, Pi also
        learns).
      additional_discount_factor: float, additional discount to compute returns.
        Defaults to 1.0, in which case, no extra discount is applied.  None that
        users must provide *only one of* `loss_str` or `loss_class`.
      max_global_gradient_norm: float or None, maximum global norm of a gradient
        to which the gradient is shrunk if its value is larger.
      optimizer_str: String defining which optimizer to use. Supported values
        are {sgd, adam}
    """
    assert bool(loss_str) ^ bool(loss_class), "Please provide only one option."
    self._kwargs = locals()
    loss_class = loss_class if loss_class else self._get_loss_class(loss_str)
    self._loss_class = loss_class

    self.num_players = num_players
    self.player_id = player_id
    self._num_actions = num_actions
    self._layer_sizes = hidden_layers_sizes
    self._batch_size = batch_size
    self._extra_discount = additional_discount_factor
    self._num_critic_before_pi = num_critic_before_pi
    self._max_global_gradient_norm = max_global_gradient_norm

    self._episode_data = []
    #self._dataset = collections.defaultdict(list)
    self._buffer = RetrospectiveBuffer(max_size)
    self._prev_time_step = None
    self._prev_action = None

    # Step counters
    self._step_counter = 0
    self._episode_counter = 0
    self._num_learn_steps = 0

    # Keep track of the last training loss achieved in an update step.
    self._last_loss_value = None

    # Network
    # activate final as we plug logit and qvalue heads afterwards.
    self._net_torso = MLPTorso(info_state_size, self._layer_sizes)
    torso_out_size = self._layer_sizes[-1]
    self._policy_logits_layer = SonnetLinear(
        torso_out_size,
        self._num_actions,
        activate_relu=False)
    # Do not remove policy_logits_network. Even if it's not used directly here,
    # other code outside this file refers to it.
    self.policy_logits_network = nn.Sequential(
        self._net_torso, self._policy_logits_layer)

    self._savers = []

    # Add baseline (V) head for A2C (or Q-head for QPG / RPG / RMPG)
    if optimizer_str == "adam":
      self._critic_optimizer = optim.Adam
    elif optimizer_str == "sgd":
      self._critic_optimizer = optim.SGD
    else:
      raise ValueError("Not implemented, choose from 'adam' and 'sgd'.")

    if loss_class.__name__ == "BatchA2CLoss":
      self._baseline_layer = SonnetLinear(
          torso_out_size, 1, activate_relu=False)
      self._critic_network = nn.Sequential(
        self._net_torso, self._baseline_layer)
    else:
      self._q_values_layer = SonnetLinear(
          torso_out_size,
          self._num_actions,
          activate_relu=False)
      self._critic_network = nn.Sequential(
        self._net_torso, self._q_values_layer)

    self._critic_optimizer = self._critic_optimizer(
        self._critic_network.parameters(), lr=critic_learning_rate)

    # Pi loss
    self.pg_class = loss_class(entropy_cost=entropy_cost)
    self._pi_network = nn.Sequential(
        self._net_torso, self._policy_logits_layer)
    if optimizer_str == "adam":
      self._pi_optimizer = optim.Adam(
          self._pi_network.parameters(),
          lr=pi_learning_rate)
    elif optimizer_str == "sgd":
      self._pi_optimizer = optim.SGD(
          self._pi_network.parameters(),
          lr=pi_learning_rate)

    self._loss_str = loss_str

  def _get_loss_class(self, loss_str):
    if loss_str == "rpg":
      return rl_losses.BatchRPGLoss
    elif loss_str == "qpg":
      return rl_losses.BatchQPGLoss
    elif loss_str == "rm":
      return rl_losses.BatchRMLoss
    elif loss_str == "a2c":
      return rl_losses.BatchA2CLoss

  def minimize_with_clipping(self, model, optimizer, loss, j):
    import copy
    self._buffer.theta[j] = copy(model.state_dict())
    optimizer.zero_grad()
    loss.backward()
    if self._max_global_gradient_norm is not None:
      nn.utils.clip_grad_norm_(model.parameters(), self._max_global_gradient_norm)
    optimizer.step()

  def _act(self, info_state, legal_actions):
    # Make a singleton batch for NN compatibility: [1, info_state_size]
    info_state = torch.Tensor(np.reshape(info_state, [1, -1]))
    torso_out = self._net_torso(info_state)
    self._policy_logits = self._policy_logits_layer(torso_out)
    policy_probs = F.softmax(self._policy_logits, dim=1).detach()

    # Remove illegal actions, re-normalize probs
    probs = np.zeros(self._num_actions)
    probs[legal_actions] = policy_probs[0][legal_actions]
    if sum(probs) != 0:
      probs /= sum(probs)
    else:
      probs[legal_actions] = 1 / len(legal_actions)
    action = np.random.choice(len(probs), p=probs)
    return action, probs

  # TODO how to make joint_policy
  def _play_game(game, joint_policy):
    """Play one game, return the trajectory."""
    import copy
    trajectory = Trajectory()
    actions = []
    state = game.new_initial_state()
    while not state.is_terminal():
      root = copy.deepcopy(state)
      action = joint_policy[state.current_player()][state.info_state]  
      trajectory.states.append(TrajectoryState(
          state.observation_tensor(), state.current_player(),
          state.legal_actions_mask(), action, joint_policy,
          root.total_reward / root.explore_count))
      action_str = state.action_to_string(state.current_player(), action)
      actions.append(action_str)
      state.apply_action(action)

    trajectory.returns = state.returns()
    return trajectory

  def _counterfactual_regrets(self, regret_player, trajectory):
    regrets = np.zeros(self.num_player_sequences[regret_player])
    reach_weights = np.zeros(self.num_player_sequences[reach_weight_player])
    def _walk_descendants(state, reach_probabilities, chance_reach_probability):
      """Compute `state`'s counterfactual regrets and reach weights.

      Args:
        state: An OpenSpiel `State`.
        reach_probabilities: The probability that each player plays to reach
          `state`'s history.
        chance_reach_probability: The probability that all chance outcomes in
          `state`'s history occur.

      Returns:
        The counterfactual value of `state`'s history.
      Raises:
        ValueError if there are too few sequence weights at any information
        state for any player.
      """

      if state.is_terminal():
        player_reach = (
            np.prod(reach_probabilities[:regret_player]) *
            np.prod(reach_probabilities[regret_player + 1:]))

        counterfactual_reach_prob = player_reach * chance_reach_probability
        u = self.terminal_values[state.history_str()]
        return u[regret_player] * counterfactual_reach_prob

      elif state.is_chance_node():
        v = 0.0
        for action, action_prob in state.chance_outcomes():
          v += _walk_descendants(
              state.child(action), reach_probabilities,
              chance_reach_probability * action_prob)
        return v

      player = state.current_player()
      info_state = state.information_state_string(player)
      sequence_idx_offset = self.info_state_to_sequence_idx[info_state]
      actions = state.legal_actions(player)

      sequence_idx_end = sequence_idx_offset + len(actions)
      my_sequence_weights = sequence_weights[player][
          sequence_idx_offset:sequence_idx_end]

      if len(my_sequence_weights) < len(actions):
        raise ValueError(
            ("Invalid policy: Policy {player} at sequence offset "
             "{sequence_idx_offset} has only {policy_len} elements but there "
             "are {num_actions} legal actions.").format(
                 player=player,
                 sequence_idx_offset=sequence_idx_offset,
                 policy_len=len(my_sequence_weights),
                 num_actions=len(actions)))

      policy = normalized_by_sum(my_sequence_weights)
      action_values = np.zeros(len(actions))
      state_value = 0.0

      is_reach_weight_player_node = player == reach_weight_player
      is_regret_player_node = player == regret_player

      reach_prob = reach_probabilities[player]
      for action_idx, action in enumerate(actions):
        action_prob = policy[action_idx]
        next_reach_prob = reach_prob * action_prob

        if is_reach_weight_player_node:
          reach_weight_player_plays_down_this_line = next_reach_prob > 0
          if not reach_weight_player_plays_down_this_line:
            continue
          sequence_idx = sequence_idx_offset + action_idx
          reach_weights[sequence_idx] += next_reach_prob

        reach_probabilities[player] = next_reach_prob

        action_value = _walk_descendants(
            state.child(action), reach_probabilities, chance_reach_probability)

        if is_regret_player_node:
          state_value = state_value + action_prob * action_value
        else:
          state_value = state_value + action_value
        action_values[action_idx] = action_value

      reach_probabilities[player] = reach_prob

      if is_regret_player_node:
        regrets[sequence_idx_offset:sequence_idx_end] += (
            action_values - state_value)
      return state_value
    _walk_descendants(trajectory.state, np.ones(num_players), 1.0)
    return regrets


  def step(self, game, time_step, n_step, is_evaluation=False):
    """Returns the action to be taken and updates the network if needed.

    Args:
      time_step: an instance of rl_environment.TimeStep.
      is_evaluation: bool, whether this is a training or evaluation call.

    Returns:
      A `rl_agent.StepOutput` containing the action probs and chosen action.
    """
    # Act step: don't act at terminal info states or if its not our turn.
    if (not time_step.last()) and (
        time_step.is_simultaneous_move() or
        self.player_id == time_step.current_player()):
      info_state = time_step.observations["info_state"][self.player_id]
      legal_actions = time_step.observations["legal_actions"][self.player_id]
      action, probs = self._act(info_state, legal_actions)
    else:
      action = None
      probs = []

    j = random.sample(range(n_step), 1)
    trajectory = self._play_game(game, joint_policy)
    data = [self.player_id, j, trajectory.returns]

    if not is_evaluation:
      self._step_counter += 1
      for history in trajectory:
        if self.player_id == history.current_player():
          s = history.state
          r = calc_regret(history)
          a = trajectory.action
          d.extend([history, s, a, r, policy(s)])



      # Add data points to current episode buffer.
      # if self._prev_time_step:
      #  self._add_transition(time_step)

      # Episode done, add to dataset and maybe learn.
      #if time_step.last():
      #  self._add_episode_data_to_dataset()
      #  self._episode_counter += 1

        if len(self._dataset["returns"]) >= self._batch_size:
          self._critic_update()
          self._num_learn_steps += 1
          if self._num_learn_steps % self._num_critic_before_pi == 0:
            self._pi_update()
          self._dataset = collections.defaultdict(list)

        self._prev_time_step = None
        self._prev_action = None
        return
      else:
        self._prev_time_step = time_step
        self._prev_action = action

    return rl_agent.StepOutput(action=action, probs=probs)


  @property
  def loss(self):
    return (self._last_critic_loss_value, self._last_pi_loss_value)

  def _add_episode_data_to_dataset(self):
    """Add episode data to the buffer."""
    info_states = [data.info_state for data in self._episode_data]
    rewards = [data.reward for data in self._episode_data]
    discount = [data.discount for data in self._episode_data]
    actions = [data.action for data in self._episode_data]

    # Calculate returns
    returns = np.array(rewards)
    for idx in reversed(range(len(rewards[:-1]))):
      returns[idx] = (
          rewards[idx] +
          discount[idx] * returns[idx + 1] * self._extra_discount)

    # Add flattened data points to dataset
    self._dataset["actions"].extend(actions)
    self._dataset["returns"].extend(returns)
    self._dataset["info_states"].extend(info_states)
    self._episode_data = []

  def _add_transition(self, time_step):
    """Adds intra-episode transition to the `_episode_data` buffer.

    Adds the transition from `self._prev_time_step` to `time_step`.

    Args:
      time_step: an instance of rl_environment.TimeStep.
    """
    assert self._prev_time_step is not None
    legal_actions = (
        self._prev_time_step.observations["legal_actions"][self.player_id])
    legal_actions_mask = np.zeros(self._num_actions)
    legal_actions_mask[legal_actions] = 1.0
    transition = Transition(
        info_state=(
            self._prev_time_step.observations["info_state"][self.player_id][:]),
        action=self._prev_action,
        reward=time_step.rewards[self.player_id],
        discount=time_step.discounts[self.player_id],
        legal_actions_mask=legal_actions_mask)

    self._episode_data.append(transition)

  def _critic_update(self):
    """Compute the Critic loss on sampled transitions & perform a critic update.

    Returns:
      The average Critic loss obtained on this batch.
    """
    # TODO(author3): illegal action handling.
    info_state = torch.Tensor(self._dataset["info_states"])
    action = torch.LongTensor(self._dataset["actions"])
    return_ =  torch.Tensor(self._dataset["returns"])
    torso_out = self._net_torso(info_state)


    # Critic loss
    # Baseline loss in case of A2C
    if self._loss_class.__name__ == "BatchA2CLoss":
      baseline = torch.squeeze(self._baseline_layer(torso_out), dim=1)
      critic_loss = torch.mean(
          F.mse_loss(baseline, return_))
      self.minimize_with_clipping(
          self._baseline_layer,
          self._critic_optimizer,
          critic_loss)
    else:
      # Q-loss otherwise.
      q_values = self._q_values_layer(torso_out)
      action_indices = torch.stack(
          [torch.range(q_values.shape[0]), action], dim=-1)
      value_predictions = torch.gather_nd(q_values, action_indices)
      critic_loss = torch.mean(
          F.mse_loss(
              value_predictions, return_))
      self.minimize_with_clipping(
          self._q_values_layer,
          self._critic_optimizer,
          critic_loss)
    self._last_critic_loss_value = critic_loss
    return critic_loss

  def _pi_update(self):
    """Compute the Pi loss on sampled transitions and perform a Pi update.

    Returns:
      The average Pi loss obtained on this batch.
    """
    # TODO(author3): illegal action handling.
    info_state = torch.Tensor(self._dataset["info_states"])
    action = torch.LongTensor(self._dataset["actions"])
    return_ =  torch.Tensor(self._dataset["returns"])
    torso_out = self._net_torso(info_state)
    self._policy_logits = self._policy_logits_layer(torso_out)

    if self._loss_class.__name__ == "BatchA2CLoss":
      baseline = torch.squeeze(self._baseline_layer(torso_out), dim=1)
      pi_loss = self.pg_class.loss(
          policy_logits=self._policy_logits,
          baseline=baseline,
          actions=action,
          returns=return_)
      self.minimize_with_clipping(
          self._baseline_layer,
          self._pi_optimizer,
          pi_loss)
    else:
      q_values = self._q_values_layer(torso_out)
      pi_loss = self.pg_class.loss(
          policy_logits=self._policy_logits, action_values=q_values)
      self.minimize_with_clipping(
          self._q_values_layer,
          self._pi_optimizer,
          pi_loss)
    self._last_pi_loss_value = pi_loss
    return pi_loss