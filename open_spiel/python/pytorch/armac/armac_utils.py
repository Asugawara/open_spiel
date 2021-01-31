import numpy as np
import torch
import torch.nn.functional as F
from nets import RNN
from losses.rl_losses import compute_regrets

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

class ARMACActor:
  def __init__(self, game, policy_net, sampled_joint_policy, sampled_critic_net, sampled_value_net, episodes):
    self.game = game
    self.num_players = game.num_players()
    self.plicy_net = policy_net
    self.sampled_critic_net = sampled_critic_net
    self.sampled_value_net = sampled_value_net
    self.sampled_joint_policy = sampled_joint_policy
    self.episodes = episodes

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
    d = []
    trajectory = self._play_game(player_id)
    tmp = [player_id, trajectory.returns]
    for history in trajectory.states:
      if history.current_player == player_id:
        info_state = torch.Tensor(history.observation)
        legals_mask = torch.LongTensor(history.legals_mask)

        policy = self.sampled_critic_net(info_state).detach()
        policy_logits = F.softmax(policy, dim=0).mul(legals_mask)
        action_value = self.sampled_value_net(info_state)
        regret=None
        #regret = compute_regrets(policy_logits, action_value)
        f = tmp.copy()
        f.append([history, regret, self.sampled_joint_policy])
        d.append(f)
    return d


class ARMACLearner:
  pass