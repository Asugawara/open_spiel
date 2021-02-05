import torch
import torch.nn as nn 
import torch.nn.functional as F

class BaseModel(nn.Module):
  def __init__(self,
               input_size):

    super(BaseModel, self).__init__()
    self.layer1 = nn.Linear(input_size, 128)
    self.layer2 = nn.Linear(256, 128)

  def forward(self, x):
    h1 = self.layer1(x) 
    h2 = torch.cat([F.relu(h1), F.relu(-1 * h1)])
    h3 = h1 + self.layer2(h2)
    return torch.cat([F.relu(h3), F.relu(-1 * h3)])

class RNN(nn.Module):
  def __init__(self, input_size, hidden_size=256):
    super(RNN, self).__init__()
    self.hidden_size = hidden_size
    self.layer1 = nn.Linear(input_size, hidden_size)
    self.rnn = nn.LSTM(hidden_size*2, hidden_size)

  def forward(self, x):
    h1 = self.layer1(x)
    h2 = torch.cat([F.relu(h1), F.relu(-1 * h1)], dim=0)
    _, (h3, _) = self.rnn(h2.view(1, -1, self.hidden_size*2))
    return h3.view(-1, self.hidden_size)

class MLP_torso(nn.Module):
  def __init__(self, input_size, hidden_size=256):
    super(MLP_torso, self).__init__()
    self.layer1 = nn.Linear(input_size, hidden_size)
    self.layer2 = nn.Linear(hidden_size, hidden_size)

  def forward(self, x):
    h1 = self.layer1(x)
    return self.layer2(h1)

class GlobalCritic(nn.Module):
  def __init__(self, player_head, input_size):
    super(GlobalCritic, self).__init__()
    self.player_head = player_head
    self.player_num = len(player_head)
    self.layer1 = nn.Linear(input_size, 128)
    self.layer2 = nn.Linear(input_size, 128)
    self.B = BaseModel(input_size)

  def forward(self, x):
    state = [self.player_head[i](x[i]) for i in range(self.player_num)]
    a0 = self.layer1(state[0]) + self.layer2(state[1])
    a1 = self.layer1(state[1]) + self.layer2(state[0])
    h1 = torch.cat([a0, a1], dim=0)
    h2 = self.B(h1)
    return h2

# immediate regret = nn.Sequential(BaseModel, nn.linear(input, actions))
# average regret = nn.Sequential(BaseModel, nn.linear(input, actions))
# mean policy = nn.Sequential(BaseModel, nn.linear(input, actions))
# share B