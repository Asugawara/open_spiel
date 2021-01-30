import torch
import torch.nn as nn 
import torch.nn.functional as F

class BaseModel(nn.Module):
  def __init__(self,
               input_size):

    super(BaseModel, self).__init__()
    self.layer1 = nn.Linear(input_size, 128)
    self.layer2 = nn.Linear(input_size, 128)

  def forward(self, x):
    h1 = self.layer1(x)
    h2 = F.relu(h1)
    h3 = h1 + self.layer2(h2)
    return F.relu(h3)

class RNN(nn.Module):
  def __init__(self, input_size):
    super(RNN, self).__init__()
    self.layer1 = nn.Linear(input_size, 256)
    self.rnn = nn.LSTM(512, 256)
    self.B = BaseModel()

  def forward(self, x):
    h1 = self.layer1(x)
    h2 = self.rnn(h1)
    return self.B(h2)

class GlobalCritic(nn.Module):
  def __init__(self, input_size):
    super(GlobalCritic, self).__init__()
    self.layer1 = nn.Linear(input_size, 128)
    self.layer2 = nn.Linear(input_size, 128)
    self.B = BaseModel()

  def forward(self, state1, state2):
    a0 = self.layer1(state1) + self.layer2(state2)
    a1 = self.layer1(state2) + self.layer2(state1)
    h1 = torch.cat(a0, a1)
    h2 = self.B(h1)
    return h2
