import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.distributions import Normal


"""
Def:

- policy: maps from state to action -> pi(a|s)
- value: evaluates how good a state is -> E[cumulative discounted rewards]

Here:

- state: the hidden layer of the RNN
- action: location action (coords) + classification action (predict target)
- policy: location_network(hidden) -> mu -> Normal(mu, fixed_std) -> loc_(t+1)


Gist:

- glimpse operation is not differentiable w.r.t location vector [x, y] ==> we need to use RL
- have a location network take as input state vector and output a vector of means.
- generate a location vector by sampling from a normal dist. parametrized by those means.
- use REINFORCE to optimize location vectors such that they minimize reward function.
- in our case, reward function is 1 if correctly classified, else 0 (so only appears at end).

- In REINFORCE,  
"""

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

    def forward(self, x):
