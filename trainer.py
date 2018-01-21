import torch
from torch import nn
from torch.autograd import Variable

from model import RecurrentAttention


class Trainer(object):
    """
    Trainer encapsulates all the logic necessary for
    training the Recurrent Attention Model.

    The cross-entropy loss is used to train the action
    network and the backpropagated gradients update
    the weights of the core and glimpse networks.

    The location network on the other hand, is trained
    with REINFORCE since it is not differentiable.

    All hyperparameters are provided by the user in the
    config file.
    """
    def __init__(self, config, data_loader):
        pass

    def train(self):
        pass

    def train_one_epoch(self, epoch):
        pass
