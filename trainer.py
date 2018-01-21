import torch
from torch import nn
from torch.autograd import Variable

from model import RecurrentAttention


class Trainer(object):
    """
    Trainer encapsulates all the logic necessary for
    training the Recurrent Attention Model.

    All hyperparameters are provided by the user in the
    config file.
    """
    def __init__(self, config, data_loader):
        

    def train(self):
        pass

    def train_one_epoch(self, epoch):
        pass
