import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from modules import glimpse_network, core_network
from modules import action_network, location_network


class RecurrentAttention(nn.Module):
    """
    A Recurrent Model of Visual Attention (RAM) [1].

    RAM is a recurrent neural network which processes
    inputs sequentially, attending to different locations
    within the image one at a time, and incrementally
    combining information from these fixations to build
    up a dynamic internal representation of the image.

    References
    ----------
    - Minh et. al., https://arxiv.org/abs/1406.6247
    """
    def __init__(self,
                 h_g=128,
                 h_l=128,
                 g=64,
                 k=3,
                 s=2,
                 c=3,,
                 hidden_size=256,
                 num_classes=10,
                 std=0.11):
        """
        Initialize the recurrent attention model and its
        different submodules.
        
        Args
        ----
        - h_g: hidden layer size of the fc layer for `phi`.
        - h_l: hidden layer size of the fc layer for `l`.
        - g: size of the square patches in the glimpses extracted
          by the retina.
        - k: number of patches to extract per glimpse.
        - s: scaling factor that controls the size of successive patches.
        - c: number of channels in each image.
        - hidden_size: hidden size of the rnn.
        - num_classes: number of classes in the dataset.
        - std: standard deviation of the Gaussian policy.
        """
        self.sensor = glimpse_network(h_g, h_l, g, k, s, c)
        self.rnn = core_network(hidden_size, hidden_size)
        self.locator = location_network(hidden_size, 2, std)
        self.classifier = action_network(hidden_size, num_classes)


    def forward():