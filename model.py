import math

import torch
import torch.nn as nn

from modules import baseline_network
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
                 g,
                 k,
                 s,
                 c,
                 h_g,
                 h_l,
                 std,
                 hidden_size,
                 num_classes):
        """
        Initialize the recurrent attention model and its
        different components.

        Args
        ----
        - g: size of the square patches in the glimpses extracted
          by the retina.
        - k: number of patches to extract per glimpse.
        - s: scaling factor that controls the size of successive patches.
        - c: number of channels in each image.
        - h_g: hidden layer size of the fc layer for `phi`.
        - h_l: hidden layer size of the fc layer for `l`.
        - std: standard deviation of the Gaussian policy.
        - hidden_size: hidden size of the rnn.
        - num_classes: number of classes in the dataset.
        - num_glimpses: number of glimpses to take per image,
          i.e. number of BPTT steps.
        """
        super(RecurrentAttention, self).__init__()
        self.std = std
        self.var = self.std ** 2

        self.sensor = glimpse_network(h_g, h_l, g, k, s, c)
        self.rnn = core_network(hidden_size, hidden_size)
        self.locator = location_network(hidden_size, 2, std)
        self.classifier = action_network(hidden_size, num_classes)
        self.baseliner = baseline_network(hidden_size, 1)

    def normal_log_prob(self, v, loc):
        """
        Compute the log of the Gaussian policy parametrized
        by mean `loc` and standard deviation `scale` evaluated
        at `v`.
        """
        p1 = ((v - loc) ** 2) / (-2 * self.var)
        p2 = - math.log(self.std)
        p3 = - math.log(math.sqrt(2 * math.pi))

        return torch.sum((p1 + p2 + p3), dim=1)

    def forward(self, x, l_t_prev, h_t_prev, last=False, train=True):
        """
        Run the recurrent attention model for 1 timestep
        on the minibatch of images `x`.

        Args
        ----
        - x: a 4D Tensor of shape (B, H, W, C). The minibatch
          of images.
        - l_t_prev: a 2D tensor of shape (B, 2). The location vector
          containing the glimpse coordinates [x, y] for the previous
          timestep `t-1`.
        - h_t_prev: a 2D tensor of shape (B, hidden_size). The hidden
          state vector for the previous timestep `t-1`.
        - last: a bool indicating whether this is the last timestep.
          If True, the action network returns an output probability
          vector over the classes and the baseline `b_t` for the
          current timestep `t`. Else, the core network returns the
          hidden state vector for the next timestep `t+1` and the
          location vector for the next timestep `t+1`.

        Returns
        -------
        - h_t: a 2D tensor of shape (B, hidden_size). The hidden
          state vector for the current timestep `t`.
        - mu: a 2D tensor of shape (B, 2). The mean that parametrizes
          the Gaussian policy.
        - l_t: a 2D tensor of shape (B, 2). The location vector
          containing the glimpse coordinates [x, y] for the
          current timestep `t`.
        - b_t: a 2D vector of shape (B, 1). The baseline for the
          current time step `t`.
        - log_probas: a 2D tensor of shape (B, num_classes). The
          output log probability vector over the classes.
        """
        g_t = self.sensor(x, l_t_prev)
        h_t = self.rnn(g_t, h_t_prev)
        mu, l_t = self.locator(h_t)

        log_pi = self.normal_log_prob(l_t, mu)

        if last:
            log_probas = self.classifier(h_t)
            b_t = self.baseliner(h_t).squeeze()
            if train:
                return (h_t, l_t, log_pi, b_t, log_probas)
            return (h_t, mu, log_probas)

        if train:
            return (h_t, l_t, log_pi)
        return (h_t, mu)
