import torch.nn as nn

from torch.autograd import Variable
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
                 h_g,
                 h_l,
                 g,
                 k,
                 s,
                 c,
                 hidden_size,
                 num_classes,
                 std,
                 num_glimpses):
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
        - num_glimpses: number of glimpses to take per image,
          i.e. number of BPTT steps.
        """
        super(RecurrentAttention, self).__init__()
        self.num_glimpses = num_glimpses
        self.hidden_size - hidden_size

        self.sensor = glimpse_network(h_g, h_l, g, k, s, c)
        self.rnn = core_network(hidden_size, hidden_size)
        self.locator = location_network(hidden_size, 2, std)
        self.classifier = action_network(hidden_size, num_classes)
        self.baseliner = baseline_network(hidden_size, 1)

        # bookeeping
        self.glimpses = []
        self.locs = []

    def forward(self, x, gd_truth):
        """
        Run the recurrent attention model for `num_glimpses`
        iterations on a minibatch of images `x`.

        Args
        ----
        - x: a 4D Tensor of shape (B, H, W, C). The minibatch
          of images.
        - gd_truth: a 1D tensor of shape (B,). The ground truth labels
          for the minibatch `x`.

        Returns
        -------
        - l_t: a 2D tensor of shape (B, 2). The location vector
          containing the glimpse coordinates [x, y] for the
          current timestep `t`.
        - h_t: a 2D tensor of shape (B, hidden_size). The hidden
          state vector for the current timestep `t`.
        - probas: a 2D tensor of shape (B, num_classes). The output
          probability vector over the classes.
        """
        batch_size = x.shape[0]

        # initialize hidden state and loc vectors
        h_t = torch.zeros(batch_size, self.hidden_size)
        l_t = torch.Tensor(batch_size, 2).uniform_(-1, 1)
        h_t, l_t = Variable(h_t), Variable(l_t)

        # BPTT
        for t in range(self.num_glimpses):
            g_t = self.sensor(x, l_t)
            h_t = self.rnn(g_t, h_t)
            mean, l_t = self.locator(h_t)

            self.glimpses.append(g_t)
            self.locs.append(l_t)

        # generate classification and calculate reward
        probas = self.classifier(h_t)
        predicted = torch.max(probas, 1)[1]
        R = (predicted == gd_truth)

        





