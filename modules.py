import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import resize_array
from torch.autograd import Variable
from torch.distributions import Normal


class glimpse_sensor(object):
    """
    Bandwidth-limited sensor that extracts
    a retina-like representation phi around
    location l from image x.

    Extracts k square patches of size g, centered
    at location l. Each successive patch has `s*g`
    size of the previous patch.

    The k patches are finally resized to (g, g) and
    concatenated.

    Args
    ----
    - x: a minibatch of images (4D Tensor) of shape (B, H, W, C).
    - l: a minibatch of (x, y) coordinates in the range [-1, 1]
         with the center corresponding to (0, 0) and the top left
         corner corresponding to (-1, -1). l is a 2D Tensor of shape
         (B, 2).
    - g: height and width of the first extracted patch.
    - k: number of patches to extract per glimpse.
    - s: scaling factor that controls the size of successive patches.

    Returns
    -------
    - phi: foveated glimpse of an image at a given location.
    """

    def __init__(self, g, k, s):
        self.g = g
        self.k = k
        self.s = s

    def __call__(self, x, l):
        phi = []
        size = self.g
        for i in range(self.k):
            # extract the patch
            phi.append(self.extract_single_patch(x, l, size))
            # scale the patch size
            size = int(self.s * size)

        # convert to numpy array to resize
        phi = [p.numpy() for p in phi]

        # resize the patches to squares of size g
        phi = [resize_array(p, self.g) if p.shape[1] != self.g
               else np.expand_dims(p, 1) for p in phi]

        # concatenate into single vector
        phi = torch.from_numpy(np.concatenate(phi, 1))

        return phi

    def denormalize(self, T, coords):
        x_original = torch.mul(coords[:, 0], int(T/2)) + int((T/2))
        y_original = torch.mul(coords[:, 1], int(T/2)) + int((T/2))
        return torch.stack([x_original, y_original])

    def extract_single_patch(self, x, center, size):
        """
        Extract a single patch for each image in the minibatch
        x, centered at the coordinates `center` and of size `size`.

        Args
        ----
        - x: a 4D Tensor of shape (B, H, W, C).
        - center: a 2D Tensor of shape (B, 2).
        - size: a scalar defining the size of the extracted patch.

        Returns
        -------
        - patch: a 4D Tensor of shape (B, size, size, C)
        """

        # compute unnormalized coords of patch center
        coords = self.denormalize(x.size()[1], center)

        # compute equivalent coords in original img
        patch_x = coords[:, 0] - (size // 2)
        patch_y = coords[:, 1] - (size // 2)

        # extract patch (need to vectorize this)
        patch = []
        for i in range(x.size()[0]):
            p = x[i].unsqueeze(0)
            p = p[:, patch_x[i]:patch_x[i]+size, patch_y[i]:patch_y[i]+size, :]
            patch.append(p)
        patch = torch.cat(patch)

        return patch


class glimpse_network(nn.Module):
    """
    A trainable, bandwidth-limited sensor that mimics
    attention by producing a glimpse representation g_t.

    Feeds the output of the glimpse_sensor to a fc layer h_,
    the glimpse location l to a fc layer, and applies a
    ReLU nonlinearity to their concatenation.

    In other words:

        `g_t = relu( fc(l) || fc(phi) )`

    where `||` signifies a concatenation.

    Args
    ----
    - h_g: hidden layer size of the fc layer for phi.
    - h_l: hidden layer size of the fc layer for l.
    - x: a minibatch of images (4D Tensor) of shape (B, H, W, C).
    - l: the location vector containing the glimpse coordinates [x, y].

    Returns
    -------
    - g_t: glimpse representation vector.
    """

    def __init__(self, h_g=128, h_l=128, g=256, k=3, s=2):
        super(glimpse_network, self).__init__()
        self.sensor = glimpse_sensor(g, k, s)
        self.fc1 = nn.Linear(g, h_g)
        self.fc2 = nn.Linear(2, h_l)

    def forward(self, x, l):
        # compute phi
        phi = self.sensor.extract(x, l)

        # feed phi and l to respective fc layers
        phi_out = F.relu(self.fc1(phi))
        l_out = F.relu(self.fc2(l))

        # concatenate and apply nonlinearity
        g_t = F.relu(torch.cat([phi_out, l_out]))

        return g_t


class core_network(nn.Module):
    """
    A RNN which maintains an internal state that summarizes
    the information extracted from the history of past
    observations. It encodes the agent's knowledge of the
    environment through a state vector h that gets updated
    at every time step t.

    Concretely, it takes the glimpse representation g_t as input,
    and combines it with its interal representation h_t_prev at
    the previous time step, to produce the new internal state h_t.

    In other words:

        `h_t = relu( fc(h_t_prev) + fc(g_t) )`

    There is no `LSTM` or `GRU` cell in PyTorch with a ReLU
    nonlinearity so this will fall back to the `tanh` activation.

    Args
    ----
    - g_t: the glimpse representation returned by the glimpse network.
    - h_t_prev: internal representatino at time step (t - 1).

    Returns
    -------
    - h_t: internal representation at time step t.
    """

    def __init__(self, input_size=256, hidden_size=256):
        super(core_network, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, 1)

    def forward(self, g_t, h_t_prev):
        g_t = torch.unsqueeze(g_t, 0)
        _, h_t = self.rnn(g_t, h_t_prev)
        return h_t

    def init_hidden(self, batch_size):
        return (
            Variable(torch.zeros(1, batch_size, self.hidden_size)),
            Variable(torch.zeros(1, batch_size, self.hidden_size))
        )


class action_network(nn.Module):
    """
    Uses the internal state h_t of the core network to
    produce the final output classification.

    Concretely, feeds the hidden state h_t to a fc
    layer and applies a softmax to create a vector
    of output probabilities over the possible classes.

    Args
    ----
    - input_size: input size of the fc layer.
    - num_classes: output size of the fc layer.
    - h_t: the hidden state vector of the core network at
           time step t.
    """

    def __init__(self, input_size, num_classes):
        super(action_network, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return F.softmax(self.fc(x))


class location_network(nn.Module):
    """
    Uses the internal state h_t of the core network to
    produce a 2D vector of means used to parametrize the
    policy for the locations l. The policy itself is a
    two-component Gaussian with a fixed variance.

    Args
    ----
    - input_size: input size of the fc layer.
    - output_size: output size of the fc layer.
    - h_t: the hidden state vector of the core network at
           time step t.

    Returns
    -------
    - mean: a 2D vector of size (B, 2).
    """

    def __init__(self, input_size, output_size=2):
        super(location_network, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        mean = F.tanh(self.fc(h_t))
        return mean


class Policy(nn.Module):
    """
    The policy for the locations l is defined as a two-component
    Gaussian with a fixed variance. The Gaussian is parametrized
    by the output of the location network.

    TODO: use `.detach()` to stop the gradient flow.

    Args
    ----
    - std: fixed std deviation of the location policy.
    - mean: mean of the location policy returned by the location
            network.

    Returns
    -------
    - l_t_next: the location vector for the next time step.
    """

    def __init__(self, std):
        super(Policy, self).__init__()
        self.std = std

    def forward(self, mean):
        l_t_next = Normal(mean, self.std).sample()
        return l_t_next
