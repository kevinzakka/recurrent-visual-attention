import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import resize_array
from torch.autograd import Variable
from torch.distributions import Normal


class retina(object):
    """
    A retina-like representation that extracts
    foveated glimpses `phi` around location `l`
    from image `x`.

    Extracts `k` square patches of size `g`, centered
    at location `l`. Each subsequent set of patches
    has `s*g` size of the previous `k` patches.

    The `k` patches are finally resized to (g, g) and
    concatenated.

    Args
    ----
    - x: a minibatch of images (4D Tensor) of shape (B, H, W, C).
    - l: a minibatch of (x, y) coordinates in the range [-1, 1]
      with the center corresponding to (0, 0) and the top left
      corner corresponding to (-1, -1). l is a 2D Tensor of
      shape (B, 2).
    - g: height and width of the first extracted patch.
    - k: number of patches to extract per glimpse.
    - s: scaling factor that controls the size of successive patches.

    Returns
    -------
    - phi: foveated glimpse of the image.
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
        phi = Variable(torch.from_numpy(np.concatenate(phi, 1)))

        return phi

    def denormalize(self, T, coords):
        """
        Convert coordinates in the range [-1, 1] to
        coordinates in the range [0, T] where `T` is
        the size of the image.
        """
        return (0.5 * ((coords + 1.0) * T)).long()

    def out_of_bounds(self, from_x, to_x, from_y, to_y, size):
        """
        Check whether the extracted patch will exceed
        the boundaries of the image.
        """
        if (
            (from_x < 0) or (from_y < 0) or (to_x > size) or (to_y.data > size)
        ):
            return True
        return False

    def extract_single_patch(self, x, center, size):
        """
        Extract a single patch for each image in the minibatch
        `x`, centered at the coordinates `center` and of size `size`.

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
        coords = self.denormalize(x.shape[1], center)
        coords = coords.data.numpy()

        # find upper left corner of patch given center
        patch_x = coords[:, 0] - (size // 2)
        patch_y = coords[:, 1] - (size // 2)

        # extract patch
        patch = []
        for i in range(len(x)):

            # grab image
            p = x[i].unsqueeze(0).data

            # compute slice indices
            from_x, to_x = patch_x[i], patch_x[i] + size
            from_y, to_y = patch_y[i], patch_y[i] + size

            if self.out_of_bounds(from_x, to_x, from_y, to_y, p.shape[1]):
                pad_dims = [
                    (0, 0), (size//2+1, size//2+1),
                    (size//2+1, size//2+1), (0, 0),
                ]
                p = p.numpy()
                p = np.pad(p, pad_dims, mode='constant')
                p = torch.from_numpy(p)

                # since size increased, correct coordinates
                from_x += (size//2+1)
                from_y += (size//2+1)
                to_x += (size//2+1)
                to_y += (size//2+1)

            # slice the patch and append
            patch.append(p[:, from_y:to_y, from_x:to_x, :])

        patch = torch.cat(patch)

        return patch


class glimpse_network(nn.Module):
    """
    A trainable, bandwidth-limited sensor that mimics
    attention by producing a glimpse representation `g_t`.

    Feeds the output of the retina `phi` to a fc layer,
    the glimpse location vector `l` to a fc layer, and
    applies a ReLU nonlinearity to their concatenation.

    In other words:

        `g_t = relu( fc(l) || fc(phi) )`

    where `||` signifies a concatenation.

    Args
    ----
    - h_g: hidden layer size of the fc layer for `phi`.
    - h_l: hidden layer size of the fc layer for `l`.
    - g: size of the square patches in the glimpses extracted
      by the retina.
    - k: number of patches to extract per glimpse.
    - s: scaling factor that controls the size of successive patches.
    - c: number of channels in each image.
    - x: a minibatch of images (4D Tensor) of shape (B, H, W, C).
    - l: the location vector containing the glimpse coordinates [x, y]. 2D
      tensor of shape (B, 2).

    Returns
    -------
    - g_t: glimpse representation vector.
    """

    def __init__(self, h_g, h_l, g, k, s, c):
        super(glimpse_network, self).__init__()
        self.retina = retina(g, k, s)

        # glimpse layer
        D_in = k*g*g*c
        self.fc1 = nn.Linear(D_in, h_g)

        # location layer
        D_in = 2
        self.fc2 = nn.Linear(D_in, h_l)

    def forward(self, x, l):
        # generate glimpse phi from image x
        phi = self.retina(x, l)

        # flatten both for fully-connected
        phi = phi.view(phi.size(0), -1)
        l = l.view(l.size(0), -1)

        # feed phi and l tgo respective fc layers
        phi_out = F.relu(self.fc1(phi))
        l_out = F.relu(self.fc2(l))

        # concatenate and apply nonlinearity
        g_t = F.relu(torch.cat([phi_out, l_out], dim=1))

        return g_t


class core_network(nn.Module):
    """
    A RNN which maintains an internal state that integrates
    information extracted from the history of past observations.
    It encodes the agent's knowledge of the environment through
    a state vector `h` that gets updated at every time step `t`.

    Concretely, it takes the glimpse representation `g_t` as input,
    and combines it with its internal state `h_t_prev` at the previous
    time step, to produce the new internal state `h_t` at the current
    time step.

    In other words:

        `h_t = relu( fc(h_t_prev) + fc(g_t) )`

    Args
    ----
    - input_size: input size of the rnn.
    - hidden_size: hidden size of the rnn.
    - g_t: the glimpse representation returned by the glimpse network.
    - h_t_prev: hidden state vector at time step `t-1`.

    Returns
    -------
    - h_t: hidden state vector at time step `t`.
    """

    def __init__(self, input_size, hidden_size):
        super(core_network, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)

    def forward(self, g_t, h_t_prev):
        h1 = self.i2h(g_t)
        h2 = self.h2h(h_t_prev)
        h_t = F.relu(h1 + h2)
        return h_t


class action_network(nn.Module):
    """
    Uses the internal state `h_t` of the core network to
    produce the final output classification.

    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a softmax to create a vector of
    output probabilities over the possible classes.

    Hence, the environment action `a_t` is drawn from a
    distribution conditioned on an affine transformation
    of the hidden state vector `h_t`, or in other words,
    the action network is simply a linear softmax classifier.

    Args
    ----
    - input_size: input size of the fc layer.
    - output_size: output size of the fc layer.
    - h_t: the hidden state vector of the core network at
      time step `t`.

    Returns
    -------
    - a_t: output probability vector over the classes.
    """

    def __init__(self, input_size, output_size):
        super(action_network, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        a_t = F.softmax(self.fc(h_t), dim=1)
        return a_t


class location_network(nn.Module):
    """
    Uses the internal state `h_t` of the core network to
    produce the location coordinates `l_t` for the next
    time step.

    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a tanh to clamp the output beween
    [-1, 1]. This produces a 2D vector of means used to
    parametrize a two-component Gaussian with a fixed
    variance from which the location coordinates `l_t`
    for the next time step are sampled.

    Hence, the location `l_t` is chosen stochastically
    from a distribution conditioned on an affine
    transformation of the hidden state vector `h_t`.

    Todo: add arg for training vs testing.

    Args
    ----
    - input_size: input size of the fc layer.
    - output_size: output size of the fc layer.
    - std: standard deviation of the normal distribution.
    - h_t: the hidden state vector of the core network at
      time step `t`.

    Returns
    -------
    - mean: a 2D vector of shape (B, 2).
    - l_t: a 2D vector of shape (B, 2).
    """

    def __init__(self, input_size, output_size, std):
        super(location_network, self).__init__()
        self.std = std
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        mean = F.tanh(self.fc(h_t))
        mean = mean.detach()
        l_t = F.tanh(Normal(mean, self.std).sample())
        l_t = l_t.detach()
        return mean, l_t
