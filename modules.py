import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.distributions import Normal

import numpy as np

from utils import resize_array


class retina(object):
    """
    A retina that extracts a foveated glimpse `phi`
    around location `l` from an image `x`. It encodes
    the region around `l` at a high-resolution but uses
    a progressively lower resolution for pixels further
    from `l`, resulting in a compressed representation
    of the original image `x`.

    Args
    ----
    - x: a 4D Tensor of shape (B, H, W, C). The minibatch
      of images.
    - l: a 2D Tensor of shape (B, 2). Contains normalized
      coordinates in the range [-1, 1].
    - g: size of the first square patch.
    - k: number of patches to extract in the glimpse.
    - s: scaling factor that controls the size of
      successive patches.

    Returns
    -------
    - phi: a 5D tensor of shape (B, k, g, g, C). The
      foveated glimpse of the image.
    """
    def __init__(self, g, k, s):
        self.g = g
        self.k = k
        self.s = s

    def foveate(self, x, l):
        """
        Extract `k` square patches of size `g`, centered
        at location `l`. The initial patch is a square of
        size `g`, and each subsequent patch is a square
        whose side is `s` times the size of the previous
        patch.

        The `k` patches are finally resized to (g, g) and
        concatenated into a tensor of shape (B, k, g, g, C).
        """
        phi = []
        size = self.g

        # extract k patches of decreasing resolution
        for i in range(self.k):
            phi.append(self.extract_patch(x, l, size))
            size = int(self.s * size)

        # resize the patches to squares of size g
        phi = [p.numpy() for p in phi]
        phi = [
            resize_array(p, self.g) if p.shape[1] != self.g
            else np.expand_dims(p, 1) for p in phi
        ]

        # concatenate into a single tensor
        phi = Variable(torch.from_numpy(np.concatenate(phi, 1)))

        return phi

    def extract_patch(self, x, l, size):
        """
        Extract a single patch for each image in the
        minibatch `x`.

        Args
        ----
        - x: a 4D Tensor of shape (B, H, W, C). The minibatch
          of images.
        - l: a 2D Tensor of shape (B, 2).
        - size: a scalar defining the size of the extracted patch.

        Returns
        -------
        - patch: a 4D Tensor of shape (B, size, size, C)
        """
        # denormalize coords of patch center
        coords = self.denormalize(x.shape[1], l)
        coords = coords.data.numpy()

        # compute top left corner of patch
        patch_x = coords[:, 0] - (size // 2)
        patch_y = coords[:, 1] - (size // 2)

        # loop through minibatch
        patch = []
        for i in range(len(x)):
            # grab img
            p = x[i].unsqueeze(0).data
            T = p.shape[1]

            # compute slice indices
            from_x, to_x = patch_x[i], patch_x[i] + size
            from_y, to_y = patch_y[i], patch_y[i] + size

            # pad the img if patch exceeds its bounds
            if self.exceeds(from_x, to_x, from_y, to_y, T):
                pad_dims = [
                    (0, 0), (size//2+1, size//2+1),
                    (size//2+1, size//2+1), (0, 0),
                ]
                p = p.numpy()
                p = np.pad(p, pad_dims, mode='constant')
                p = torch.from_numpy(p)

                # add correction factor
                from_x += (size//2+1)
                from_y += (size//2+1)
                to_x += (size//2+1)
                to_y += (size//2+1)

            # extract the patch
            patch.append(p[:, from_y:to_y, from_x:to_x, :])

        patch = torch.cat(patch)

        return patch

    def denormalize(self, T, coords):
        """
        Convert coordinates in the range [-1, 1] to
        coordinates in the range [0, T] where `T` is
        the size of the image.
        """
        return (0.5 * ((coords + 1.0) * T)).long()

    def exceeds(self, from_x, to_x, from_y, to_y, T):
        """
        Check whether the extracted patch will exceed
        the boundaries of the image of size `T`.
        """
        if (
            (from_x < 0) or (from_y < 0) or (to_x > T) or (to_y > T)
        ):
            return True
        return False


class glimpse_network(nn.Module):
    """
    A network that combines the "what" and the "where"
    into a glimpse feature vector g_t.

    - "what": glimpse extracted from the retina.
    - "where": location tuple where glimpse was extracted.

    Concretely, feeds the output of the retina `phi` to
    a fc layer and the glimpse location vector `l_t_prev`
    to a fc layer. Finally, these outputs are fed each
    through a fc layer and their sum is rectified.

    In other words:

        `g_t = relu( fc( fc(l) ) + fc( fc(phi) ) )`

    Args
    ----
    - h_g: hidden layer size of the fc layer for `phi`.
    - h_l: hidden layer size of the fc layer for `l`.
    - g: size of the square patches in the glimpses extracted
      by the retina.
    - k: number of patches to extract per glimpse.
    - s: scaling factor that controls the size of successive patches.
    - c: number of channels in each image.
    - x: a 4D Tensor of shape (B, H, W, C). The minibatch
      of images.
    - l_t_prev: a 2D tensor of shape (B, 2). Contains the glimpse
      coordinates [x, y] for the previous timestep `t-1`.

    Returns
    -------
    - g_t: a 2D tensor of shape (B, hidden_size). The glimpse
      representation returned by the glimpse network for the
      current timestep `t`.
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

        # what-where layer
        self.fc3 = nn.Linear(h_g, h_g+h_l)

    def forward(self, x, l_t_prev):
        # B, H, W, C
        x = x.permute(0, 2, 3, 1)

        # generate glimpse phi from image x
        phi = self.retina.foveate(x, l_t_prev)

        # flatten both
        phi = phi.view(phi.size(0), -1)
        l_t_prev = l_t_prev.view(l_t_prev.size(0), -1)

        # feed phi and l to respective fc layers
        phi_out = F.relu(self.fc1(phi))
        l_out = F.relu(self.fc2(l_t_prev))

        what = self.fc3(phi_out)
        where = self.fc3(l_out)

        # feed to fc layer
        g_t = F.relu(what + where)

        return g_t


class core_network(nn.Module):
    """
    An RNN which maintains an internal state that integrates
    information extracted from the history of past observations.
    It encodes the agent's knowledge of the environment through
    a state vector `h_t` that gets updated at every time step `t`.

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
    - g_t: a 2D tensor of shape (B, hidden_size). The glimpse
      representation returned by the glimpse network for the
      current timestep `t`.
    - h_t_prev: a 2D tensor of shape (B, hidden_size). The
      hidden state vector for the previous timestep `t-1`.

    Returns
    -------
    - h_t: a 2D tensor of shape (B, hidden_size). The hidden
      state vector for the current timestep `t`.
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
    - h_t: the hidden state vector of the core network for
      the current time step `t`.

    Returns
    -------
    - a_t: output probability vector over the classes.
    """
    def __init__(self, input_size, output_size):
        super(action_network, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        a_t = F.log_softmax(self.fc(h_t), dim=1)
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

    Args
    ----
    - input_size: input size of the fc layer.
    - output_size: output size of the fc layer.
    - std: standard deviation of the normal distribution.
    - h_t: the hidden state vector of the core network for
      the current time step `t`.

    Returns
    -------
    - mu: a 2D vector of shape (B, 2).
    - l_t: a 2D vector of shape (B, 2).
    """
    def __init__(self, input_size, output_size, std):
        super(location_network, self).__init__()
        self.std = std
        self.fc = nn.Linear(input_size, output_size)

    def gaussian(self, mu):
        B = mu.shape[0]

        # initialize standard normal
        zeros = torch.zeros(B, 2)
        ones = torch.ones(B, 2)
        gauss = Normal(zeros, ones)

        # sample from standard normal
        x = Variable(gauss.sample())

        # shift to desired mean and std
        x = mu + self.std*x

        return x

    def forward(self, h_t):
        mu = F.tanh(self.fc(h_t))
        l_t = F.tanh(self.gaussian(mu)).detach()
        return mu, l_t


class baseline_network(nn.Module):
    """
    Regresses the baseline in the reward function
    to reduce the variance of the gradient update.

    Args
    ----
    - input_size: input size of the fc layer.
    - output_size: output size of the fc layer.
    - h_t: the hidden state vector of the core network
      for the current time step `t`.

    Returns
    -------
    - b_t: a 2D vector of shape (B, 1). The baseline
      for the current time step `t`.
    """
    def __init__(self, input_size, output_size):
        super(baseline_network, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        b_t = F.relu(self.fc(h_t))
        return b_t
