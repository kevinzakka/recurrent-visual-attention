import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import resize_array


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
    - x: an 3D tensor of shape (H, W, C).
    - l: (x, y) coordinates in the range [-1, 1] with the
         center corresponding to (0, 0) and the top left
         corner corresponding to (-1, -1).
    - g: height and width of the first extracted patch.
    - k: number of patches to extract per glimpse.
    - s: scaling factor that controls the size of successive patches.

    Returns
    -------
    - phi: foveated glimpse of an image at a given location.
    """

    def __init__(self, g=256, k=3, s=2):
        self.g = g
        self.k = k
        self.s = s

    def extract(self, x, l):
        phi = []
        size = self.g
        for i in range(self.k):
            # extract the patch
            phi.append(self._extract_single_patch(x, l, size))
            # scale the patch size
            size = int(self.s * size)

        # convert to numpy array to resize
        phi = [p.numpy() for p in phi]

        # resize the patches to squares of size g
        phi = [resize_array(p, self.g) if p.shape[0] != self.g
               else np.expand_dims(p, axis=0) for p in phi]

        # concatenate into single vector
        phi = torch.from_numpy(np.concatenate(phi))

        return phi

    def _denormalize(self, T, x, y):
        x_original = int((T/2)*x + (T/2))
        y_original = int((T/2)*y + (T/2))
        return x_original, y_original

    def _extract_single_patch(self, x, center, size):
        # compute unnormalized coords of patch center
        height, width = self._denormalize(x.size()[1], *center)

        # compute equivalent coords in original img
        patch_x = int(height - (size / 2))
        patch_y = int(width - (size / 2))

        # extract patch
        patch = x[patch_x:patch_x+size, patch_y:patch_y+size, :]

        return patch


class glimpse_network(nn.Module):
    """
    A trainable, bandwidth-limited sensor that mimics
    attention by producing a glimpse representation g_t.

    Feeds the output of the glimpse_sensor to a fc layer h_,
    the glimpse location l to a fc layer, and applies a
    ReLU nonlinearity to their sum.

    In other words:

        `g_t = relu( fc(l) + fc(phi) )`

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

    def __init__(self, h_g, h_l, g, k, s):
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

        # sum and apply nonlinearity
        g_t = F.relu(phi_out + l_out)

        return g_t
