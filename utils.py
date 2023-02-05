import os, errno
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import torch
import torch.nn.functional as F

from PIL import Image


def denormalize(T, coords):
    """Convert coordinates in the range [-1, 1] to
    coordinates in the range [0, T] where `T` is
    the size of the image.
    """
    return (0.5 * ((coords + 1.0) * T)).long()


def bounding_box(x, y, size, color="w"):
    x = int(x - (size / 2))
    y = int(y - (size / 2))
    rect = patches.Rectangle(
        (x, y), size, size, linewidth=1, edgecolor=color, fill=False
    )
    return rect


# https://github.com/pytorch/examples/blob/master/imagenet/main.py
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def resize_array(x, size):
    # 3D and 4D tensors allowed only
    assert x.ndim in [3, 4], "Only 3D and 4D Tensors allowed!"

    # 4D Tensor
    if x.ndim == 4:
        res = []
        for i in range(x.shape[0]):
            img = array2img(x[i])
            img = img.resize((size, size))
            img = np.asarray(img, dtype="float32")
            img = np.expand_dims(img, axis=0)
            img /= 255.0
            res.append(img)
        res = np.concatenate(res)
        res = np.expand_dims(res, axis=1)
        return res

    # 3D Tensor
    img = array2img(x)
    img = img.resize((size, size))
    res = np.asarray(img, dtype="float32")
    res = np.expand_dims(res, axis=0)
    res /= 255.0
    return res


def img2array(data_path, desired_size=None, expand=False, view=False):
    """
    Util function for loading RGB image into a numpy array.

    Returns array of shape (1, H, W, C).
    """
    img = Image.open(data_path)
    img = img.convert("RGB")
    if desired_size:
        img = img.resize((desired_size[1], desired_size[0]))
    if view:
        img.show()
    x = np.asarray(img, dtype="float32")
    if expand:
        x = np.expand_dims(x, axis=0)
    x /= 255.0
    return x


def array2img(x):
    """
    Util function for converting anumpy array to a PIL img.

    Returns PIL RGB img.
    """
    x = np.asarray(x)
    x = x + max(-np.min(x), 0)
    x_max = np.max(x)
    if x_max != 0:
        x /= x_max
    x *= 255
    return Image.fromarray(x.astype("uint8"), "RGB")


def plot_images(images, gd_truth):

    images = images.squeeze()
    assert len(images) == len(gd_truth) == 9

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    for i, ax in enumerate(axes.flat):
        # plot the image
        ax.imshow(images[i], cmap="Greys_r")

        xlabel = "{}".format(gd_truth[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


def prepare_dirs(config):
    for path in [config.data_dir, config.ckpt_dir, config.logs_dir]:
        if not os.path.exists(path):
            os.makedirs(path)


def save_config(config):
    model_name = "ram_{}_{}x{}_{}_{}_{}_{}_{}_{}".format(
        config.num_glimpses,
        config.patch_size,
        config.patch_size,
        config.glimpse_scale,
        config.num_patches,
        config.num_bits_g_t,
        config.num_bits_h_t,
        config.num_bits_phi,
        config.num_bits_lt
    )
    filename = model_name + "_params.json"
    param_path = os.path.join(config.ckpt_dir, filename)

    print("[*] Model Checkpoint Dir: {}".format(config.ckpt_dir))
    print("[*] Param Path: {}".format(param_path))

    with open(param_path, "w") as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)

def quantize_tensor(t, b, min_t = None, max_t = None):
    """Quantize a tensor.

    Args:
        t: tensor
        b: number of bits available for quantizing

    Returns:
        A quantized tensor in floating points between [min{t}, max{t}].
    """
    if min_t == None:   min_t = torch.min(t)
    if max_t == None:   max_t = torch.max(t)

    return (torch.round( ( (t - min_t) / (max_t-min_t) ) * (2**b - 1) )) * (max_t-min_t) / (2**b-1)

global_phi_max = 1.0
global_phi_min = 0.0

def silent_file_remove(filename):
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred

def closest_result(csv_file: str, p_t: torch.Tensor, h_t: torch.Tensor, l_t: torch.Tensor) -> np.ndarray:
    # Load the data from the csv file into a pandas dataframe
    df = pd.read_csv(csv_file)
    
    # Convert the dataframe to a numpy array
    data = df.to_numpy()
    
    # Convert the torch tensors to numpy arrays
    p_t_np = p_t.detach().numpy()
    h_t_np = h_t.detach().numpy()
    l_t_np = l_t.detach().numpy()
    
    # Extract the first three vectors for each row into separate arrays
    h_arr = data[:, :64]
    l_arr = data[:, 64:66]
    phi_arr = data[:, 66:114]
    
    # Calculate the difference between the target values and the values in each row for each vector
    p_diff = np.sum((p_t_np[:, np.newaxis, :] - phi_arr[np.newaxis, :, :])**2, axis=-1)
    h_diff = np.sum((h_t_np[:, np.newaxis, :] - h_arr[np.newaxis, :, :])**2, axis=-1)
    l_diff = np.sum((l_t_np[:, np.newaxis, :] - l_arr[np.newaxis, :, :])**2, axis=-1)
    
    # Calculate the total difference for each row for each vector
    diff = p_diff + h_diff + l_diff
    
    # Find the index of the row with the minimum difference for each vector
    min_index = np.argmin(diff, axis=-1)
    
     # Create a matrix with the 3 outputs of the closest row (`ht1`, `lt1`, and `a` if present)
    closest_outputs = data[min_index, 114:]
    
    return closest_outputs



class RetinaBasedMemoryInference:
    """Equal to retina, but there isn't the denormalize.

    Extracts a foveated glimpse `phi` around location `l`
    from an image `x`.

    Concretely, encodes the region around `l` at a
    high-resolution but uses a progressively lower
    resolution for pixels further from `l`, resulting
    in a compressed representation of the original
    image `x`.

    Args:
        x: a 4D Tensor of shape (B, C, H, W). The minibatch
            of images. B=batches, C=channels(=1), H=height, W=width
        l: a 2D Tensor of shape (B, 2). Contains normalized
            coordinates in the range [-1, 1].
        g: size of the first square patch.
        k: number of patches to extract in the glimpse.
        s: scaling factor that controls the size of
            successive patches.

    Returns:
        phi: a 5D tensor of shape (B, k, g, g, C). The
            foveated glimpse of the image.
    """

    def __init__(self, g, k, s):
        self.g = g
        self.k = k
        self.s = s

    def foveate(self, x, l):
        """Extract `k` square patches of size `g`, centered
        at location `l`. The initial patch is a square of
        size `g`, and each subsequent patch is a square
        whose side is `s` times the size of the previous
        patch.

        The `k` patches are finally resized to (g, g) and
        concatenated into a tensor of shape (B, k, g, g, C).
        """
        phi = []    # phi is a list k batches of patch tensors 
        size = self.g

        # extract k patches of increasing size
        for i in range(self.k):
            phi.append(self.extract_patch(x, l, size))
            size = int(self.s * size)

        # resize the patches to squares of size g
        for i in range(1, len(phi)):
            k = phi[i].shape[-1] // self.g
            phi[i] = F.avg_pool2d(phi[i], k)

        # concatenate into a single tensor and flatten
        phi = torch.cat(phi, 1)
        phi = phi.view(phi.shape[0], -1)    #phi becomes a tensor of B batches with size*size elements

        return phi

    def extract_patch(self, x, l, size):
        """Extract a single patch for each image in `x`.

        Args:
        x: a 4D Tensor of shape (B, H, W, C). The minibatch
            of images.
        l: a 2D Tensor of shape (B, 2).
        size: a scalar defining the size of the extracted patch.

        Returns:
            patch: a 4D Tensor of shape (B, size, size, C)
        """
        B, C, H, W = x.shape

        start = l  #TODO provare a renderlo simmetrico e vediamo se funziona....
        end = start + size

        # pad with zeros
        x = F.pad(x, (size // 2, size // 2, size // 2, size // 2))

        # loop through mini-batch and extract patches
        patch = []  # list of B patch tensors SIXExSIZE pixels
        for i in range(B):
            patch.append(x[i, :, start[i, 1] : end[i, 1], start[i, 0] : end[i, 0]]) # It takes batch i, all the channels ([:]), and then it takes the height and the width starting from the start pixel to the end pixel. It extracts B batches of patch_size x patch_size pixels in this way
        return torch.stack(patch)
