import numpy as np

from PIL import Image


def resize_array(x, size):
    img = array2img(x)
    img = img.resize((size, size))
    x = np.asarray(img, dtype='float32')
    x = np.expand_dims(x, axis=0)
    x /= 255.0
    return x


def img2array(data_path, desired_size=None, expand=False, view=False):
    """
    Util function for loading RGB image into 4D numpy array.

    Returns array of shape (1, H, W, C)
    """
    img = Image.open(data_path)
    img = img.convert('RGB')
    if desired_size:
        img = img.resize((desired_size[1], desired_size[0]))
    if view:
        img.show()
    x = np.asarray(img, dtype='float32')
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
    return Image.fromarray(x.astype('uint8'), 'RGB')
