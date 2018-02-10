import sys
sys.path.append("..")

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
from modules import retina
from functools import reduce
from torch.autograd import Variable
from utils import img2array, array2img

# params
TEST_GLIMPSE = True
TEST_BOUNDING = True
SAVE = False
plot_dir = '../plots/'
data_dir = '../data/'


def denormalize(T, coords):
    """
    Convert coordinates in the range [-1, 1] to
    coordinates in the range [0, T] where T is
    the size of the image.
    """
    return (0.5 * ((coords + 1.0) * T)).long()


def bounding_box(x, y, size, color='w'):
    x = int(x - (size / 2))
    y = int(y - (size / 2))
    rect = patches.Rectangle(
        (x, y), size, size, linewidth=1, edgecolor=color, fill=False
    )
    return rect


def merge_images(image1, image2):
    """
    Merge two images into one, displayed side by side.

    https://stackoverflow.com/questions/10657383/stitching-photos-together
    """
    (width1, height1) = image1.size
    (width2, height2) = image2.size

    result_width = width1 + width2
    result_height = max(height1, height2)

    result = Image.new('RGB', (result_width, result_height))
    result.paste(im=image1, box=(0, 0))
    result.paste(im=image2, box=(width1, 0))
    return result


def main():

    # load images
    imgs = []
    paths = [data_dir + './lenna.jpg', data_dir + './cat.jpg']
    for i in range(len(paths)):
        img = img2array(paths[i], desired_size=[512, 512], expand=True)
        imgs.append(torch.from_numpy(img))
    imgs = Variable(torch.cat(imgs))
    imgs = imgs.permute(0, 3, 1, 2)

    # loc = torch.Tensor(2, 2).uniform_(-1, 1)
    loc = torch.from_numpy(np.array([[0., 0.], [0., 0.]]))
    loc = Variable(loc)

    ret = retina(g=64, k=3, s=2)
    glimpse = ret.foveate(imgs, loc).data.numpy()

    glimpse = np.reshape(glimpse, [2, 3, 3, 64, 64])
    glimpse = np.transpose(glimpse, [0, 1, 3, 4, 2])

    merged = []
    for i in range(len(glimpse)):
        g = glimpse[i]
        g = list(g)
        g = [array2img(l) for l in g]
        res = reduce(merge_images, list(g))
        merged.append(res)

    merged = [np.asarray(l, dtype='float32')/255.0 for l in merged]

    fig, axs = plt.subplots(nrows=2, ncols=1)
    for i, ax in enumerate(axs.flat):
        axs[i].imshow(merged[i])
        axs[i].get_xaxis().set_visible(False)
        axs[i].get_yaxis().set_visible(False)
    plt.show()
    # plt.savefig(plot_dir + 'glimpses.png', format='png', dpi=100, bbox_inches='tight')

    # plt.imshow(merged[0])
    # plt.axis('off')
    # plt.savefig(plot_dir + 'g1.png', format='png', dpi=100,
    #     bbox_inches='tight')
    # plt.imshow(merged[1])
    # plt.axis('off')
    # plt.savefig(plot_dir + 'g2.png', format='png', dpi=100,
    #     bbox_inches='tight')

    # if TEST_GLIMPSE:

    #     ret = retina(g=64, k=3, s=2)
    #     glimpse = ret.foveate(imgs, loc).data.numpy()
    #     print("Glimpse: {}".format(glimpse.shape))

    #     rows, cols = glimpse.shape[0], glimpse.shape[1]
    #     fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(5, 2))
    #     for i in range(rows):
    #         for j in range(cols):
    #             axs[i, j].imshow(glimpse[i, j, :])
    #             axs[i, j].get_xaxis().set_visible(False)
    #             axs[i, j].get_yaxis().set_visible(False)
    #     if SAVE:
    #         plt.savefig(plot_dir + 'glimpses.png', format='png', dpi=300,
    #                     bbox_inches='tight')

    # if TEST_BOUNDING:

    #     fig, ax = plt.subplots(nrows=1, ncols=2)
    #     coords = denormalize(imgs.shape[1], loc.data)
    #     imgs = imgs.data.numpy()
    #     for i in range(len(imgs)):
    #         ax[i].imshow(imgs[i])
    #         size = 64
    #         for j in range(3):
    #             rect = bounding_box(
    #                 coords[i, 0], coords[i, 1], size
    #             )
    #             ax[i].add_patch(rect)
    #             size = size * 2
    #         ax[i].get_xaxis().set_visible(False)
    #         ax[i].get_yaxis().set_visible(False)
    #     if SAVE:
    #         plt.savefig(plot_dir + 'bbox.png', format='png', dpi=300,
    #                     bbox_inches='tight')

    # plt.show()


if __name__ == '__main__':
    main()
