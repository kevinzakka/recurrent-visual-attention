import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils import img2array
from modules import retina
from torch.autograd import Variable

# params
TEST_GLIMPSE = True
TEST_BOUNDING = True
plot_dir = './plots/'
data_dir = './data/'


def denormalize(T, coords):
    """
    Convert coordinate in the range [-1, 1] to
    coordinates in the range [0, T] where T
    is the size of the image.
    """
    x = 0.5 * ((coords[:, 0] + 1.0) * T)
    y = 0.5 * ((coords[:, 1] + 1.0) * T)
    return torch.stack([x, y], dim=1).long()


def bounding_box(x, y, size, color='w'):
    x = int(x - (size / 2))
    y = int(y - (size / 2))
    rect = patches.Rectangle(
        (x, y), size, size, linewidth=1, edgecolor=color, fill=False
    )
    return rect


def main():

    # load images
    imgs = []
    paths = [data_dir + './lenna.jpg', data_dir + './cat.jpg']
    for i in range(len(paths)):
        img = img2array(paths[i], desired_size=[512, 512], expand=True)
        imgs.append(torch.from_numpy(img))
    imgs = Variable(torch.cat(imgs))

    loc = Variable(torch.Tensor([[-1., -1.], [-1., -1.]]))

    if TEST_GLIMPSE:

        ret = retina(g=64, k=3, s=2)
        glimpse = ret(imgs, loc).data.numpy()
        print("Glimpse: {}".format(glimpse.shape))

        rows, cols = glimpse.shape[0], glimpse.shape[1]
        fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(5, 2))
        for i in range(rows):
            for j in range(cols):
                axs[i, j].imshow(glimpse[i, j, :])
                axs[i, j].get_xaxis().set_visible(False)
                axs[i, j].get_yaxis().set_visible(False)
        # plt.savefig(plot_dir + 'glimpses.png', format='png', dpi=300,
        #             bbox_inches='tight')
        # plt.show()

    if TEST_BOUNDING:

        fig, ax = plt.subplots(nrows=1, ncols=2)
        coords = denormalize(imgs.shape[1], loc.data)
        imgs = imgs.data.numpy()
        for i in range(len(imgs)):
            ax[i].imshow(imgs[i])
            size = 64
            for j in range(3):
                rect = bounding_box(
                    coords[i, 0], coords[i, 1], size, color='r'
                )
                ax[i].add_patch(rect)
                size = size * 2
            ax[i].get_xaxis().set_visible(False)
            ax[i].get_yaxis().set_visible(False)
        # plt.savefig(plot_dir + 'bbox.png', format='png', dpi=300,
        #             bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    main()
