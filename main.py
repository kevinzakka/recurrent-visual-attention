import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils import img2array
from modules import glimpse_sensor


TEST_GLIMPSE = True
TEST_BOUNDING = False
plot_dir = './plots/'
data_dir = './data/'


def denormalize(T, x, y):
    x_original = int((T/2)*x + (T/2))
    y_original = int((T/2)*y + (T/2))
    return x_original, y_original


def bounding_box(x, y, size):
    x = int(x - (size / 2))
    y = int(y - (size / 2))
    rect = patches.Rectangle((x, y), size, size, linewidth=1,
                             edgecolor='w', fill=False)
    return rect


def main():

    imgs = []
    paths = [data_dir + './lenna.jpg', data_dir + './cat.jpg']
    for i in range(len(paths)):
        img = img2array(paths[i], desired_size=[512, 512], expand=True)
        imgs.append(torch.from_numpy(img))
    imgs = torch.cat(imgs)

    loc = torch.LongTensor([[0, 0], [0, 0]])

    if TEST_GLIMPSE:

        sensor = glimpse_sensor(g=64, k=3, s=2)
        glimpse = sensor.extract(imgs, loc).numpy()

        rows, cols = glimpse.shape[0], glimpse.shape[1]
        fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(8, 2))
        for i in range(rows):
            for j in range(cols):
                axs[i, j].imshow(glimpse[i, j, :])
                axs[i, j].get_xaxis().set_visible(False)
                axs[i, j].get_yaxis().set_visible(False)
        plt.savefig(plot_dir + 'glimpses.png', format='png', dpi=300,
                    bbox_inches='tight')
        plt.show()

    if TEST_BOUNDING:

        fig, ax = plt.subplots(1)
        x, y = denormalize(img.shape[1], 0, 0)
        ax.imshow(img)
        size = 64
        for i in range(3):
            rect = bounding_box(x, y, size)
            ax.add_patch(rect)
            size = size * 2
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.savefig(plot_dir + 'bbox.png', format='png', dpi=300,
                    bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    main()
