import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils import *
from modules import spatial_glimpse

def denormalize(T, x, y):
    x_original = int((T/2)*x + (T/2))
    y_original = int((T/2)*y + (T/2))
    return x_original, y_original

def main():

    img_path = './lenna.jpg'
    img = img2array(img_path)

    # sensor = spatial_glimpse(g=64, k=3, s=2)
    # glimpse = sensor.extract(img, [0, 0])

    # N = glimpse.shape[0]
    # plt.figure(figsize=(8, 2))
    # for i in range(N):
    #     ax = plt.subplot(1, N, i+1)
    #     plt.imshow(glimpse[i])
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)
    # plt.savefig('/Users/kevin/Desktop/glimpses.png', format='png', dpi=300)
    # plt.show()

    size = 64
    scale = 2

    fig, ax = plt.subplots(1)
    c1 = denormalize(img.shape[1], -1, -1)
    c2 = denormalize(img.shape[1], 0, 0)
    c3 = denormalize(img.shape[1], 0.5, 0.5)
    c4 = denormalize(img.shape[1], -0.5, -0.5)
    c5 = denormalize(img.shape[1], 1, 1)
    ax.imshow(img)
    # rect1 = patches.Rectangle(c1, 64, 64, linewidth=1, edgecolor='w', fill=False)
    # rect2 = patches.Rectangle(c2, 64, 64, linewidth=1, edgecolor='w', fill=False)
    # rect3 = patches.Rectangle(c3, 64, 64, linewidth=1, edgecolor='w', fill=False)
    # rect4 = patches.Rectangle(c4, 64, 64, linewidth=1, edgecolor='w', fill=False)
    rect5 = patches.Rectangle(c5, 64, 64, linewidth=1, edgecolor='w', fill=False, angle=180)
    # ax.add_patch(rect1)
    # ax.add_patch(rect2)
    # ax.add_patch(rect3)
    # ax.add_patch(rect4)
    ax.add_patch(rect5)
    plt.show()

if __name__ == '__main__':
    main()
