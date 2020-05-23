import sys
sys.path.append("..")

import torch

import model
import utils


if __name__ == "__main__":
    # paths
    plot_dir = "../plots/"
    data_dir = "../data/"

    # load images
    imgs = []
    paths = [data_dir + "./lenna.jpg", data_dir + "./cat.jpg"]
    for i in range(len(paths)):
        img = utils.img2array(paths[i], desired_size=[512, 512], expand=True)
        imgs.append(torch.from_numpy(img))
    imgs = torch.cat(imgs).permute((0, 3, 1, 2))

    B, C, H, W = imgs.shape
    l_t_prev = torch.FloatTensor(B, 2).uniform_(-1, 1)
    h_t_prev = torch.zeros(B, 256)

    ram = model.RecurrentAttention(64, 3, 2, C, 128, 128, 0.11, 256, 10)
    h_t, l_t, _, _ = ram(imgs, l_t_prev, h_t_prev)

    assert h_t.shape == (B, 256)
    assert l_t.shape == (B, 2)
