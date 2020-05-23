import sys
sys.path.append("..")

import torch

import modules
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

    loc = torch.Tensor([[-1.0, 1.0], [-1.0, 1.0]])
    sensor = modules.GlimpseNetwork(h_g=128, h_l=128, g=64, k=3, s=2, c=3)
    g_t = sensor(imgs, loc)
    assert g_t.shape == (B, 256)

    rnn = modules.CoreNetwork(input_size=256, hidden_size=256)
    h_t = torch.zeros(g_t.shape[0], 256)
    h_t = rnn(g_t, h_t)
    assert h_t.shape == (B, 256)

    classifier = modules.ActionNetwork(256, 10)
    a_t = classifier(h_t)
    assert a_t.shape == (B, 10)

    loc_net = modules.LocationNetwork(256, 2, 0.11)
    mu, l_t = loc_net(h_t)
    assert l_t.shape == (B, 2)

    base = modules.BaselineNetwork(256, 1)
    b_t = base(h_t)
    assert b_t.shape == (B, 1)
