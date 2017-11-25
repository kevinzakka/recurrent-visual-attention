import numpy as np
import torch
import torch.nn as nn

from utils import img2array
from modules import glimpse_network


data_dir = './data/'
batch_size = 2

imgs = []
paths = [data_dir + './lenna.jpg', data_dir + './cat.jpg']
for i in range(len(paths)):
    img = img2array(paths[i], desired_size=[512, 512], expand=True)
    imgs.append(torch.from_numpy(img))
imgs = torch.cat(imgs)

# initialize a uniform [-1, 1] location vector
loc = torch.Tensor(batch_size, 2).uniform_(-1, 1)
print("[-1, 1] loc: {}".format(loc))

# get first glimpse
glimpse = glimpse_network()
initial_glimpse = glimpse(imgs, loc)
