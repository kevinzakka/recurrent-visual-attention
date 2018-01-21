import torch

from utils import img2array
from torch.autograd import Variable
from model import RecurrentAttention

# params
plot_dir = './plots/'
data_dir = './data/'


def main():

    # load images
    imgs = []
    paths = [data_dir + './lenna.jpg', data_dir + './cat.jpg']
    for i in range(len(paths)):
        img = img2array(paths[i], desired_size=[512, 512], expand=True)
        imgs.append(torch.from_numpy(img))
    imgs = Variable(torch.cat(imgs))

    B, H, W, C = imgs.shape

    l_t_prev = torch.Tensor(B, 2).uniform_(-1, 1)
    l_t_prev = Variable(l_t_prev)
    h_t_prev = Variable(torch.zeros(B, 256))

    ram = RecurrentAttention()
    h_t, l_t = ram(imgs, l_t_prev, h_t_prev)

    print("h_t: {}".format(h_t.shape))
    print("l_t: {}".format(l_t.shape))


if __name__ == '__main__':
    main()
