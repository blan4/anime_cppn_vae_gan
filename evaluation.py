# -*- coding: utf-8 -*-

import torch
from torch import autograd

from coordinates import get_coordinates
from options import Options
from save_images import save_images


class ImageGenerator:
    def __init__(self, opt: Options) -> None:
        self.opt = opt
        self.x, self.y, self.r = get_coordinates(opt.x_dim, opt.y_dim, scale=opt.scale, batch_size=opt.batch_size)

    def generate(self, frame, netG):
        noise = torch.randn(self.opt.batch_size, self.opt.latent_dim)
        if self.opt.use_cuda:
            noise = noise.cuda(0)
        noisev = autograd.Variable(noise, volatile=True)

        ones = torch.ones(self.opt.batch_size, self.opt.x_dim * self.opt.y_dim, 1)
        if self.opt.use_cuda:
            ones = ones.cuda()

        seed = torch.bmm(ones, noisev.unsqueeze(1))

        samples = netG(self.x, self.y, self.r, seed)

        samples = samples.view(-1, self.opt.z_dim, self.opt.x_dim, self.opt.y_dim)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.cpu().data.numpy()

        save_images(samples, './tmp/cppn/samples_{}.jpg'.format(frame))
