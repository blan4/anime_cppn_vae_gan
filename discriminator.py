# -*- coding: utf-8 -*-

import torch.nn as nn

from options import Options


class Discriminator(nn.Module):
    def __init__(self, opt: Options):
        super(Discriminator, self).__init__()
        self.opt = opt

        self.main = nn.Sequential(
            # 3x96x96
            nn.Conv2d(opt.c_dim, opt.filters, kernel_size=5, stride=3, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # 64x32x32
            nn.Conv2d(opt.filters, opt.filters * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(opt.filters * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # 128x16x16
            nn.Conv2d(opt.filters * 2, opt.filters * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(opt.filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 256x8x8
            nn.Conv2d(opt.filters * 4, opt.filters * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(opt.filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 512x4x4
            nn.Conv2d(opt.filters * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # 512x1x1
            nn.Sigmoid()
        )

    def forward(self, input_):
        return self.main(input_).view(-1, 1)


def weights_init_d(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
