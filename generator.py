# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn

from options import Options


class Generator(nn.Module):
    def __init__(self, opt: Options):
        super(Generator, self).__init__()
        self.opt = opt

        # Build NN graph
        self.linear1 = nn.Linear(opt.z_dim, opt.net_size)
        self.linear2 = nn.Linear(1, opt.net_size, bias=False)
        self.linear3 = nn.Linear(1, opt.net_size, bias=False)
        self.linear4 = nn.Linear(1, opt.net_size, bias=False)

        self.linear5 = nn.Linear(opt.net_size, opt.net_size)
        self.linear6 = nn.Linear(opt.net_size, opt.net_size)
        self.linear7 = nn.Linear(opt.net_size, opt.net_size)
        self.linear8 = nn.Linear(opt.net_size, opt.c_dim)

        # self.linear9 = nn.Linear(self.net_size, self.c_dim)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        # self.softplus = nn.Softplus()

        self.lin_seq = nn.Sequential(self.tanh, self.linear5, self.tanh, self.linear6, self.tanh,
                                     self.linear7, self.tanh, self.linear8, self.sigmoid)

    def forward(self, x, y, r, z):
        U = self.linear1(z) + self.linear2(x) + self.linear3(y) + self.linear4(r)

        result = torch.transpose(self.lin_seq(U), 1, 2)
        x = int(math.sqrt(result.size()[2]))  # is same as `y`
        res = result.view(-1, self.opt.c_dim, x, x)

        return res


def weights_init_g(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        try:
            m.bias.data.fill_(0)
        except:
            pass
