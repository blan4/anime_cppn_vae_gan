# -*- coding: utf-8 -*-
import os
import time

import torch
import torchvision.utils as vutils
from torch import nn
from torch import optim
from torch.autograd import Variable

from coordinates import get_coordinates
from data import load_data
from discriminator import Discriminator, weights_init_d
from generator import Generator, weights_init_g
from options import Options


def train(opt: Options):
    real_label = 1
    fake_label = 0

    netG = Generator(opt)
    netD = Discriminator(opt)
    print(netG)
    print(netD)

    netG.apply(weights_init_g)
    netD.apply(weights_init_d)

    # summary(netD, (opt.c_dim, opt.x_dim, opt.y_dim))

    dataloader = load_data(opt.data_root, opt.x_dim, opt.y_dim, opt.batch_size, opt.workers)

    x, y, r = get_coordinates(x_dim=opt.x_dim, y_dim=opt.y_dim, scale=opt.scale, batch_size=opt.batch_size)

    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

    criterion = nn.BCELoss()
    # criterion = nn.L1Loss()

    noise = torch.FloatTensor(opt.batch_size, opt.z_dim)
    ones = torch.ones(opt.batch_size, opt.x_dim * opt.y_dim, 1)
    input_ = torch.FloatTensor(opt.batch_size, opt.c_dim, opt.x_dim, opt.y_dim)
    label = torch.FloatTensor(opt.batch_size, 1)

    input_ = Variable(input_)
    label = Variable(label)
    noise = Variable(noise)

    if opt.use_cuda:
        netG = netG.cuda()
        netD = netD.cuda()
        x = x.cuda()
        y = y.cuda()
        r = r.cuda()
        ones = ones.cuda()
        criterion = criterion.cuda()
        input_ = input_.cuda()
        label = label.cuda()
        noise = noise.cuda()

    noise.data.normal_()
    fixed_seed = torch.bmm(ones, noise.unsqueeze(1))

    def _update_discriminator(data):
        # for p in netD.parameters():
        #     p.requires_grad = True  # to avoid computation
        netD.zero_grad()
        real_cpu, _ = data
        input_.data.copy_(real_cpu)
        label.data.fill_(real_label-0.1)  # use smooth label for discriminator

        output = netD(input_)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        noise.data.normal_()
        seed = torch.bmm(ones, noise.unsqueeze(1))

        fake = netG(x, y, r, seed)
        label.data.fill_(fake_label)
        output = netD(fake.detach())  # add ".detach()" to avoid backprop through G
        errD_fake = criterion(output, label)
        errD_fake.backward()  # gradients for fake/real will be accumulated
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()  # .step() can be called once the gradients are computed

        return fake, D_G_z1, errD, D_x

    def _update_generator(fake):
        # for p in netD.parameters():
        #     p.requires_grad = False  # to avoid computation
        netG.zero_grad()

        label.data.fill_(real_label)  # fake labels are real for generator cost

        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()  # True if backward through the graph for the second time
        D_G_z2 = output.data.mean()
        optimizerG.step()

        return D_G_z2, errG

    def _save_model(epoch):
        os.makedirs(opt.models_root, exist_ok=True)
        if epoch % 1 == 0:
            torch.save(netG.state_dict(), os.path.join(opt.models_root, "G-cppn-wgan-anime_{}.pth".format(epoch)))
            torch.save(netD.state_dict(), os.path.join(opt.models_root, "D-cppn-wgan-anime_{}.pth".format(epoch)))

    def _log(i, epoch, errD, errG, D_x, D_G_z1, D_G_z2, delta_time):
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f Elapsed %.2f s'
              % (epoch, opt.iterations, i, len(dataloader), errD.data.item(), errG.data.item(), D_x, D_G_z1, D_G_z2,
                 delta_time))

    def _save_images(i, epoch):
        os.makedirs(opt.images_root, exist_ok=True)
        if i % 100 == 0:
            fake = netG(x, y, r, fixed_seed)
            fname = os.path.join(opt.images_root, "fake_samples_{:02}-{:04}.png".format(epoch, i))
            vutils.save_image(fake.data[0:64, :, :, :], fname, nrow=8)

    def _start():
        print("Start training")
        for epoch in range(opt.iterations):
            for i, data in enumerate(dataloader, 0):
                start_iter = time.time()

                fake, D_G_z1, errD, D_x = _update_discriminator(data)
                D_G_z2, errG = _update_generator(fake)

                end_iter = time.time()

                _log(i, epoch, errD, errG, D_x, D_G_z1, D_G_z2, end_iter - start_iter)
                _save_images(i, epoch)
            _save_model(epoch)

    _start()
