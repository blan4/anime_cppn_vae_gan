# -*- coding: utf-8 -*-


class Options:
    def __init__(self) -> None:
        self.x_dim = 96
        self.y_dim = 96
        self.c_dim = 3
        self.z_dim = 32
        self.batch_size = 128
        self.iterations = 100  # How many generator iterations to train for
        self.workers = 4
        self.data_root = "/home/ilya/Data/anime-faces"
        self.models_root = "./tmp/models/"
        self.images_root = "./tmp/images/"
        self.d_labelSmooth = 0
        self.lr = 1e-4
        self.beta1 = 0.5
        self.beta2 = 0.9
        self.critic_iterations = 5  # How many critic iterations per generator iteration
        self.use_cuda = True
        self.gpu = 0
        self.gpus = [self.gpu]
        self.lambda_ = 10  # Gradient penalty lambda hyperparameter
        self.scale = 8

        # discriminator
        self.filters = 64

        # generator
        self.net_size = 64
