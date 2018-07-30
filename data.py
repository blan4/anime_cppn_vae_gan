# -*- coding: utf-8 -*-

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def load_data(data_root, x_dim, y_dim, batch_size, workers):
    # https://discuss.pytorch.org/t/questions-about-dataloader-and-dataset/806/4
    # from PIL import ImageFile
    # ImageFile.LOAD_TRUNCATED_IMAGES = True

    dataset = dset.ImageFolder(
        root=data_root,
        transform=transforms.Compose([
            transforms.Resize((x_dim, y_dim)),
            # transforms.CenterCrop(opt.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # bring images to (-1,1)
        ])
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)

    return dataloader
