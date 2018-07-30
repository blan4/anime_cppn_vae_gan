# -*- coding: utf-8 -*-

import numpy as np
import torch


def get_coordinates(x_dim, y_dim, scale, batch_size):
    """
    calculates and returns a vector of x and y coordinates, and corresponding radius from the centre of image.
    """
    n_points = x_dim * y_dim

    # То есть координаты точек задаются в диапазоне [-1,1],
    # что позволяет бесконечно увеличивать разрешение и обеспечивать плавный переход цветов
    # creates a list of x_dim values ranging from -1 to 1, then scales them by scale
    x_range = scale * (np.arange(x_dim) - (x_dim - 1) / 2.0) / (x_dim - 1) / 0.5
    y_range = scale * (np.arange(y_dim) - (y_dim - 1) / 2.0) / (y_dim - 1) / 0.5
    x_mat = np.matmul(np.ones((y_dim, 1)), x_range.reshape((1, x_dim)))
    y_mat = np.matmul(y_range.reshape((y_dim, 1)), np.ones((1, x_dim)))
    r_mat = np.sqrt(x_mat * x_mat + y_mat * y_mat)
    x_mat = np.tile(x_mat.flatten(), batch_size).reshape(batch_size, n_points, 1)
    y_mat = np.tile(y_mat.flatten(), batch_size).reshape(batch_size, n_points, 1)
    r_mat = np.tile(r_mat.flatten(), batch_size).reshape(batch_size, n_points, 1)

    return torch.from_numpy(x_mat).float(), torch.from_numpy(y_mat).float(), torch.from_numpy(r_mat).float()
