# Adapted from: https://github.com/bingsyslab/360projection/blob/master/equirectangular.py

import numpy as np

def xrotation(th):
    c = np.cos(th)
    s = np.sin(th)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])

def yrotation(th):
    c = np.cos(th)
    s = np.sin(th)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def make_grid_np(theta0, phi0, width, img_shape, fov_h=np.pi/2, fov_v=np.pi/2):
    """
    theta0 is pitch
    phi0 is yaw
    render view at (pitch, yaw) with fov_h by fov_v
    width is the number of horizontal pixels in the view

    Returns a (height, width, 2) matrix containing the
    pixels to sample from (in the source img)
    to project from sphere->perspective.
    """
    m = np.dot(yrotation(phi0), xrotation(theta0))

    (base_height, base_width, _) = img_shape

    height = int(width * np.tan(fov_v / 2) / np.tan(fov_h / 2))

    DI = np.ones((height * width, 3), np.int)
    trans = np.array([[2.*np.tan(fov_h / 2) / float(width), 0., -np.tan(fov_h / 2)],
                      [0., -2.*np.tan(fov_v / 2) / float(height), np.tan(fov_v / 2)]])

    xx, yy = np.meshgrid(np.arange(width), np.arange(height))

    DI[:, 0] = xx.reshape(height * width)
    DI[:, 1] = yy.reshape(height * width)

    v = np.ones((height * width, 3), np.float)

    v[:, :2] = np.dot(DI, trans.T)
    v = np.dot(v, m.T)

    diag = np.sqrt(v[:, 2] ** 2 + v[:, 0] ** 2)
    theta = np.pi / 2 - np.arctan2(v[:, 1], diag)
    phi = np.arctan2(v[:, 0], v[:, 2]) + np.pi

    ey = theta * base_height / np.pi
    ex = phi * base_width / (2 * np.pi)

    ex[ex >= base_width] = base_width - 1
    ey[ey >= base_height] = base_height - 1

    # Need to scale grid coordinates between [-1, +1] for use in grid_sample
    ey_grid = ey.reshape(height, width) / (base_height / 2.) - 1
    ex_grid = ex.reshape(height, width) / (base_width / 2.) - 1
    return np.stack([ex_grid, ey_grid], axis=-1)
