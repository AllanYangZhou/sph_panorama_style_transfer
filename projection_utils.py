# Adapted from: https://github.com/bingsyslab/360projection/

import numpy as np

def xrotation(th):
    c = np.cos(th)
    s = np.sin(th)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])

def yrotation(th):
    c = np.cos(th)
    s = np.sin(th)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def make_proj_grid(theta0, phi0, width, fov_h=np.pi/2, fov_v=np.pi/2):
    """
    theta0 is pitch
    phi0 is yaw
    width is the number of horizontal pixels in the view

    Returns a (height, width, 2) matrix which you can use
    to project from sphere->perspective.
    """
    m = np.dot(yrotation(phi0), xrotation(theta0))

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

    ey = theta / np.pi
    ex = phi / (2 * np.pi)

    # Scale grid coordinates between [-1, +1]
    ey_grid = ey.reshape(height, width) * 2. - 1
    ex_grid = ex.reshape(height, width) * 2. - 1
    return np.stack([ex_grid, ey_grid], axis=-1)
