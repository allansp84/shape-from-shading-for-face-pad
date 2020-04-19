# -*- coding: utf-8 -*-

import numpy as np
from scipy import ndimage
from skimage.util.shape import view_as_windows
import pdb


class LightEstimationError(Exception):
    pass


def lee_method(img):
    # print 'Lee\'s method'

    iy = ndimage.filters.sobel(img, axis=0)
    ix = ndimage.filters.sobel(img, axis=1)

    tilt = np.atan(iy.mean()/ix.mean())
    tilt = np.degrees(tilt)

    if tilt < 0:
        tilt += 360

    print('tilt', tilt)


def estimate_albedo_illumination(image):

    image = np.array(image)

    # -- normalizing the image
    e = image/image.max()

    # -- compute the average of the image brightness
    mu_1 = np.mean(e)
    # -- compute the average of the image brightness square
    mu_2 = np.mean(np.mean(e*e))

    # -- now lets compute the image's spatial gradient in x and y directions
    ex, ey = np.gradient(e)

    # -- normalize the gradients to be unit vectors
    exy = np.sqrt(ex*ex + ey*ey)
    n_ex = ex / (exy + 0.0001)
    n_ey = ey / (exy + 0.0001)

    # -- computing the average of the normalized gradients
    avg_ex = np.mean(n_ex)
    avg_ey = np.mean(n_ey)

    # -- now lets estimate the surface albedo
    gamma = np.sqrt((6 * np.pi*np.pi * mu_2) - (48 * mu_1*mu_1))
    albedo = gamma/np.pi

    # -- estimating the slant
    zl = 4*mu_1/gamma
    zl = min(1.0, max(-1.0, zl))
    slant = np.arccos(zl)

    # -- estimating the tilt
    tilt = np.arctan(avg_ey/avg_ex)
    if np.degrees(tilt) < 0:
        tilt += 2*np.pi

    # -- the illumination direction will be:
    l = np.array([np.cos(tilt)*np.sin(slant), np.sin(tilt)*np.sin(slant), np.cos(slant)])

    return l, albedo, slant, tilt


def pentland_method(img):
    # print 'Pentland\'s method'

    img = np.array(img)

    beta = np.array([[1.0, np.sqrt(2.0)/2.0, 0.0, -np.sqrt(2.0)/2.0, -1.0, -np.sqrt(2.0)/2.0,  0.0,  np.sqrt(2.0)/2.0],
                     [0.0, np.sqrt(2.0)/2.0, 1.0,  np.sqrt(2.0)/2.0,  0.0, -np.sqrt(2.0)/2.0, -1.0, -np.sqrt(2.0)/2.0],
                     ]).transpose()

    kernels = np.array([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                        [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
                        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                        [[2, 1, 0], [1, 0, -1], [0, -1, -2]],
                        [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                        [[0, -1, -2], [1, 0, -1], [2, 1, 0]],
                        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                        [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]],
                        ])

    di = []
    for d in range(kernels.shape[0]):
        di.append((ndimage.convolve(img, kernels[d], cval=0.0)/4.0).mean())
    di = np.array(di).reshape(-1, 1)

    xl, yl = np.dot(np.dot(np.linalg.inv(np.dot(beta.transpose(), beta)), beta.transpose()), di)

    k = di.std()
    # slant = np.sqrt(np.arccos(1.0-((xl*xl + yl*yl)/k*k))) if (xl*xl + yl*yl <= k*k) else 0
    # tilt = np.arctan(yl/xl)

    if k:
        xl = float(xl/k)
        yl = float(yl/k)
        zl = float(np.sqrt(np.abs(1.0 - xl**2 - yl**2)))

        l = np.abs(np.array([xl, yl, zl]))
        l /= np.linalg.norm(l)

        # l[2] = -1
    else:
        l = [0.01, 0.01, 1.0]

    return l


def farid_method(normals, intensities):
    normals = np.array(normals)
    normals = np.reshape(normals, (-1, 3))

    intensities = np.array(intensities)
    intensities = intensities.flatten()

    hv = np.ones((normals.shape[0], 1))
    m = np.concatenate((normals, hv), axis=1)
    m1 = np.dot(m.transpose(), m)
    # condition = np.linalg.norm(m1, 2)* np.linalg.norm(np.linalg.pinv(m1), 2)
    m2 = np.linalg.pinv(m1)
    m3 = np.dot(m2, m.transpose())
    m4 = m3*intensities

    m5 = np.array([abs(m4[0, 0]), abs(m4[1, 0]), abs(m4[2, 0])])

    try:
        if np.linalg.norm(m5) != 0:
            light_direction = m5/np.linalg.norm(m5)
        else:
            raise LightEstimationError("Error in the estimation of the light direction")
    except Exception:
        raise LightEstimationError("Error in the estimation of the light direction")

    return light_direction


def light_direction_global_estimation(normals, intensities):

    # albedo = 1.0
    # light_direction = [0., 0., 1.]
    #
    # if 'farid' in method:
    #
    #     light_direction = farid_method(normals, intensities)
    #     print light_direction
    #
    # elif 'pentland' in method:
    #
    #     light_direction = pentland_method(intensities)
    #     print light_direction
    #
    # elif 'lee' in method:
    #
    #     light_direction, albedo, slant, tilt = estimate_albedo_illumination(intensities)
    #     print light_direction, albedo, slant, tilt
    #
    # else:
    #     pass

    light_direction = farid_method(normals, intensities)

    return light_direction


def light_direction_local_estimation(normals, intensities):

    intensities = intensities[1:, 1:]
    norm = np.reshape(normals, (intensities.shape + (3,)))

    local_ests = local_light_direction(norm, intensities)

    final_ld = np.zeros(intensities.shape + (3,), dtype=np.float32)

    n_rows, n_cols = intensities.shape
    row_l = 0

    for row in range(0, n_rows, 16):
        col_l = 0
        for col in range(0, n_cols, 16):
            final_ld[row:row+16, col:col+16, :] = local_ests[row_l + col_l*16, :]
            col_l += 1
        row_l += 1

    return final_ld


def local_light_direction(norm, img):

    block_size = 9
    stride = 9
    shape = (block_size, block_size)

    c_img = np.ascontiguousarray(img)
    c_norm0 = np.ascontiguousarray(norm[:, :, 0])
    c_norm1 = np.ascontiguousarray(norm[:, :, 1])
    c_norm2 = np.ascontiguousarray(norm[:, :, 2])

    img_windows = view_as_windows(c_img, shape, stride)

    norm_windows_x = view_as_windows(c_norm0[:, :], (block_size, block_size), stride)
    norm_windows_y = view_as_windows(c_norm1[:, :], (block_size, block_size), stride)
    norm_windows_z = view_as_windows(c_norm2[:, :], (block_size, block_size), stride)

    norm_block = np.zeros((block_size*block_size, 3), dtype=norm_windows_x.dtype)

    n_rows, n_cols = img_windows.shape[:2]
    local_ld_estimation = []
    for row in range(n_rows):
        for col in range(n_cols):
            norm_block[:, 0] = norm_windows_x[row, col].flatten()
            norm_block[:, 1] = norm_windows_y[row, col].flatten()
            norm_block[:, 2] = norm_windows_z[row, col].flatten()

            # print norm_block.shape
            # print img_windows[row, col].shape

            local_ld_estimation += [light_direction_global_estimation(norm_block, img_windows[row, col])]

    return np.array(local_ld_estimation)
