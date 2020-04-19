# -*- coding: utf-8 -*-

import sys
import cv2
import math
import matplotlib.pyplot as plt
from antispoofing.sfsnet.features.estimation.lightdirection import *


class Tsai(object):

    def __init__(self, image=None, input_fname='', output_fname='', local_estimation=True, light_direction='constant'):

        self.image = image

        self.input_fname = input_fname
        self.output_fname = output_fname

        self.depth_map = None
        self.albedo_map = None
        self.reflectance_map = None

        self.normals = None
        self.albedo_free_image = None
        self.albedo_free_depth = None

        self.NB_ITERATIONS = 5
        self.Wn = 0.001

        self.local_estimation = local_estimation
        self.light_direction = light_direction

        self.debug = False

    def load_image(self):
        self.image = np.asarray(cv2.imread(self.input_fname, cv2.IMREAD_GRAYSCALE))

    @staticmethod
    def normalization(data, min_value=0., max_value=255.):
        data = np.array(data, dtype=np.float32)

        data_min = data.min()
        data_max = data.max()

        if data_min != data_max:
            data_norm = (data - data_min) / (data_max - data_min)
            data_scaled = data_norm * (max_value - min_value) + min_value
        else:
            data_scaled = data

        return data_scaled

    # @profile  # -- used for line_profile and memory_profiler packages
    def __compute_global_sfs(self):

        self.image = np.array(self.image, dtype=np.float32)

        image_shape = self.image.shape[:2]

        self.depth_map = np.zeros(image_shape, dtype=np.float32)
        self.reflectance_map = np.zeros(image_shape, dtype=np.float32)
        self.albedo_map = np.zeros(image_shape, dtype=np.float32)

        zk1 = np.zeros(image_shape, dtype=np.float32)
        sk1 = np.ones(image_shape, dtype=np.float32)

        if 'constant' in self.light_direction:
            light = [0.1, 0.1, 1.0]
            xl, yl, zl = light
        else:
            raise(Exception, 'Method for light estimation not found!')

        try:
            max_pixel = self.image.max()
            assert max_pixel != 0, 'invalid value for max value pixel!'
        except AssertionError:
            # -- the image is flat so there is nothing to do
            return None

        ps = np.ones(image_shape, dtype=np.float32) * (xl / zl)  # 0.0
        qs = np.ones(image_shape, dtype=np.float32) * (yl / zl)  # 1.0

        for it in range(self.NB_ITERATIONS):

            # -- compute gradient
            p, q = np.gradient(zk1)

            # -- create normal vector for each surface point
            self.normals = np.dstack([-p, -q, np.ones(p.shape)])

            # -- compute the norm of each surface point
            norm = np.linalg.norm(self.normals, axis=2)

            # -- compute the unit vectors
            self.normals[:, :, 0] /= norm
            self.normals[:, :, 1] /= norm
            self.normals[:, :, 2] /= norm

            pq = 1.0 + p*p + q*q
            pqs = 1.0 + ps*ps + qs*qs

            # -- compute the reflectance map
            self.reflectance_map = (1.0 + p*ps + q*qs) / (np.sqrt(pq) * np.sqrt(pqs))
            self.reflectance_map = np.maximum(np.zeros(image_shape), self.reflectance_map)

            fz = -1.0 * ((self.image/max_pixel) - self.reflectance_map)
            dfz = -1.0 * ((ps+qs)/(np.sqrt(pq)*np.sqrt(pqs))-((p+q)*(1.0+p*ps+q*qs)/(np.sqrt(pq*pq*pq)*np.sqrt(pqs))))

            y = fz + (dfz * zk1)
            k = (sk1 * dfz) / (self.Wn + sk1 * dfz * dfz)

            sk = (1.0 - (k * dfz)) * sk1
            self.depth_map = zk1 + k * (y - (dfz * zk1))

            # -- update depth map
            zk1 = self.depth_map
            sk1 = sk

            # -- estimation of the light direction for next iteration
            if 'iterative' in self.light_direction:
                try:
                    light = light_direction_global_estimation(self.normals, self.image)
                    light = abs(light)
                    print(light)
                    sys.stdout.flush()

                    xl, yl, zl = light

                    ps = np.ones(image_shape, dtype=np.float32) * (xl / zl)  # 0.0
                    qs = np.ones(image_shape, dtype=np.float32) * (yl / zl)  # 1.0

                except LightEstimationError:
                    pass

            l_dot_n = np.abs(np.dot(self.normals, np.array(light)))
            l_dot_n[l_dot_n == 0.] = 1.

            self.albedo_map = self.image / l_dot_n
            self.albedo_map[self.albedo_map > 255] = 255.
            self.albedo_map[self.albedo_map < 1.] = 1.

        self.reflectance_map = self.reflectance_map / np.pi

    def __compute_local_sfs(self):

        n_rows, n_cols = self.image.shape[:2]

        self.depth_map = np.zeros((n_rows, n_cols), dtype=np.float32)
        self.reflectance_map = np.zeros((n_rows, n_cols), dtype=np.float32)
        self.albedo_map = np.zeros((n_rows, n_cols), dtype=np.float32)

        sk = np.zeros((n_rows, n_cols), dtype=np.float32)
        fzk = np.zeros((n_rows, n_cols), dtype=np.float32)
        dfzk = np.zeros((n_rows, n_cols), dtype=np.float32)

        zk1 = np.zeros((n_rows, n_cols), dtype=np.float32)
        sk1 = np.ones((n_rows, n_cols), dtype=np.float32)

        ps = np.zeros((n_rows, n_cols), dtype=np.float32)
        qs = np.ones((n_rows, n_cols), dtype=np.float32)

        for it in range(self.NB_ITERATIONS):

            self.normals = []
            for i in range(1, n_rows):
                for j in range(1, n_cols):

                    # -- compute dz/dx
                    if (j - 1) >= 0:
                        p = zk1[i, j] - zk1[i, (j - 1)]
                    else:
                        p = 0.0

                    # -- compute dz/dy
                    if (i - 1) >= 0:
                        q = zk1[i, j] - zk1[(i - 1), j]
                    else:
                        q = 0.0

                    n = np.array([-1.*p, -1.*q, 1.])
                    n /= np.linalg.norm(n)
                    self.normals += [n]

                    # -- compute the reflectance map
                    pq = 1.0 + p * p + q * q
                    pqs = 1.0 + ps[i, j] * ps[i, j] + qs[i, j] * qs[i, j]
                    rij = max(0.0, (1 + p * ps[i, j] + q * qs[i, j]) / (math.sqrt(pq) * math.sqrt(pqs)))
                    self.reflectance_map[i, j] = rij / math.pi

                    # -- compute fz function and its derivate
                    eij = self.image[i, j] / self.image.max()

                    fz = -1.0 * (eij - rij)
                    dfz = -1.0 * ((ps[i, j] + qs[i, j]) / (math.sqrt(pq) * math.sqrt(pqs)) - (p + q) *
                                  (1.0 + p * ps[i, j] + q * qs[i, j]) / (math.sqrt(pq * pq * pq) * math.sqrt(pqs)))

                    y = fz + (dfz * zk1[i, j])
                    k = (sk1[i, j] * dfz) / (self.Wn + sk1[i, j] * dfz * dfz)

                    sk[i, j] = (1.0 - (k * dfz)) * sk1[i, j]
                    self.depth_map[i, j] = zk1[i, j] + k * (y - (dfz * zk1[i, j]))

                    fzk[i, j] = fz
                    dfzk[i, j] = dfz

            # -- update depth map
            for i in range(n_rows):
                for j in range(n_cols):
                    zk1[i, j] = self.depth_map[i, j]
                    sk1[i, j] = sk[i, j]

            # -- estimation of the light direction
            xl, yl, zl = 0.01, 0.01, 1.00
            if not 'constant' in self.light_direction:
                try:
                    # xl, yl, zl = farid_method(self.normals, self.image)
                    # xl, yl, zl = new_method_opt(self.normals, self.image)

                    light_directions = light_direction_local_estimation(self.normals, self.image)

                    zero_row = np.zeros((1, n_cols - 1, 3), dtype=np.float32)
                    zero_row[0, 0:n_cols-1, :] = light_directions[0, 0:n_cols-1, :]

                    zero_col = np.zeros((n_rows, 1, 3), dtype=np.float32)
                    zero_col[0:n_rows-1, 0, :] = light_directions[0:n_rows-1, 0, :]
                    zero_col[-1, 0, :] = light_directions[-1, 0, :]

                    light_directions = np.concatenate((zero_row, light_directions), axis=0)
                    light_directions = np.concatenate((zero_col, light_directions), axis=1)

                    for i in range(n_rows):
                        for j in range(n_cols):
                            xl, yl, zl = light_directions[i, j]

                            tilt_l = np.arctan2(yl, (xl + self.Wn))

                            zl = max(-1.0, float(format("%.8f" % zl)))
                            zl = min(1.0, float(format("%.8f" % zl)))
                            slant_l = np.arccos(zl)

                            if np.degrees(tilt_l) < 0:
                                tilt_l += 2 * np.pi

                            if np.degrees(slant_l) < 0:
                                slant_l += 2 * np.pi

                            ps[i, j] = np.cos(tilt_l) * np.sin(slant_l) / np.cos(slant_l)
                            qs[i, j] = np.sin(tilt_l) * np.sin(slant_l) / np.cos(slant_l)

                except LightEstimationError:
                    xl, yl, zl = 0.01, 0.01, 1.00

            self.normals = np.array(self.normals)
            self.normals = np.reshape(self.normals, (n_rows - 1, n_cols - 1, 3))
            zero_row = np.zeros((1, n_cols - 1, 3), dtype=np.float32)
            zero_col = np.zeros((n_rows, 1, 3), dtype=np.float32)

            self.normals = np.concatenate((zero_row, self.normals), axis=0)
            self.normals = np.concatenate((zero_col, self.normals), axis=1)

            for y in range(n_rows):
                for x in range(n_cols):

                    # if y != 0 and x != 0:
                    if np.abs(np.dot(self.normals[y, x, :], np.array([xl, yl, zl]))) != 0:
                        self.albedo_map[y, x] = self.image[y, x] / np.abs(np.dot(self.normals[y, x, :],
                                                                                 np.array([xl, yl, zl])))

                    else:
                        self.albedo_map[y, x] = self.image[y, x]

            self.albedo_map[self.albedo_map > 255] = 255.
            self.albedo_map[self.albedo_map < 1.] = 1.

    def compute_sfs(self):

        if self.image is None:
            self.load_image()

        if self.local_estimation:
            self.__compute_local_sfs()
        else:
            self.__compute_global_sfs()

        self.depth_map = self.normalization(self.depth_map)
        self.reflectance_map = self.normalization(self.reflectance_map)

        self.albedo_free_image = self.image/self.albedo_map
        self.albedo_free_depth = self.depth_map/self.albedo_map

        self.albedo_free_image = self.normalization(self.albedo_free_image)
        self.albedo_free_depth = self.normalization(self.albedo_free_depth)

        self.albedo_map = np.round(self.albedo_map).astype(np.uint8)
        self.depth_map = np.round(self.depth_map).astype(np.uint8)
        self.reflectance_map = np.round(self.reflectance_map).astype(np.uint8)
        self.albedo_free_image = np.round(self.albedo_free_image).astype(np.uint8)
        self.albedo_free_depth = np.round(self.albedo_free_depth).astype(np.uint8)

        return True

    def write_depth_data(self):
        output_depth_data = '{0}.txt'.format(self.output_fname)
        np.savetxt(output_depth_data, self.depth_map, fmt='%f')

    def show_depth_image(self):

        plot_col = 4
        plot_row = 2
        fs = 10

        fig = plt.figure(figsize=(8, 6), dpi=100)

        titles = ['Original', 'Reflectance Map', 'Depth Map', 'Albedo Map', 'Nx', 'Ny', 'Nz', '']
        data_plt = [self.image, self.reflectance_map, self.depth_map, self.albedo_map,
                    self.normals[:, :, 0], self.normals[:, :, 1], self.normals[:, :, 2],
                    np.ones(self.image.shape, dtype=np.uint8) * 255,
                    np.ones(self.image.shape, dtype=np.uint8) * 255,
                    ]

        axis = []
        for k in range((plot_col * plot_row)):
            axis += [fig.add_subplot(plot_row, plot_col, k + 1)]

        for k in range((plot_col * plot_row)):
            axis[k].imshow(data_plt[k], cmap='gray')
            axis[k].set_title(titles[k], fontsize=fs)
            axis[k].set_xticklabels([])
            axis[k].set_yticklabels([])

        plt.show()

    @staticmethod
    def show_normal_surface():

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2),
                              np.arange(-0.8, 1, 0.2),
                              np.arange(-0.8, 1, 0.8))

        u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
        v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
        w = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(np.pi * y) *
             np.sin(np.pi * z))

        ax.quiver(x, y, z, u, v, w, length=0.1)

        plt.show()


if __name__ == '__main__':
    if __package__ is None:
        import sys

    sfs = Tsai(input_fname=sys.argv[1], output_fname='out_c')
    sfs.compute_sfs()
    sfs.show_depth_image()
