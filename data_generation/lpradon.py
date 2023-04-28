import numpy as np
import tensorflow as tf
from keras.layers import *  # todo


class LPRadon(Layer):
    def __init__(self, n_angles, n_det=None, n_span=3, cor=None, interp_type='cubic', *args, **kwargs):
        super(LPRadon, self).__init__(*args, **kwargs)

        self.cor = cor  # I have no idea what this variable does. It should be n // 2
        self.interp_type = interp_type
        self.n_span = n_span
        self.n_angles = n_angles
        self.n_det = n_det

        self.root_2 = tf.convert_to_tensor(np.sqrt(2.), dtype=self.dtype)

        self.complex_dtype = tf.complex(self.root_2, self.root_2).dtype  # hack

        self.n = None
        self.batch_size = None
        self.beta = None
        self.a_R = None
        self.a_m = None
        self.g = None
        self.d_rho = None
        self.n_th = None
        self.n_rho = None
        self.d_th = None
        self.angles = None
        self.s = None
        self.th_lp = None
        self.rho_lp = None
        self.b3_com = None
        self.zeta_coeffs = None
        self.lp2c = None
        self.p2lp = None
        # self.lp2c1 = []
        # self.lp2c2 = []  # literally stands for log-polar to cartesian
        self.pids = []  # indices by span
        # self.p2lp1 = []
        # self.p2lp2 = []

        self.reshape_1 = None
        self.reshape_2 = None

    def build(self, input_shape):
        b, h, w, c = input_shape
        assert h == w, "Image must be square"
        self.n = h

        self.height = self.n_det
        self.width = self.n_det
        self.channels = c

        self.cor = self.cor or self.n // 2

        self.beta = np.pi / self.n_span

        # expand the image so that we have enough room to rotate
        self.n = int(np.ceil((self.n + abs(self.n / 2 - self.cor) * 2.) / 16.) * 16.)
        self.n_det = self.n_det or self.n

        # oversample, this will change the shape of the image
        os_angles = int(max(round(3. * self.n / 2 / self.n_angles), 1))
        self.n_angles = os_angles * self.n_angles

        # polar grid
        self.angles = np.arange(0, self.n_angles) * np.pi / self.n_angles - self.beta / 2
        self.s = np.linspace(-1, 1, self.n_det)  # idk why one is np.arange and the other is np.linspace

        self.get_lp_params()

        # log-polar grid
        self.th_lp = np.arange(-self.n_th / 2, self.n_th / 2) * self.d_th
        self.rho_lp = np.arange(-self.n_rho, 0) * self.d_rho

        # compensate for cubic interpolation
        b3_th = splineB3(self.th_lp, 1)
        b3_th = np.fft.fft(np.fft.ifftshift(b3_th))
        b3_rho = splineB3(self.rho_lp, 1)
        b3_rho = np.fft.fft(np.fft.ifftshift(b3_rho))
        self.b3_com = np.outer(b3_rho, b3_th)

        # forward projection params
        # convolution function
        self.zeta_coeffs = np.fft.fftshift(self.get_zeta_coeffs())

        # log-polar to cartesian
        tmp1 = np.outer(np.exp(np.array(self.rho_lp)), np.cos(np.array(self.th_lp))).flatten()
        tmp2 = np.outer(np.exp(np.array(self.rho_lp)), np.sin(np.array(self.th_lp))).flatten()

        # print(self.rho_lp.shape, self.th_lp.shape)

        lp2c1 = []
        lp2c2 = []

        for k in range(self.n_span):
            lp2c1.append(((tmp1 - (1 - self.a_R)) * np.cos(k * self.beta + self.beta / 2) -
                          tmp2 * np.sin(k * self.beta + self.beta / 2)) / self.a_R)
            lp2c2.append(((tmp1 - (1 - self.a_R)) * np.sin(k * self.beta + self.beta / 2) +
                          tmp2 * np.cos(k * self.beta + self.beta / 2)) / self.a_R)
            lp2c2[k] *= (-1)
            # cids = np.where((lp2c1[k] ** 2 + lp2c2[k] ** 2) <= 1)[0]
            # lp2c1[k] = lp2c1[k][cids]
            # lp2c2[k] = lp2c2[k][cids]

        s0, th0 = np.meshgrid(self.s, self.angles)
        th0 = th0.flatten()
        s0 = s0.flatten()
        for k in range(self.n_span):
            self.pids.append((th0 >= k * self.beta - self.beta / 2) &
                             (th0 < k * self.beta + self.beta / 2))

        # (self.pids[0].shape)

        p2lp1 = []
        p2lp2 = []

        # polar to log-polar coordinates
        for k in range(self.n_span):
            th00 = th0 - k * self.beta
            s00 = s0
            p2lp1.append(th00)
            p2lp2.append(np.log(s00 * self.a_R + (1 - self.a_R) * np.cos(th00)))
            np.nan_to_num(p2lp2[k], copy=False)

        # transform to unnormalized coordinates for interp
        for k in range(self.n_span):
            lp2c1[k] = (lp2c1[k] + 1) / 2 * (self.n_det - 1)
            lp2c2[k] = (lp2c2[k] + 1) / 2 * (self.n_det - 1)
            p2lp1[k] = (p2lp1[k] - self.th_lp[0]) / (self.th_lp[-1] - self.th_lp[0]) * (self.n_th - 1)
            p2lp2[k] = (p2lp2[k] - self.rho_lp[0]) / (self.rho_lp[-1] - self.rho_lp[0]) * (self.n_rho - 1)

        self.lp2c = tf.stack([lp2c1, lp2c2], axis=-1)
        self.p2lp = tf.stack([p2lp2, p2lp1], axis=-1)

        const = np.sqrt(self.n_det * os_angles / self.n_angles) * np.pi / 4 / self.a_R / np.sqrt(2)  # black magic
        self.zeta_coeffs = self.zeta_coeffs[:, :self.n_th // 2 + 1] * const

        if self.interp_type == 'cubic':
            self.zeta_coeffs /= self.b3_com[:, :self.n_th // 2 + 1]

        self.reshape_1 = Reshape((self.n_rho, self.n_th, 1), name="reshape_1")
        self.reshape_2 = Reshape((self.n_angles, self.n_det, 1), name="reshape_2")

        # some other pre-computation stuff
        self.e_rho = tf.reshape(tf.exp(tf.convert_to_tensor(self.rho_lp, dtype=self.dtype)), (1, -1, 1, 1))
        self.zeta_coeffs = tf.convert_to_tensor(self.zeta_coeffs, dtype=self.complex_dtype)
        self.pids = [tf.reshape(tf.convert_to_tensor(self.pids[k], dtype=self.dtype),
                                (1, self.n_angles * self.n_det, 1)) for k in range(len(self.pids))]

        # precompute interpolation stuff

        # first interpolation
        grid_shape = (self.height, self.width)
        query_shape = tf.shape(self.lp2c[0:1])

        self.num_queries = query_shape[1]

        self.floors = []  # tf.TensorArray(tf.int32, size=len(self.lp2c), dynamic_size=False)
        for k in range(len(self.lp2c)):
            floors = []  # tf.TensorArray(tf.int32, size=2, dynamic_size=False)

            index_order = [0, 1]
            unstacked_query_points = tf.unstack(self.lp2c[k:k + 1], axis=2, num=2)

            for i, dim in enumerate(index_order):
                queries = unstacked_query_points[dim]
                queries = tf.cast(queries, dtype=self.dtype)

                size_in_indexing_dimension = grid_shape[i]

                # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
                # is still a valid index into the grid.
                max_floor = tf.cast(size_in_indexing_dimension - 2, self.dtype)
                min_floor = tf.constant(0.0, dtype=self.dtype)

                floor = tf.math.minimum(
                    tf.math.maximum(min_floor, tf.math.floor(queries)), max_floor
                )
                int_floor = tf.cast(floor, tf.int32)
                floors.append(int_floor)

            self.floors.append(tf.stack(floors))

        self.floors = tf.stack(self.floors)

        # second interpolation
        self.height_2 = int(self.n_rho)
        self.width_2 = int(self.n_angles * self.n_det / self.n_rho)

        grid_shape_2 = (self.height_2, self.width_2)
        query_shape_2 = tf.shape(self.p2lp[0:1])

        self.num_queries_2 = query_shape_2[1]

        self.floors_2 = []  # tf.TensorArray(tf.int32, size=len(self.p2lp), dynamic_size=False)
        for k in range(len(self.p2lp)):
            floors_2 = []  # tf.TensorArray(tf.int32, size=2, dynamic_size=False)

            index_order_2 = [0, 1]
            unstacked_query_points_2 = tf.unstack(self.p2lp[k:k + 1], axis=2, num=2)

            for i, dim in enumerate(index_order):
                queries_2 = unstacked_query_points_2[dim]
                queries_2 = tf.cast(queries_2, dtype=self.dtype)

                size_in_indexing_dimension_2 = grid_shape_2[i]

                # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
                # is still a valid index into the grid.
                max_floor_2 = tf.cast(size_in_indexing_dimension_2 - 2, self.dtype)
                min_floor_2 = tf.constant(0.0, dtype=self.dtype)

                floor_2 = tf.math.minimum(
                    tf.math.maximum(min_floor_2, tf.math.floor(queries_2)), max_floor_2
                )
                int_floor_2 = tf.cast(floor_2, tf.int32)

                floors_2.append(int_floor_2)

            self.floors_2.append(tf.stack(floors_2))

        self.floors_2 = tf.stack(self.floors_2)

    def get_lp_params(self):
        self.a_R = np.sin(self.beta / 2) / (1 + np.sin(self.beta / 2))
        self.a_m = (np.cos(self.beta / 2) - np.sin(self.beta / 2)) / (1 + np.sin(self.beta / 2))

        t = np.linspace(-np.pi / 2, np.pi / 2, 1000)
        w = self.a_R * np.cos(t) + (1 - self.a_R) + 1j * self.a_R * np.sin(t)
        self.g = np.nanmax(np.log(abs(w)) + np.log(np.cos(self.beta / 2 - np.arctan2(w.imag, w.real))))

        self.n_th = self.n_det
        self.n_rho = 2 * self.n_det

        self.d_th = 2 * self.beta / self.n_th
        self.d_rho = (self.g - np.log(self.a_m)) / self.n_rho

    def get_zeta_coeffs(self, a=0, osthlarge=4):
        k_rho = np.arange(-self.n_rho / 2, self.n_rho / 2, dtype='float32')
        n_th_large = osthlarge * self.n_th
        th_sp_large = np.arange(-n_th_large / 2, n_th_large / 2) / n_th_large * self.beta * 2
        fZ = np.zeros([self.n_rho, n_th_large], dtype='complex64')
        h = np.ones(n_th_large, dtype='float32')
        # correcting = 1+[-3 4 -1]/24correcting(1) = 2*(correcting(1)-0.5)
        # correcting = 1+array([-23681,55688,-66109,57024,-31523,9976,-1375])/120960.0correcting[0]
        # = 2*(correcting[0]-0.5)
        correcting = 1 + np.array([-216254335, 679543284, -1412947389, 2415881496, -3103579086,
                                   2939942400, -2023224114, 984515304, -321455811, 63253516, -5675265]) / 958003200.0
        correcting[0] = 2 * (correcting[0] - 0.5)
        h[0] = h[0] * (correcting[0])

        for j in range(1, len(correcting)):
            h[j] = h[j] * correcting[j]
            h[-1 - j + 1] = h[-1 - j + 1] * (correcting[j])

        for j in range(len(k_rho)):
            fcosa = np.power(np.cos(th_sp_large), (-2 * np.pi * 1j * k_rho[j] / (self.g - np.log(self.a_m)) - 1 - a))
            fZ[j, :] = np.fft.fftshift(np.fft.fft(np.fft.fftshift(h * fcosa)))

        fZ = fZ[:, n_th_large // 2 - self.n_th // 2:n_th_large // 2 + self.n_th // 2]
        fZ = fZ * (th_sp_large[1] - th_sp_large[0])

        # put imag to 0 for the border
        fZ[0] = 0
        fZ[:, 0] = 0
        return fZ

    def interpolate(self, grid, k):
        batch_size = tf.shape(grid)[0]

        floors = self.floors[k]

        flattened_grid = tf.reshape(grid, [-1, self.channels])
        batch_offsets = tf.reshape(
            tf.range(batch_size) * self.height * self.width, [-1, 1]
        )

        def gather(y_coords, x_coords):
            linear_coordinates = batch_offsets + y_coords * self.width + x_coords
            gathered_values = tf.gather(flattened_grid, linear_coordinates)
            return tf.reshape(gathered_values, [-1, self.num_queries, self.channels])

        return gather(floors[0], floors[1])

    def interpolate_2(self, grid, k):
        batch_size = tf.shape(grid)[0]

        floors = self.floors_2[k]

        flattened_grid = tf.reshape(grid, [-1, self.channels])
        batch_offsets = tf.reshape(
            tf.range(batch_size) * self.height_2 * self.width_2, [-1, 1]
        )

        def gather(y_coords, x_coords):
            linear_coordinates = batch_offsets + y_coords * self.width_2 + x_coords
            gathered_values = tf.gather(flattened_grid, linear_coordinates)
            return tf.reshape(gathered_values, [-1, self.num_queries_2, self.channels])

        # grab the pixel values in the 4 corners around each query point
        top_left = gather(floors[0], floors[1])
        return top_left

    def call(self, inputs, *args, **kwargs):
        b, h, w, c = inputs.shape

        f = tf.image.pad_to_bounding_box(inputs, (self.n_det - h) // 2, (self.n_det - h) // 2, self.n_det, self.n_det)

        out = tf.zeros((1, self.n_angles * self.n_det, 1))
        for k in range(self.n_span):
            # interpolate to log-polar grid
            lp_img = self.reshape_1(self.interpolate(f, k))

            # multiply by e^rho
            lp_img *= self.e_rho

            # fft
            fft_img = tf.signal.rfft2d(tf.squeeze(lp_img, axis=-1))
            fft_img *= self.zeta_coeffs

            # ifft
            lp_sinogram = tf.expand_dims(tf.signal.irfft2d(fft_img), -1)
            p_sinogram = self.interpolate_2(lp_sinogram, k)
            p_sinogram *= self.pids[k]
            out += p_sinogram

        return self.reshape_2(out)


def splineB3(x2, r):
    sizex = len(x2)
    x2 = x2 - (x2[-1] + x2[0]) / 2
    stepx = x2[1] - x2[0]
    ri = int(np.ceil(2 * r))
    r = r * stepx
    x2c = x2[int(np.ceil((sizex + 1) / 2.0)) - 1]
    x = x2[int(np.ceil((sizex + 1) / 2.0) - ri - 1):int(np.ceil((sizex + 1) / 2.0) + ri)]
    d = np.abs(x - x2c) / r
    B3 = x * 0
    for ix in range(-ri, ri + 1):
        id_ = ix + ri
        if d[id_] < 1:  # use the first polynomial
            B3[id_] = (3 * d[id_] ** 3 - 6 * d[id_] ** 2 + 4) / 6
        else:
            if d[id_] < 2:
                B3[id_] = (-d[id_] ** 3 + 6 * d[id_] ** 2 - 12 * d[id_] + 8) / 6

    B3f = x2 * 0
    B3f[int(np.ceil((sizex + 1) / 2.0) - ri - 1):int(np.ceil((sizex + 1) / 2.0) + ri)] = B3
    return B3f


def main():
    import matplotlib.pyplot as plt
    import h5py
    import time
    with h5py.File("../data/ground_truth_test/ground_truth_test_000.hdf5") as f:
        img = f['data'][:64]
        img = img[:, :, :, tf.newaxis]
    print(img.shape)

    test = LPRadon(1024, 513, n_span=3)
    start_time = time.time()
    plt.imshow(test(img)[0], cmap='gray')
    print(time.time() - start_time)
    plt.show()

    # plt.imshow(np.reshape(test.pids[0], (test.n_angles, test.n_det)), cmap='gray')
    # plt.show()
    # plt.imshow(np.reshape(test.pids[1], (test.n_angles, test.n_det)), cmap='gray')
    # plt.show()
    # plt.imshow(np.reshape(test.pids[2], (test.n_angles, test.n_det)), cmap='gray')
    # plt.show()
    # plt.imshow(test(img)[0].numpy(), cmap='gray')
    # plt.show()
    # plt.imshow(test.zeta_coeffs.real, cmap='gray')
    # plt.show()
    #
    # plt.scatter(x=test.lp2c1[0], y=test.lp2c2[0], s=0.01)
    # plt.gca().set_aspect('equal')
    # plt.show()
    #
    # plt.scatter(x=test.lp2c1[1], y=test.lp2c2[1], s=0.01)
    # plt.gca().set_aspect('equal')
    # plt.show()
    #
    # plt.scatter(x=test.lp2c1[2], y=test.lp2c2[2], s=0.01)
    # plt.gca().set_aspect('equal')
    # plt.show()
    #
    # plt.imshow(test.pids, aspect='auto', interpolation='nearest')
    # plt.show()


if __name__ == '__main__':
    main()
