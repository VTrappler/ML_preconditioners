import numpy as np


class BackgroundError:
    def __init__(self, dim) -> None:
        self.bck_error_covariance_matrix = np.zeros((dim, dim))
        self.inv_bck_error_covariance_matrix = np.zeros((dim, dim))
        self.half_bck_error_covariance_matrix = np.eye(dim)


class BackgroundErrorArray(BackgroundError):
    def __init__(self, dim, path) -> None:
        super().__init__(dim)
        self.load_array(path=path)
        self.inv_bck_error_covariance_matrix = np.linalg.inv(
            self.bck_error_covariance_matrix
        )
        self.half_bck_error_covariance_matrix = np.linalg.cholesky(
            self.bck_error_covariance_matrix
        )

    def load_array(self, path):
        self.line = np.genfromtxt(path)
        self.bck_error_covariance_matrix = self.construct_matrix_line(self.line)

    def construct_matrix_line(self, array):
        matrix = np.zeros((array.size, array.size))
        for i in range(array.size):
            matrix[i, :] = np.roll(array, i)
        return matrix
