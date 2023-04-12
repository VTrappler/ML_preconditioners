import numpy as np


class ObservationError:
    def __init__(self, dim) -> None:
        self.inv_obs_error_covariance_matrix = np.eye(dim)
        self.obs_error_covariance_matrix = np.eye(dim)


class ObservationErrorArray(ObservationError):
    def __init__(self, dim, path) -> None:
        super().__init__(dim)
        self.load_array(path=path)

        self.inv_obs_error_covariance_matrix = np.linalg.inv(
            self.obs_error_covariance_matrix
        )
        self.half_obs_error_covariance_matrix = np.linalg.cholesky(
            self.obs_error_covariance_matrix
        )

    def load_array(self, path):
        self.line = np.genfromtxt(path)
        self.obs_error_covariance_matrix = self.construct_matrix_line(self.line)

    def construct_matrix_line(self, array):
        matrix = np.zeros((array.size, array.size))
        for i in range(array.size):
            matrix[i, :] = np.roll(array, i)
        return matrix
