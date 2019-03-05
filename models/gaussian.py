import numpy as np
from math import exp


def gaussian(width, sigma):
    """ Returns a 2D Gaussian kernel array. """
    gauss_1d = np.array([exp(-(x-width//2)**2 / float(2 * sigma**2)) for x in range(width)])
    gauss_1d = gauss_1d / np.sum(gauss_1d)
    kernel = np.outer(gauss_1d, gauss_1d)
    return kernel

# Run the Gaussian smoothing on Hi-C matrix
# matrix is a numpy matrix form of the Hi-C interaction hearmap
def Gaussian_filter(matrix, kernlen=11, sigma=1.5):
    m, n = matrix.shape
    padding = kernlen // 2
    result = np.zeros(matrix.shape)
    kernel = gau_kern(kernlen, nsig=sigma)
    for i in range(padding, m-padding):
        for j in range(padding, n-padding):
            result[i, j] = np.sum(matrix[i-padding:i+padding+1, j-padding:j+padding+1] * kernel)
    result = result[padding:m-padding, padding:n-padding]
    return result
