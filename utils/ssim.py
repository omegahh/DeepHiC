import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp

def _toimg(mat):
    m = torch.tensor(mat)
    # convert to float and add channel dimension
    return m.float().unsqueeze(0)

def _tohic(mat):
    mat.squeeze_()
    return mat.numpy()#.astype(int)

def gaussian(width, sigma):
    gauss = torch.Tensor([exp(-(x-width//2)**2 / float(2 * sigma**2)) for x in range(width)])
    return gauss / gauss.sum()

def create_window(window_size, channel, sigma=3):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def gaussian_filter(img, width, sigma=3):
    img = _toimg(img).unsqueeze(0)
    _, channel, _, _ = img.size()
    window = create_window(width, channel, sigma)
    mu1 = F.conv2d(img, window, padding=width // 2, groups=channel)
    return _tohic(mu1)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
    img1 = _toimg(img1).unsqueeze(0)
    img2 = _toimg(img2).unsqueeze(0)
    _, channel, _, _ = img1.size()
    window = create_window(window_size, channel)
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)
    
def noise_estimator(mat):
    # https://www.cnblogs.com/algorithm-cpp/p/4105943.html
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])
    img = _toimg(mat).unsqueeze(0)
    ker = _toimg(kernel).unsqueeze(0)
    out = F.conv2d(img, ker)
    out = _tohic(out)
    noise = np.sum(out)/(out.shape[0]*out.shape[1])
    return noise
