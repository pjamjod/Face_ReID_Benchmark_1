import numpy as np
from scipy import signal
from scipy import ndimage
from skimage.metrics import structural_similarity

def gaussian2(size, sigma):
    """Returns a normalized circularly symmetric 2D gauss kernel array
    
    f(x,y) = A.e^{-(x^2/2*sigma^2 + y^2/2*sigma^2)} where
    
    A = 1/(2*pi*sigma^2)
    
    as define by Wolfram Mathworld 
    http://mathworld.wolfram.com/GaussianFunction.html
    """
    A = 1/(2.0*np.pi*sigma**2)
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = A*np.exp(-((x**2/(2.0*sigma**2))+(y**2/(2.0*sigma**2))))
    return g

def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

def calculate_psnr(img1, img2):
    # https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def calculate_ssim(img1, img2, window_size=7, K1=0.01, K2=0.03, sigma=1.5, data_range=None, return_cs=False):
    # TODO: Revise with Skimage implementation https://github.com/scikit-image/scikit-image/blob/v0.24.0/skimage/metrics/_structural_similarity.py#L231
    # https://en.wikipedia.org/wiki/Structural_similarity_index_measure
    assert window_size % 2 != 0, "Window size must be odd."
    assert img1.shape == img2.shape, "Input images must have the same dimensions."
    assert img1.shape[0] >= window_size and img1.shape[1] >= window_size, "Input images must be larger than the window size."
    assert img1.dtype == img2.dtype, "Input images must have the same dtype."
    
    if data_range is None:
        if img1.dtype == np.uint8 and img2.dtype == np.uint8:
            data_range = 255
        else:
            raise ValueError("Data range must be provided for non-uint8 images.")

    if len(img1.shape) == 3:
        ssim_channels = []
        for i in range(img1.shape[-1]):
            ssim_channels.append(calculate_single_channel_ssim(img1[..., i], img2[..., i], window_size=window_size, K1=K1, K2=K2, sigma=sigma, data_range=data_range, return_cs=return_cs))
        if return_cs:
            ssim = [x[0] for x in ssim_channels]
            cs = [x[1] for x in ssim_channels]
            return np.mean(ssim), np.mean(cs)
        else:
            return np.mean(ssim_channels)
    else:
        #print(structural_similarity(img1, img2, multichannel=False, channel_axis=-1, win_size=window_size, sigma=1.5, gaussian_weights=True, use_sample_covariance=False, data_range=data_range))
        return calculate_single_channel_ssim(img1, img2, window_size=window_size, K1=K1, K2=K2, sigma=sigma, data_range=data_range, return_cs=return_cs)

from scipy import ndimage as ndi
def calculate_single_channel_ssim(img1, img2, window_size=7, K1=0.01, K2=0.03, sigma=1.5, data_range=None, return_cs=False, return_map=False):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    window = fspecial_gauss(window_size, sigma)
    
    C1 = (K1*data_range)**2
    C2 = (K2*data_range)**2

    mu1 = signal.fftconvolve(img1, window, mode='valid')
    mu2 = signal.fftconvolve(img2, window, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(img1*img1, window, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(img2*img2, window, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(img1*img2, window, mode='valid') - mu1_mu2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))
    ssim_cs_map = (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2)

    pad = (window_size - 1) // 2
    ssim_map = ssim_map[pad:-pad, pad:-pad]
    ssim_cs_map = ssim_cs_map[pad:-pad, pad:-pad]

    if not return_map:
        ssim_map = ssim_map.mean()
        ssim_cs_map = ssim_cs_map.mean()
    if return_cs:
        return ssim_map, ssim_cs_map
    else:
        return ssim_map


def calculate_msssim(img1, img2, level=5, window_size=7, kernel_size=2, K1=0.01, K2=0.03, sigma=1.5, data_range=None, weight=np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])):
    # https://github.com/mubeta06/python/blob/master/signal_processing/sp/ssim.py#L50
    if data_range is None:
        if img1.dtype == np.uint8 and img2.dtype == np.uint8:
            data_range = 255
        else:
            raise ValueError("Data range must be provided for non-uint8 images.")

    downsample_filter = np.ones((kernel_size, kernel_size))/kernel_size**2
    im1 = img1.astype(np.float64)
    im2 = img2.astype(np.float64)
    mssim = np.array([])
    mcs = np.array([])
    for l in range(level):
        ssim, cs = calculate_ssim(im1.astype(np.uint8), im2.astype(np.uint8), window_size=window_size, K1=K1, K2=K2, sigma=sigma, data_range=data_range, return_cs=True)
        mssim = np.append(mssim, ssim)
        mcs = np.append(mcs, cs)

        if len(im1.shape) == 3:
            filtered_im1 = np.zeros_like(im1)
            filtered_im2 = np.zeros_like(im2)
            for c in range(im1.shape[-1]):
                filtered_im1[:,:,c] = ndimage.filters.convolve(im1[:,:,c], downsample_filter, 
                                                        mode='reflect')
                filtered_im2[:,:,c] = ndimage.filters.convolve(im2[:,:,c], downsample_filter, 
                                                        mode='reflect')
            im1 = filtered_im1[::2, ::2, :]
            im2 = filtered_im2[::2, ::2, :]
        else:
            filtered_im1 = ndimage.filters.convolve(im1, downsample_filter, mode='reflect')
            filtered_im2 = ndimage.filters.convolve(im2, downsample_filter, mode='reflect')                                        
            im1 = filtered_im1[::2, ::2]
            im2 = filtered_im2[::2, ::2]
    return (np.prod(mcs[0:level-1]**weight[0:level-1])*
                    (mssim[level-1]**weight[level-1]))
