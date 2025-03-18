import numpy as np 

import jax 
import jax.numpy as jnp 
from jax.numpy.fft import fftn,ifftn,fftshift,ifftshift

from kltpicker_3d.spectral_estimation import estimate_isotropic_powerspectrum_tensor

def generate_exponential_autocorrelation(N, K,T=1,seed=42):
    """
    Generates noisy tensors with autocorrelation of the form exp(-T*r).
    The function returns the samples along with a reference power spectrum for benchmark.

    Args:
        N: size of noise image (NxN)
        K: number of noise images to generate 
        T: the decay rate of the ACF
    """
    M = 2*N - 1 
    grid = np.arange(-(N-1),(N-1))
    i,j,k = np.meshgrid(grid,grid,grid)
    autocorrelation = np.exp(- T * (i**2 + j**2 + k**2))

    # We adjust the scale of the frequency content to have energy
    # of M^2.
    acf_fft = fftn(autocorrelation).real
    eneregy_const = M^2 / np.sum(acf_fft)
    autocorrelation = ifftshift(np.sqrt(acf_fft * eneregy_const))

    window = np.hanning(M)

    noise_matrices = np.random.normal((K,M,M))
    corrupted_acf = ifftn(fftn(noise_matrices)*acf_fft) # Convolving with noise
    noise_samples = corrupted_acf[:,1:N,1:N].real
    P = fftn(corrupted_acf * window)
    P = np.mean(np.abs(P)**2, dim=0)
    P /= np.norm(P)

    # The radial Fourier transform of the 3D ACF is:
    # sqrt(8*omega_r/pi)  * 1 / (T**3 * (1 + (omega_r/T)**2)**2)
    # One can derive this using Hankel transform
    omega = 2*np.pi / M 
    omega_x, omega_y, omega_z = omega*i, omega*j, omega*k
    c = 2*np.pi * eneregy_const * T
    S = np.zeros_like(P)

    # To account for the slow decay of the autocorrelation we use the Poisson
    # summation formula. We let M be the sampling interval and the DFT of the ACF
    # by summing samples of the true fourier transform at the points 2*pi*k/M.
    for l in range(-100,100):
        for r in range(-100,100):
            omega_r = (omega_x + l*M)**2 + (omega_y + r*M)**2 
            S+= np.sqrt((8*omega_r)/np.pi)*(1 / (T ** 3 * (1 + (omega_r/T)**2))**2)
    
    S /= np.norm(S)
    return S
    




