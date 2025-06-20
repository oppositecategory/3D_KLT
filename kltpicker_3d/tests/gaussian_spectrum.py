import jax 
#jax.config.update("jax_platform_name", "cpu")

from kltpicker_3d.spectral_estimation import estimate_isotropic_powerspectrum_tensor, cfftn

import numpy as np 
import jax.numpy as jnp


def generate_gaussian_spectrum(N,K):
    T = -np.log(1e-15)/30
    M = 2*N - 1 

    sigma = 1 
    w = 2*5*sigma 
    w0 = w / M

    grid = np.arange(-(N-1),N)
    i,j,k  = np.meshgrid(grid,grid,grid)
    omega_x, omega_y, omega_z = i*w0, j*w0, k*w0
    omega_r = np.sqrt(omega_x ** 2 + omega_y ** 2 + omega_z ** 2)
    H = (1/np.sqrt(2*np.pi * sigma ** 2)) * np.exp(- omega_r ** 2 / (2*sigma ** 2)) * (1 + 0.1 * np.cos(10*omega_r))
    C = M ** 3 / H.sum()
    normalized_H = np.fft.ifftshift(np.sqrt(H * C))

    window = np.bartlett(M)
    samples = np.zeros((K,N,N,N))
    spectrum = np.zeros((K, M,M,M))
    for k in range(K):
        noise = np.random.normal(size=(M,M,M))
        sample = np.fft.ifftn(np.fft.fftn(noise) * normalized_H)
        samples[k,...] = sample[:N,:N,:N]
        spectrum[k,...] = cfftn(sample * window)

    spectrum = np.mean(np.abs(spectrum),axis=0)
    spectrum /= np.linalg.norm(spectrum) 

    samples = samples.real 
    S = H / np.linalg.norm(H)
    return samples, S, spectrum 

def benchmark_gaussian_spectrum(N,K):
    samples, true_spectrum, avg_spectrum = generate_gaussian_spectrum(N,K)

    max_d = int(np.floor(N/2))
    noises = jnp.array(samples)
    p3 = estimate_isotropic_powerspectrum_tensor(noises,max_d)
    p3 = p3/jnp.linalg.norm(p3)
    return p3, true_spectrum, avg_spectrum
