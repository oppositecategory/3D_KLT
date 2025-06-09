import jax 
jax.config.update("jax_platform_name", "cpu")

from kltpicker_3d.spectral_estimation import estimate_isotropic_powerspectrum_tensor, cfftn

import numpy as np 
import jax.numpy as jnp


def generate_exponential_autocorrelation(N, K,T=1):
    """
    Generates noisy tensors with autocorrelation (ACF) of the form exp(-T*r).
    The function returns the samples along with a reference power spectrum for benchmark.

    Args:
        N: size of noisy tomograms (NxNxN)
        K: number of noise tomograms to generate 
        T: the decay rate of the ACF

    Returns: 
        noises: K 3D tensors of size NxNxN with expoenential autocorrelation.
        spectrum: power spectrum of the noise estimated using squared absolute value of the generated noise.
        S: samples of the analytical Fourier transform at the frequencies 2*pi/(M*T)*j for j=-N..N
    """
    M = 2*N - 1 
    T = -np.log(1e-15)/30
    grid = np.arange(-(N-1),N)
    i,j,k = np.meshgrid(grid,grid,grid)
    acf = np.exp(-T * np.sqrt(i**2 + j**2 + k**2))

    H = cfftn(acf).real
    C = M**3 / H.sum()
    normalized_H = jnp.fft.ifftshift(jnp.sqrt(H * C)) # The power spectrum we want 

    window = np.hanning(M)

    samples = np.zeros((K,N,N,N))
    spectrum = np.zeros((K,M,M,M))
    for i in range(K):
        gaussian_noise = np.random.normal(size=(M,M,M))
        noise = jnp.fft.ifftn(jnp.fft.fftn(gaussian_noise) * normalized_H)
        samples[i] = noise[:N,:N,:N].real
        spectrum[i] = cfftn(noise * window)

    spectrum = np.mean(np.abs(spectrum),axis=0)
    spectrum /= np.linalg.norm(spectrum)
    

    # The radial Fourier transform of the 3D ACF is:
    # 2 * np.sqrt(2) * (2*np.pi)**(3/2) / (T**3 * (1 + (omega_r/T)**2)**2)
    # One can derive this using Hankel transform.
    # To account for slow convergence we return Poisson summation of the transform.
    w = 2*np.pi/M
    grid = np.arange(-(N-1),N)
    i,j,k  = np.meshgrid(grid,grid,grid)
    omega_x, omega_y, omega_z = w*i, w*j, w*k
    poisson_fft = jnp.zeros((M,M,M))
    C = 8 * np.pi ** (3/2) / (T**3)
    for j1 in range(-10,11):
        for j2 in range(-10,11):
            for j3 in range(-10,11):
                omega_r_squared = (omega_x + j1*2*np.pi)**2 + (omega_y + j2*2*np.pi)**2 + (omega_z + j3*2*np.pi)**2
                poisson_fft += C/(1+ omega_r_squared/(T**2))**2
    poisson_fft /= jnp.linalg.norm(poisson_fft)
    return samples, spectrum, poisson_fft

def benchmark_exp_autocorrelation(N,K):
    noises, spectrum, poisson_fft = generate_exponential_autocorrelation(N,K,T=1)

    max_d = int(np.floor(N/3))
    noises, spectrum, poisson_fft = jnp.array(noises), jnp.array(spectrum), jnp.array(poisson_fft)

    p3 = estimate_isotropic_powerspectrum_tensor(noises,max_d)
    p3 = p3/jnp.linalg.norm(p3)
    return p3, spectrum,poisson_fft


