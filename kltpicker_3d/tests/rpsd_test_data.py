import numpy as np


def generate_rpsd_data(K,N, std):
    K = 200
    N = 32
    M = 2*N - 1 
    max_d = int(np.floor(N/3))

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
    alphas = np.random.beta(2,5,size=(K)) 
    # Asymetric distriution with 0-skewness
    for k in range(K):
        noise = np.random.normal(size=(M,M,M)) 
        additive_noise = np.random.normal(scale=std,size=(M,M,M))
        sample = np.fft.ifftn(normalized_H*np.fft.fftn(noise))
        sample = alphas[k]*sample + additive_noise
        samples[k] = sample[:N,:N,:N]

    samples = samples.real 
    return H, samples