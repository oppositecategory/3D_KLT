import jax 

import numpy as np
import jax.numpy as jnp

from functools import partial

def prewhiten_patch(patch, noise_psd):
    """ Pre-whitening a patch using approximation of the noise RPSD.

        Args:
            patch: sub-tomogram of size LxLxL 
            noise_psd: tensor of size MxMxM for M=2*L-1 containing the noise    RPSD

        returns:
            p3: sub-tomogram after cleaning the approximated noise from it's spectrum
    """
    L,_,_ = patch.shape
    M,_,_ = noise_psd.shape
    midpoint = M//2

    start = midpoint - L//2
    end = midpoint + L//2 

    filter = jnp.sqrt(noise_psd)
    filter /= jnp.linalg.norm(filter)
    
    #Symmetrize the PSD across each axis
    filter = (filter + jnp.flip(filter,axis=0))/2
    filter = (filter + jnp.flip(filter,axis=1))/2
    filter = (filter + jnp.flip(filter,axis=2))/2

    mask = filter > 1e-14
    filter = jnp.where(mask, 1 / filter,0)

    padded = jnp.zeros_like(noise_psd)
    padded = padded.at[start:end, start:end, start:end].set(patch)

    fp = jnp.fft.fftshift(jnp.fft.fftn(jnp.fft.ifftshift(padded)))
    fp *= filter 
    pp2 = jnp.fft.fftshift(jnp.fft.ifftn(jnp.fft.ifftshift(fp)))
    p2 = pp2[start:end,start:end,start:end].real 
    return p2 

def generate_uniform_radial_sampling_points(L):
    """
        For a 3-dimensional signal of size LxLxL the function generates 
        equidistance sampling points in radial basis for later interpolation.

        Note that as we can't use exact sampled magnitudes we use bins
        instead of indices that fall between two consecutive sampling points.
        Then we treat the corresponding sampling point as the average of the bin's endpoints.

        args:
            L: size of 3-dimensional signal
        
        returns:
            uniform_points: uniform sampling points in radial basis
            bins: bins[j] contains indices such that their magnitude is bounded by <= r_bins[j+1] and >= r_bins[j] 
    """
    N = 2*L-1
    grid = 2/(N-1)*np.arange(N) - 1
    i,j,k= np.meshgrid(grid,grid, grid)

    r = (i**2 + j** 2 + k**2) ** 0.5 
    spacing = 1 /(L-1)

    # Uniform spacings for creating equdisant radial sampling points
    r_bins = np.linspace(-spacing/2, spacing/2 + 1, L+1 )

    # Approximated uniform sampling points by averaging bins end-points
    uniform_points = (r_bins[0:-1] + r_bins[1:])/2 

    bins = []
    for j in range(L - 1):
        bins.append(np.where(np.logical_and(r >= r_bins[j], 
                                        r < r_bins[j + 1])))
    
    bins.append(np.where(np.logical_and(r >= r_bins[L - 1], 
                                    r <= 1)))
    return uniform_points, bins

def trigonometric_interpolation(x,y,z):
    """ 
    Vectorized trigonometric interpolation for equidistant points. 

    args:
        - x: equidistant sample points 
        - y: function values at sampled points
        - z: evaluation points for interpolation
    """
    n = x.shape[0]
    
    scale = (x[1] - x[0]) * n / 2 
    x_scaled = (x / scale) * np.pi / 2 
    z_scaled = (z / scale) * np.pi / 2

    delta = z_scaled[:, None] - x_scaled[None, :]

    if n % 2 == 0:
        M = np.sin(n*delta) / (n *np.sin(delta))
    else:
        M = np.sin(n*delta)/ (n*np.tan(delta))
    M[np.isclose(delta,0)] = 1

    p = M @ y 
    return p


def radial_average(X, bins, n):
    S = np.zeros(n)
    for j in range(n):
        bin_len = bins[j][0].size
        S[j] += np.sum(X[bins[j]])/bin_len
    return S
