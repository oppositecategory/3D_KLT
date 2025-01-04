import numpy as np 
from numpy.fft import fftn, ifftn,fftshift,ifftshift

import jax
import jax.numpy as jnp

#import numba

from utils import bsearch, create_gaussian_window

def estimate_isotropic_powerspectrum_3D(tomogram, ids,max_d):
    n = tomogram.shape[0]
    if max_d >= n:
        max_d = n -1 

    # We estimate R(k) which is the isotropic 1-D autocorrelation function
    r, distances, _ = estimate_isotropic_autocorrelation_1D(tomogram, ids,max_d)

    # using the 1D ACF we create a 3D ACF by assiging (i,j,k) position with R(i**2 + j ** 2 + k**2)
    # note that this value is averaged over all k1,k2,3 such that their norm is equal to the position.
    r3 = autocorr_3d(max_d, distances,r,n)

    g_window = create_gaussian_window(n,max_d)

    # Estimate the 3D power spectrum by Wiener-Khinchin theorem
    p3 = fftshift(fftn(ifftshift(r3*g_window))).real

    # energy is equal to the total average energy of the samples
    energy = np.sum(np.square(tomogram[ids] - np.mean(tomogram[ids])))
    mean_energy = energy / len(ids[0])

    # Normalize the 3D powerspectrum such that its mean energy is preserved and is equal to mean energy
    # NOTE: p3 is already in units of energy and so the total energy is given by sum(p3) and not norm(p3)
    p3 = (p3/p3.sum())*mean_energy * p3.size

    p3 = np.where(p3 < 0, 0, p3) # Truncate negative values by 0 
    return p3


def estimate_isotropic_autocorrelation_1D(patch, max_d, ids):
    n = patch.shape[0]
    
    dists = np.array([[[i**2 + j ** 2 + k **2 for i in range(max_d+1)] for j in range(max_d+1)] for k in range(max_d+1)])
    dsquare = np.sort(np.unique(dists[dists <= max_d ** 2]))
   
    # A distance map such that i,j,k holds the index in dsquarae
    # of distance i**2 + j**2 + k**2
    dist_map = distance_map(max_d, dsquare,dists.shape)
    valid_dists = np.where(dists != -1) # Relevant indicies
    
    # An efficient way to compute the number of terms k1,k2,k3 
    # such that k1**2 + k2 ** 2 + k3 ** 2 = d 
    # we compute the auto-correlation of the 1-constant signal and hence we count the terms
    c = np.zeros(max_d) 
    mask = np.zeros((n,n,n))
    mask[ids] = 1 
    tmp = np.zeros((2*n+1,2*n+1,2*n+1))
    tmp[:n,:n,:n] = mask 
    tmp_fft = np.fft.fftn(tmp)
    c = np.fft.ifftn(tmp_fft*np.conj(tmp_fft))
    c = c[:max_d+1,:max_d+1,:max_d+1]
    c = np.round(c.real).astype('int')

    patch_fft = jnp.zeros((2*n+1,2*n+1,2*n+1),dtype=jnp.complex64)
    patch_fft = patch_fft.at[ids].set(patch[ids])
    ACF = np.array(estimate_ACF(patch_fft)).real

    r,cnt = turn_ACF_isotropic(ACF, dist_map, valid_dists,dsquare.shape[0], c)
    idx = np.where(r == 0)
    dsquare[idx] = 0
    return r, dsquare, cnt

#@numba.jit(nopython=True)
def autocorr_3d(max_d,dsquare,r,n):
    r3 = np.zeros((2*n-1,2*n-1,2*n-1),dtype=np.float64)
    for i in range(-max_d,max_d):
        for j in range(-max_d,max_d):
            for k in range(-max_d,max_d):
                d = i ** 2 + j ** 2 + k ** 2
                if d <= max_d ** 2:
                    idx, _ = bsearch(dsquare, d * (1 - 1e-13), d * (1 + 1e-13))
                    r3[i + n - 1, j + n - 1,k+n-1] = r[int(idx) - 1] 
    return r3

#@numba.jit(nopython=True)
def distance_map(max_d, dsquare, dists_shape):
    dist_map = np.zeros(dists_shape)
    for i in range(max_d + 1):
        for j in range(max_d + 1):
            d = i ** 2 + j ** 2
            if d <= max_d ** 2:
                idx, _ = bsearch(dsquare, d - 1e-13, d + 1e-13)
                dist_map[i, j] = idx
    dist_map = dist_map.astype(np.int32) - 1
    return dist_map

@jax.jit
def estimate_ACF(patch):
    # Estiming the auto-correlation function by compting the FFT of the power spectrum
    patch_fft = jnp.fft.fftn(patch)
    ACF = jnp.fft.ifftn(patch_fft * jnp.conj(patch_fft))
    return ACF 
    
#@numba.jit
def turn_ACF_isotropic(ACF,dist_map,valid_dists,num_dists,dists_counts):
    r = np.zeros(num_dists)
    corrs_count = np.zeros(num_dists)
    for d in zip(*valid_dists):
        id = dist_map[d]
        r[id] += ACF[d]
        corrs_count[id] += dists_counts[d]
    idx = np.where(corrs_count != 0)[0]
    r[idx] = (r[idx] / corrs_count[idx])
    return r,corrs_count[idx]


max_d = 10
X = np.random.normal(size=(10,10,10))
ids = np.array([[[i**2 + j ** 2 + k**2 for i in range(10)] for j in range(10)] for k in range(10)])
ids = np.where(ids<max_d)
estimate_isotropic_autocorrelation_1D(X,max_d, ids)