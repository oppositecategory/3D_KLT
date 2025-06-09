import numpy as np 
from functools import partial

from skimage.filters import window 

import jax 
import jax.numpy as jnp 
from jax.numpy.fft import fftn,ifftn,fftshift,ifftshift

from kltpicker_3d.utils import jitted_binarysearch,gaussian_window

def estimate_isotropic_powerspectrum_tensor(tomograms,max_d):
    """ 
    Estimates the isotropic power spectrum of a given 3D tomogram using the autocorrelation of the tomogram.

    Args:
        tomograms: 4-dimensional tensor containing samples of noisy tomograms.
        max_d: the maximum distance for the isotogrpic powerspectrum

    Returns:
        p3: 3D Power Spectrum tensor
    """
    K,N,_,_ = tomograms.shape

    if max_d >= N:
        max_d = N - 1 

    r3 = estimate_isotropic_autocorrelation(tomograms,max_d)

    # A gaussian window to truncate the Fourier transform 
    # The constatn 3 comes to "shrink" the width of the transform.
    w = jnp.array(window(('gaussian', max_d), (2*N-1,2*N-1,2*N-1)))


    # Estimate the 3D power spectrum by Wiener-Khinchin theorem
    p3 = cfftn(r3*w).real
    p3 = jnp.where(p3 < 0, 0, p3)

    # Energy is equal to the total average energy of the samples
    E = 0
    for k in range(K):
        sample = tomograms[k]
        E += np.sum((sample - jnp.mean(sample)) ** 2)

    mean_energy = E/ (K * N ** 3)

    # Normalize the 3D power spectrum to preserve mean energy
    p3 = (p3/p3.sum())*mean_energy * p3.size
    p3 = jnp.where(p3 < 0, 0, p3) # Truncate negative values by 0 
    return p3


def estimate_isotropic_autocorrelation(tomograms, max_d):
    K,N,_,_ = tomograms.shape

    grid = jnp.arange(max_d+1)
    i,j,k = jnp.meshgrid(grid,grid,grid)
    d = i**2 + j**2 + k**2
    valid_dists = jnp.where(d <= max_d ** 2)
    dists = jnp.sort(jnp.unique(d[d <= max_d ** 2]))
   
    # A distance map such that i,j,k holds the index in dists
    # of distance i**2 + j**2 + k**2
    idx = jitted_binarysearch(dists,d)
    dist_map = jnp.zeros(d.shape, dtype=jnp.int32)
    dist_map = dist_map.at[valid_dists].set(idx[valid_dists])

    # compute the ACF of the 1-constant siganl to count number of k1,k2,k3
    # such that k1**2 + k2 ** 2 + k3 ** 2 = d 
    mask = jnp.ones((N,N,N)) 
    tmp = jnp.zeros((2*N-1,2*N-1,2*N-1))
    tmp = tmp.at[:N,:N,:N].set(mask) 
    c_padded = calculate_autocorrelation(tmp)
    c = c_padded[:max_d+1,:max_d+1,:max_d+1]
    c = jnp.round(c.real).astype(jnp.float32)

    corrs = jnp.zeros_like(dists)
    corrcount = jnp.zeros_like(dists)
    for k in range(K):
        sample = tomograms[k]

        tomogram_fft = jnp.zeros((2*N+1,2*N+1,2*N+1),dtype=jnp.complex64)
        tomogram_fft = tomogram_fft.at[:N,:N,:N].set(sample)
        tomogram_acf = calculate_autocorrelation(tomogram_fft).real
        tomogram_acf = tomogram_acf[:max_d+1, :max_d+1,:max_d+1]

        init = jnp.zeros((2,dists.shape[0]))
        r,cnt = accumulate_acf_radially(tomogram_acf, dist_map, valid_dists,c,init)
        corrs += r 
        corrcount += cnt 

    idx1 = jnp.where(corrcount != 0)[0]
    result = corrs.at[idx1].set(corrs[idx1] / corrcount[idx1])

    r3 = create_autocorrelation_tensor(result, dists, N, max_d)
    return r3

@partial(jax.jit, static_argnames=['N','max_d'])
def create_autocorrelation_tensor(r, dists, N, max_d):
    r3 = jnp.zeros((2*N-1,2*N-1,2*N-1))
    for i in range(-max_d, max_d):
        for j in range(-max_d, max_d):
            for k in range(-max_d, max_d):
                d = i**2 + j**2 + k**2 
                if d <= max_d ** 2:
                    id = jnp.searchsorted(dists,d,side='left')
                    r3 = r3.at[(N-1) + i, (N-1) +j, (N-1)+k].set(r[id])
    return r3

@jax.jit
def cfftn(x):
    return jnp.fft.fftshift(jnp.fft.fftn(jnp.fft.ifftshift(x)))

@jax.jit
def calculate_autocorrelation(patch):
    # Estimating the autocorrelation by Wiener-Khinchin theorem
    patch_fft = fftn(patch)
    acf = ifftn(patch_fft * jnp.conj(patch_fft))
    return acf 

@jax.jit
def accumulate_acf_radially(acf,dist_map,valid_dists,dists_counts,init):
    """ The function transforms the autocorrelation function into it's
        isotoropic form by averaging over all possible distances of given distance.
    """
    def scan_fn(carry, idx):
        i,j,k = idx
        d = dist_map[i,j,k]
        carry = carry.at[0,d].add(acf[i,j,k])
        carry = carry.at[1,d].add(dists_counts[i,j,k])
        return (carry,None) 

    vv = jnp.stack(valid_dists,axis=-1)
    v, _ = jax.lax.scan(scan_fn,init,vv)
    return v[0,...],v[1,...]

