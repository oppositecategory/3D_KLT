from functools import partial

from skimage.filters import window 

import jax 
import jax.numpy as jnp 
from jax.numpy.fft import fftn,ifftn,fftshift,ifftshift


def estimate_isotropic_powerspectrum_tensor(tomogram,max_d):
    """ 
    Estimates the isotropic power spectrum of a given 3D tomogram using the autocorrelation of the tomogram.

    Args:
        tomogram: 3-dimensional tensor containing sample of noisy tomogram.
        max_d: the maximum distance for the isotogrpic powerspectrum

    Returns:
        p3: 3D Power Spectrum tensor
    """
    N,_,_ = tomogram.shape

    if max_d >= N:
        max_d = N - 1 

    r3 = estimate_isotropic_autocorrelation(tomogram,max_d)
    w = jnp.array(window(('gaussian', max_d), (2*N-1,2*N-1,2*N-1)))
    p3 = cfftn(r3*w).real
    p3 = jnp.where(p3 < 0, 0, p3)
    mean_energy = jnp.sum(jnp.square(tomogram - jnp.mean(tomogram))) / N**3
    
    # Normalize the 3D power spectrum to preserve mean energy
    p3 = (p3/p3.sum())*mean_energy * p3.size
    p3 = jnp.where(p3 < 0, 0, p3) 
    return p3

def estimate_isotropic_autocorrelation(tomogram, max_d):
    N,_,_ = tomogram.shape

    grid = jnp.arange(max_d+1)
    i,j,k = jnp.meshgrid(grid,grid,grid)
    d = i**2 + j**2 + k**2
    valid_dists = jnp.where(d <= max_d ** 2)
    dists = jnp.sort(jnp.unique(d[d <= max_d ** 2]))
   
    # A distance map such that i,j,k holds the index in dists
    # of distance i**2 + j**2 + k**2
    idx = jnp.searchsorted(dists,d,side='left')
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
    tomogram_fft = jnp.zeros((2*N+1,2*N+1,2*N+1),dtype=jnp.complex64)
    tomogram_fft = tomogram_fft.at[:N,:N,:N].set(tomogram)
    tomogram_acf = calculate_autocorrelation(tomogram_fft).real
    init = jnp.zeros((2,dists.shape[0]))
    r,cnt = accumulate_acf_radially(tomogram_acf, dist_map, valid_dists,c,init)
    nonzero_mask = (cnt != 0)
    result = jnp.where(nonzero_mask, r / cnt, 0.0)
    r3 = create_autocorrelation_tensor(result, dists, N, max_d)
    return r3


def create_autocorrelation_tensor(r, dists, N, max_d):
    grid = jnp.arange(-max_d, max_d)
    i, j, k = jnp.meshgrid(grid, grid, grid, indexing="ij")
    
    d = i**2 + j**2 + k**2
    mask = d <= max_d**2
    idx = jnp.searchsorted(dists, d, side="left")
    r3 = jnp.zeros((2*N-1, 2*N-1, 2*N-1))

    idx1= idx[mask]
    i1, j1, k1 = i[mask], j[mask],k[mask]
    values = r[idx1]

    x,y,z = (N - 1) + i1, (N-1) + j1, (N-1) + k1
    r3 = r3.at[x, y, z].set(values)
    return r3


@jax.jit
def cfftn(x):
    return fftshift(fftn(ifftshift(x)))

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
