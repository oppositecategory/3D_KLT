import numpy as np 
from numpy.fft import fftn, ifftn,fftshift,ifftshift

import jax 
import jax.numpy as jnp 


from utils import jitted_binarysearch,create_gaussian_window

def estimate_isotropic_powerspectrum_tensor(tomogram, ids,max_d):
    n = tomogram.shape[0]
    if max_d >= n:
        max_d = n -1 

    r, distances, _ = estimate_isotropic_autocorrelation_vector(tomogram, ids,max_d)

    # Create a 3D autocorrelation tensor
    r3 = jnp.zeros((2*n-1,2*n-1,2*n-1),dtype=jnp.float64)

    grid = jnp.arange(-max_d,max_d)
    i,j,k = jnp.meshgrid(grid,grid,grid)
    d = (i ** 2 + j**2 + k**2)

    mask = jnp.where(d <= max_d ** 2)
    mask,idx = jitted_binarysearch(distances, d)
    idx = jnp.clip(idx,0,distances.shape[0]-1)

    r3 = r3.at[mask].set(r[idx[mask]])

    # A gaussian window to truncate the Fourier transform 
    gaussian_window = create_gaussian_window(n,max_d)

    # Estimate the 3D power spectrum by Wiener-Khinchin theorem
    p3 = fftshift(fftn(ifftshift(r3*gaussian_window))).real

    # energy is equal to the total average energy of the samples
    energy = jnp.sum(jnp.square(tomogram[ids] - np.mean(tomogram[ids])))
    mean_energy = energy / len(ids[0])

    # Normalize the 3D power spectrum to preserve mean energy
    p3 = (p3/p3.sum())*mean_energy * p3.size

    p3 = jnp.where(p3 < 0, 0, p3) # Truncate negative values by 0 
    return p3


def estimate_isotropic_autocorrelation_vector(tomogram, max_d, ids):
    n = tomogram.shape[0]

    grid = jnp.arange(max_d+1)
    i,j,k = jnp.meshgrid(grid,grid,grid)
    d = i**2 + j **2 + k**2
    valid_dists = jnp.where(d <= max_d ** 2)
    dists = jnp.sort(jnp.unique(d[d <= max_d ** 2]))
   
    # A distance map such that i,j,k holds the index in dsquarae
    # of distance i**2 + j**2 + k**2
    idx = jitted_binarysearch(dists,max_d)
    dist_map = jnp.zeros(dists.shape)
    dist_map = dist_map.at[valid_dists].set(idx[valid_dists])


    # compute the ACF of the 1-constant signal to count number of k1,k2,k3
    # such that k1**2 + k2 ** 2 + k3 ** 2 = d for given d
    c = jnp.zeros(max_d) 
    mask = jnp.zeros((n,n,n))
    mask = mask.at[ids].set(1) 
    tmp = jnp.zeros((2*n+1,2*n+1,2*n+1))
    tmp.at[:n,:n,:n].set(mask) 
    c_padded = calculate_autocorrelation(tmp)
    c = c.set(c_padded[:max_d+1,:max_d+1,:max_d+1])
    c = jnp.round(c.real).astype('int')

    tomogram_fft = jnp.zeros((2*n+1,2*n+1,2*n+1),dtype=jnp.complex64)
    tomogram_fft = tomogram_fft.at[ids].set(tomogram[ids])
    tomogram_acf = calculate_autocorrelation(tomogram_fft).real

    init = jnp.zeros((2,dists.shape[0]))
    r,cnt = average_acf_radially(tomogram_acf, dist_map, valid_dists,c,init)
    idx = jnp.where(cnt != 0)[0]
    
    r = r.at[idx].set(r[idx] / cnt[idx])
    idx = jnp.where(r == 0)
    dists.at[idx].set(0)
    return r, dists, cnt

@jax.jit
def calculate_autocorrelation(patch):
    # Estiming the autocorrelation by Wiener-Khinchin theorem
    patch_fft = fftn(patch)
    acf = ifftn(patch_fft * jnp.conj(patch_fft))
    return acf 

@jax.jit
def average_acf_radially(acf,dist_map,valid_dists,dists_counts,init):
    def scan_fn(carry, idx):
        i,j,k = idx
        d = dist_map[i,j,k]
        carry = carry.at[0,d].add(acf[i,j,k])
        carry = carry.at[1,d].add(dists_counts[i,j,k])
        return (carry,None) 

    vv = jnp.stack(valid_dists,axis=-1)
    v, _ = jax.lax.scan(scan_fn,init,vv)
    return v[0,...],v[1,...]


# max_d = 10
# X = np.random.normal(size=(10,10,10))
# ids = np.array([[[i**2 + j ** 2 + k**2 for i in range(10)] for j in range(10)] for k in range(10)])
# ids = np.where(ids<max_d)
# estimate_isotropic_autocorrelation_1D(X,max_d, ids)