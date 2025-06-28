import numpy as np 
import jax
import jax.numpy as jnp 

from kltpicker_3d.alt_least_squares import alternating_least_squares_solver
from kltpicker_3d.spectral_estimation import estimate_isotropic_powerspectrum_tensor
from kltpicker_3d.utils import * 


vect_spectrum_estimation = jax.vmap(estimate_isotropic_powerspectrum_tensor,
                                    in_axes=(0,None))

vect_prewhite_patch = jax.vmap(prewhiten_patch,
                               in_axes=(0,None))

def factorize_RPSD(tomograms):
    K,N,_,_ = tomograms.shape
    max_d = int(np.floor(N/3))

    _, bins = generate_uniform_radial_sampling_points(N)

    tomograms = tomograms- np.mean(tomograms, axis=(1,2,3)).reshape(-1,1,1,1)
    psds = vect_spectrum_estimation(tomograms,max_d)

    rblocks = np.zeros((K,N))
    for k in range(N):
        rblocks[k] = radial_average(psds[k],bins,N)
    rblocks = jnp.array(rblocks)
    
    factorization = alternating_least_squares_solver(rblocks,200,1e-2)
    return factorization

def prewhiten_tomogram(tomograms, factorization):
    _,N,_,_ = tomograms.shape
    
    uniform_points, _ = generate_uniform_radial_sampling_points(N)
    noise_rpsd = factorization.v

    # The extracted particle's RPSD and noise's RPSD are 1-dimensional.
    # However we need to prewhiten the 3D tomograms and 
    # for that we need to interpolate the noise RPSD into 3D tensor.
    grid = jnp.arange(-(N-1), N) * jnp.pi / N
    i,j,k = jnp.meshgrid(grid,grid,grid)
    r_matrix = jnp.sqrt(i**2 + j**2 + k**2)
    magnitudes, idx = jnp.unique(r_matrix, return_inverse=True)
    nodes = magnitudes[magnitudes < uniform_points[-1]*jnp.pi]

    interpolated_noise_rpsd = trigonometric_interpolation(uniform_points*np.pi, noise_rpsd, nodes)
    noise_rpsd_mat = jnp.pad(interpolated_noise_rpsd, 
                      (0,
                       magnitudes.size - interpolated_noise_rpsd.size),
                      'constant',
                      constant_values=interpolated_noise_rpsd[-1])
    noise_rpsd_mat = jnp.reshape(noise_rpsd_mat[idx], [grid.size, grid.size, grid.size])
    
    whitened_tomograms = vect_prewhite_patch(tomograms,noise_rpsd_mat)
    whitened_tomograms -= jnp.mean(whitened_tomograms)
    whitened_tomograms /= jnp.linalg.norm(whitened_tomograms)
    return whitened_tomograms

def process_tomogram(tomograms):
    factorization = factorize_RPSD(tomograms)
    particle_psd, noise_psd = factorization.gamma, factorization.v
    whitened_tomograms = prewhiten_tomogram(tomograms, noise_psd)
    factorization = factorize_RPSD(whitened_tomograms)


    