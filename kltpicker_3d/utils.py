import jax 
import jax.numpy as jnp

import scipy
import numpy as np

from functools import partial

def radial_average_jax(X, shell_ids, counts, nbins):
    x = X.ravel()
    ids = shell_ids.ravel()
    mask = ids >= 0
    sums = jnp.bincount(ids[mask], weights=x[mask], length=nbins)
    return sums / counts

def prewhiten_patch(patch, noise_psd):
    """ Pre-whitening a patch using approximation of the noise RPSD.

        Args:
            patch: sub-tomogram of size LxLxL 
            noise_psd: tensor of size MxMxM containing the noise RPSD

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

def generate_uniform_radial_sampling_points(L, r_max, nbins=None):
    k = np.fft.fftshift(np.fft.fftfreq(L, d=1.0 / (2 * r_max)))
    kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
    r = np.sqrt(kx**2 + ky**2 + kz**2)
    eps = np.finfo(float).eps

    r_edge = r.max()
    df = 2.0 * r_max / L

    if nbins is None:
        nbins = int(max(1, np.floor(r_edge / df)))

    r_edges = np.linspace(0.0, r_edge, nbins + 1)
    uniform_points = 0.5 * (r_edges[:-1] + r_edges[1:])

    shell_ids = np.digitize(r, r_edges[1:-1], right=False).astype(np.int32)
    shell_ids = np.clip(shell_ids, 0, nbins - 1)

    counts = np.bincount(shell_ids.ravel(), minlength=nbins)
    return uniform_points, shell_ids, counts

def trigonometric_interpolation(x,y,z):
    n = x.shape[0]
    
    scale = (x[1] - x[0]) * n / 2 
    x_scaled = (x / scale) * jnp.pi / 2 
    z_scaled = (z / scale) * jnp.pi / 2

    delta = z_scaled[:, None] - x_scaled[None, :]
    # We take n to be only even 
    M = jnp.sin(n*delta) / (n *jnp.sin(delta))
    M = M.at[jnp.isclose(delta,0)].set(1.0)

    p = M @ y 
    return p

def generate_legendre_points(n, a, b):
    """
    Get n leggauss points in interval [a, b]
    Parameters
    ----------
    n : int
        Number of points.
    a : float
        Interval starting point.
    b : float
        Interval end point.
    Returns
    -------
    x : numpy.ndarray
        Sample points.
    w : numpy.ndarray
        Weights.
    """
    x1, w = scipy.special.roots_legendre(n)
    m = (b - a) / 2
    c = (a + b) / 2
    x = m * x1 + c
    w = m * w
    x = np.flipud(x)
    return x, w
