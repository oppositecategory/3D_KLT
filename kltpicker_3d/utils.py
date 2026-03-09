import jax 
import jax.numpy as jnp

import scipy
import numpy as np

from functools import partial

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


def generate_uniform_radial_sampling_points(L,r_max, nbins=None):
    # Frequency lattice in cycles/pixel
    k = np.fft.fftshift(np.fft.fftfreq(L, d=1.0/(2*r_max)))      
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    r = np.sqrt(kx**2 + ky**2 + kz**2)  
    eps = np.finfo(float).eps

    mask = (r <= r_max)
    r_inside = r[mask]
    r_edge = r_inside.max()
    df = 2.0 * r_max / L
    # Number of bins
    if nbins is None:
        nbins = int(max(1, np.floor(r_edge / df)))

    # Uniform-in-radius edges & centers on [0, r_max]
    r_edges = np.linspace(0.0, r_edge, nbins + 1)
    uniform_points = 0.5 * (r_edges[:-1] + r_edges[1:])

    # Indices per shell (last bin inclusive on upper edge)
    bins = []
    for j in range(nbins):
        lo, hi = r_edges[j], r_edges[j+1]
        if j < nbins - 1:
            mask = (r >= lo) & (r < hi)
        else:
            mask = (r >= lo) & (r <= hi + eps)
        bins.append(np.where(mask))

    return uniform_points, bins

def picking_from_scoring_vol_3d(score_vol, offsets, settings):
    nx, ny, nz = score_vol.shape
    num_particles = settings.num_particles
    max_iter = settings.max_iter
    threshold = settings.threshold

    r_del = settings.particle_diameter // 2
    r2 = r_del ** 2

    # Grids for spherical suppression mask
    gx = np.arange(nx)[:, None, None]
    gy = np.arange(ny)[None, :, None]
    gz = np.arange(nz)[None, None, :]

    log_max = np.max(score_vol)
    eps = 1e-12

    scoring = score_vol.copy()
    particle_list = []

    num_limit = np.inf if num_particles == -1 else int(num_particles)
    num_picked = 0

    while num_picked < min(max_iter, num_limit):
        flat_idx = np.argmax(scoring)
        p_max = scoring.flat[flat_idx]

        if not (p_max > threshold):
            break

        ix, iy, iz = np.unravel_index(flat_idx, scoring.shape)

        # Convert valid-score index -> tomogram center coordinate
        cx = ix + offsets[0]
        cy = iy + offsets[1]
        cz = iz + offsets[2]

        particle_list.append([cx, cy, cz, p_max / (log_max + eps)])

        # 3D spherical NMS
        mask = (gx - ix) ** 2 + (gy - iy) ** 2 + (gz - iz) ** 2 <= r2
        scoring[mask] = -np.inf

        num_picked += 1

    particle_coords = np.array(particle_list, dtype=np.float64) if particle_list else np.zeros((0, 4))
    num_picked_particles = particle_coords.shape[0]

    return num_picked_particles, particle_coords

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

def radial_average(X, bins, n):
    S = np.zeros(n)
    for j in range(n):
        bin_len = bins[j][0].size
        S[j] += np.sum(X[bins[j]])/bin_len
    return S
