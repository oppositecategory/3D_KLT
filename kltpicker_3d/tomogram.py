import numpy as np 
import scipy

import jax
import jax.numpy as jnp 

from kltpicker_3d.alt_least_squares import alternating_least_squares_solver
from kltpicker_3d.spectral_estimation import estimate_isotropic_powerspectrum_tensor
from kltpicker_3d.fredholm_solver import solve_radial_fredholm_equation,create_GPSF_templates
from kltpicker_3d.utils import * 

# Compiled functions 
vect_spectrum_estimation = jax.vmap(estimate_isotropic_powerspectrum_tensor,
                                    in_axes=(0,None))

class KLTParticleDetector3D:
    def __init__(self, 
                 tomogram,
                 particle_diameter: float,
                 mgscale: float,
                 bandlimit: float,
                 num_particles: int, 
                 legendre_order:int = 150,
                 threshold: float = 0,
                 max_iter: int = 500, 
                 max_order: int = 10):
        self.tomogram = tomogram 
        self.particle_diameter = particle_diameter 
        self.mgscale = mgscale
        self.max_order = max_order
        self.bandlimit = bandlimit

        patch_size = np.floor(0.8 * self.mgscale * self.particle_diameter)
        #patch_size = self.particle_diameter
        if np.mod(patch_size,2)  == 0:
            patch_size -= 1
        self.patch_size = patch_size 
        template_diameter = np.floor(0.4 * self.mgscale * particle_diameter)
        template_diameter = self.particle_diameter
        if np.mod(template_diameter,2) == 0:
            template_diameter -= 1 
        self.template_diameter = int(template_diameter)
        self.max_iter = max_iter 
        
        S = int(2*patch_size-1)
        uniform_points, bins = generate_uniform_radial_sampling_points(S, bandlimit)
        self.uniform_points = uniform_points
        self.bins = bins

        self.legendre_order = legendre_order
        self.num_particles = num_particles
        self.threshold = threshold

        self.eigvals = None 
        self.eiguncs = None 
    
    def process_tomogram(self):
        factorization, noise_var_approx = self.factorize_RPSD()
        particle_psd, noise_psd = factorization.gamma, factorization.v
        #whitened_tomograms = prewhiten_tomogram(tomogram, noise_psd)
        
        c = self.bandlimit 
        a = self.particle_diameter
        K = self.legendre_order
        X,w = scipy.special.roots_legendre(K)
        X_scaled = c/2*X + c/2
        Gx = trigonometric_interpolation(self.uniform_points, 
                                        particle_psd, 
                                        X_scaled)
        
        eigvals, eigfuncs = [],[]
        max_order = self.max_order
        for i in range(max_order):
            lambdas, funcs,W = solve_radial_fredholm_equation(Gx,i,a,c)
            eigvals.append(lambdas)
            eigfuncs.append(funcs)

        eigfuncs = np.array(eigfuncs).reshape(-1,K)
        eigvals = np.array(eigvals).reshape(-1)

        orders = np.tile(np.arange(max_order).reshape(-1, 1), (1, K)).reshape(-1)

        idx = np.argsort(eigvals)[::-1]
        orders = orders[idx]

        eigfuncs = eigfuncs[idx,:]
        eigvals = eigvals[idx]

        idx = np.where(eigvals > np.spacing(1))[0]
        eigvals = eigvals[idx]
        eigfuncs = eigfuncs[idx,:]
        orders = orders[idx]

        templates, eigvals = self.create_GPSF_templates(eigvals,
                                                        eigfuncs,
                                                        orders, 
                                                        Gx)

        num_detected,coords = self.detect_particles(templates, 
                                                    noise_var_approx)
        return num_detected,coords

    def factorize_RPSD(self):
        M = int(self.patch_size)
        bins = self.bins
        max_d = int(np.floor(0.3*M))

        micro_size = np.min(self.tomogram.shape)
        m = int(np.floor(micro_size/ M))

        t = self.tomogram[:m*M, :m*M, :m*M]
        # (m*M, m*M, m*M) -> (m, M, m, M, m, M) -> (m, m, m, M, M, M)
        patches = t.reshape(m, M, m, M, m, M).transpose(0, 2, 4, 1, 3, 5)
        # Flatten patch index to match your original (m**3, M, M, M)
        patches = patches.reshape(m**3, M, M, M)

        patches = patches - jnp.mean(patches, axis=(1,2,3)).reshape(-1,1,1,1)

        patches_var = jnp.var(patches,axis=(1,2,3))
        patches_var = patches_var.sort()
        noise_var_approx = jnp.mean(
            patches_var[:jnp.floor(0.25 * patches_var.size).astype('int')]
        )

        psds = vect_spectrum_estimation(patches,max_d)

        rblocks = np.array([radial_average(psds[k], bins, len(bins)) for k in range(patches.shape[0])])
        factorization = alternating_least_squares_solver(rblocks,self.max_iter,1e-4)
        return factorization,noise_var_approx

    def create_GPSF_templates(self,
                              eigvals,
                              eigfuncs,
                              orders,
                              G):
        """
            Generates 3D templates in Generalized Prolate Spherodial Function basis
            using radial solutions of KLT equations and their spectrum.

            args:
                orders: the N orders of the corresponding solutions
                G: particle function's radial PSD 
            
            returns:
                templates: 3D templates composed of radial and angular basis functions. 
                Shape of returned tensor: truncate_idx X max_N x (grid_shape)
                eigvals: spectrum of all eigenfunctions truncated at 99%
                orders: corresonding N order of each solution
        """
        a = self.particle_diameter
        c = self.bandlimit 
        template_size = self.template_diameter
        K = self.legendre_order 

        radmax = np.floor((template_size-1)/2)
        grid = np.arange(-radmax,radmax+1,1)
        X,Y,Z = np.meshgrid(grid,grid,grid)
        r_tensor = np.sqrt(X**2 + Y**2 + Z**2)
        rho_uniform, idx = np.unique(r_tensor,return_inverse=True)

        # Legendre roots for both integrals
        rho_leg, w = scipy.special.roots_legendre(K)
        rho_leg_a =  (a * 0.5) * rho_leg + a * 0.5 
        rho_leg_c =  (c * 0.5)* rho_leg + c * 0.5
        
        # Truncates the spectra at 99%
        eigval_cumsum = np.cumsum(eigvals / np.sum(eigvals))
        truncate_idx = (eigval_cumsum > 0.99).argmax() + 1

        eigfuncs = eigfuncs[:truncate_idx,...]
        eigvals = eigvals[:truncate_idx,...]
        orders = orders[:truncate_idx,...]

        self.eigfuncs = eigfuncs 
        self.eigvals = eigvals 

        # We interpolate the radial solutions into uniform radial basis
        # using the Fredholm equation (re-expressing new values of R_{N,m} using  
        # values of it at Legendre roots.
        r_grid_uni = np.outer(rho_uniform, rho_leg_c)
        r_grid_leg = np.outer(rho_leg_a, rho_leg_c)

        def Hn(x,N):
            return 4*np.pi * ((1j**N) * scipy.special.spherical_jn(N,x))
        
        max_N = int(orders.max()) + 1

        # Hn evaluated at multiples of uniform radial points in [0,a]
        Hn_uniform = np.array(
            [Hn(r_grid_uni,N) for N in range(max_N)]
        )

        # Hn evaluated at multiples Legendre roots in [0,a]
        Hn_leg = np.array(
            [Hn(r_grid_leg,N) for N in range(max_N)]
        )

        Hn_leg = Hn_leg[orders]
        Hn_uniform = Hn_uniform[orders]

        sgn = np.where(orders % 2 == 1, -1, 1)
        D = c * 0.5* w * G * (rho_leg_c**2)
        W = a * 0.5 * w* rho_leg_a**2

        H_right = sgn[:,None,None]* Hn_leg
        psi = (Hn_uniform * D[None,None,:]) @ H_right

        # Shape: truncate_idx X rho_uniform.length 
        eigfuncs_uniform = np.einsum('bik,k,bk->bi', psi,W,eigfuncs) / eigvals[:,None]
        radial_templates = eigfuncs_uniform[:,idx]

        # Spherical Harmonics 
        theta = np.arctan2(Y, X)
        phi = np.arctan2(Z,np.sqrt(X**2 + Y**2))

        sph_harm = np.zeros((max_N, max_N) + theta.shape, dtype=np.complex64)
        for n in range(max_N):
            for m in range(n+1):
                sph_harm[n,m] = scipy.special.sph_harm(m,n,theta,phi)
        
        templates = sph_harm[orders]*radial_templates[:,None,...]
        return templates, eigvals
    
    def detect_particles(self,
                         templates, 
                         noise_var_approx):
        """ GPU-Accelerated particle detection.
            Function apply FFT-based convolution to run each generated template-kernel across
            the whole 3D tomogram. 
        """
        M = int(self.patch_size)
        n_radial, n_harm, nx, ny, nz = templates.shape
        psi = templates.reshape(n_radial * n_harm, nx * ny * nz)
        eigvals_r = jnp.repeat(self.eigvals, n_harm)

        Q,R = jnp.linalg.qr(psi.T)
        H = R @ np.diag(eigvals_r) @ R.T + noise_var_approx * np.eye(R.shape[0])
        H_inv = jnp.linalg.inv(H)
        T = (1 / noise_var_approx) * np.eye(R.shape[0]) - H_inv 

        mu = jnp.linalg.slogdet((1/ noise_var_approx) * H)[1]

        D,P = jnp.linalg.eigh(T)
        D, P = D[::-1],P[:,::-1]

        B = Q @ P
        kernels = B.T.reshape(n_radial * n_harm, nx,ny,nz)

        x_num = self.tomogram.shape[0] - nx + 1
        y_num = self.tomogram.shape[1] - ny + 1
        z_num = self.tomogram.shape[2] - nz + 1
        score_mat = jnp.zeros((x_num, y_num, z_num), dtype=jnp.float64)

        for i in range(kernels.shape[0]):
            kernel = jnp.conj(np.flip(kernels[i], axis=(0,1,2)))
            response = jax.scipy.signal.fftconvolve(self.tomogram, kernel,mode='valid')
            score_mat += D[i] * jnp.abs(response)**2

        score_mat = np.array(score_mat - mu)
        num_particles, coords = self.picking_from_scoring_vol_3d(score_mat)
        return num_particles, coords
    
    def picking_from_scoring_vol_3d(self, score_vol):
        nx, ny, nz = score_vol.shape
        num_particles = self.num_particles
        max_iter = self.max_iter
        threshold = self.threshold
        offset = self.template_diameter // 2 

        r_del = self.particle_diameter // 2
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
            cx = ix + offset
            cy = iy + offset
            cz = iz + offset

            particle_list.append([cx, cy, cz, p_max / (log_max + eps)])

            # 3D spherical NMS
            mask = (gx - ix) ** 2 + (gy - iy) ** 2 + (gz - iz) ** 2 <= r2
            scoring[mask] = -np.inf

            num_picked += 1

        particle_coords = np.array(particle_list, dtype=np.float64) if particle_list else np.zeros((0, 4))
        num_picked_particles = particle_coords.shape[0]

        return num_picked_particles, particle_coords

# def old_prewhiten_tomogram(tomograms, factorization):
#     """ 
#         The function whitens each sub-tomogram using the extracted noise spectrum.

#         Args:
#             tomograms: a tensor containing K sub-tomograms of length N
#             factorization: A dataclass of type RPSDFactorization with ALS solution

#         Returns: 
#             whitened_tomograms: tensor containing K whitend sub-tomograms

#     """
#     _,N,_,_ = tomograms.shape
    
#     uniform_points, _ = generate_uniform_radial_sampling_points(N)
#     noise_rpsd = factorization.v

#     grid = jnp.arange(-(N-1), N) * jnp.pi / N
#     i,j,k = jnp.meshgrid(grid,grid,grid)
#     r_matrix = jnp.sqrt(i**2 + j**2 + k**2)
#     magnitudes, idx = jnp.unique(r_matrix, return_inverse=True)
#     nodes = magnitudes[magnitudes < uniform_points[-1]*jnp.pi]

#     interpolated_noise_rpsd = trigonometric_interpolation(uniform_points*np.pi, noise_rpsd, nodes)
#     noise_rpsd_mat = jnp.pad(interpolated_noise_rpsd, 
#                       (0,
#                        magnitudes.size - interpolated_noise_rpsd.size),
#                       'constant',
#                       constant_values=interpolated_noise_rpsd[-1])
#     noise_rpsd_mat = jnp.reshape(noise_rpsd_mat[idx], [grid.size, grid.size, grid.size])
    
#     whitened_tomograms = vect_prewhite_patch(tomograms,noise_rpsd_mat)
#     whitened_tomograms -= jnp.mean(whitened_tomograms)
#     whitened_tomograms /= jnp.linalg.norm(whitened_tomograms)
#     return whitened_tomograms







    