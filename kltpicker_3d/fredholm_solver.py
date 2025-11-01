import numpy as np
import scipy
from scipy.special import spherical_jn, sph_harm


def solve_radial_fredholm_equation(Gx,
                           N: int,
                           a: float,
                           c: float,
                           K=150):
    """
        Solves radial KLT integral equation using Nystrom method.
        
        args:
            Gx: particle function's radial PSD
            a: particle diameter
            c: particle function's bandlimit 
            K: Legendre quadrature order (bounds the n indices in the solutions)

        returns:
            eigvals: eigenvalues lambda_{N,n} for n below K
            eigfuncs: eigenfunctions R_{N,n} for n below K
    """
    def Hn(x):
        return 2*np.pi * (2*(1j**N) * spherical_jn(N,x))

    X,w = scipy.special.roots_legendre(K)
    inner_rho = (c/2)*X + c/2
    outer_rho = (a/2)*X + a/2

    inner_grid = np.outer(inner_rho,inner_rho)
    vv = Hn(inner_grid)
    sgn = -1.0 if (N % 2) else 1.0
    D = (c/2) * w * Gx * (inner_rho**2)
    H = (vv * D[None,:]) @ (sgn *vv).T
    W = a/2 * np.diag(w*(outer_rho)**2)

    # Symmetrize the matrix
    A = np.sqrt(W) @ H @ np.sqrt(W)    
    eigvals, Y = np.linalg.eigh(A)
    R = np.linalg.solve(np.sqrt(W), Y) 

    eigvals = eigvals[::-1]
    eigfuncs = R[:,::-1]

    eigvals = np.where(np.abs(eigvals) < np.spacing(1),0, eigvals)
    eigfuncs[:,eigvals ==0] = 0
    return eigvals, eigfuncs, W


def create_GPSF_templates(eigfuncs, 
                          eigvals, 
                          orders,
                          G,
                          a: float,c: float, 
                          N: int, K=150):
    """
        Generates 3D templates in Generalized Prolate Spherodial Function basis
        using radial solutions of KLT equations and their spectrum.

        args:
            eigfuncs: radial solutions of KLT equations
            eigvals: correspoonding eigenvalues
            orders: the N orders of the corresponding solutions
            G: particle function's radial PSD 
            a: particle diameter
            c: particle's function bandlimit
            N: size of patch
            K: Legendre quadrature order
        
        returns:
            templates: 3D templates composed of radial and angular basis functions. 
            Shape of returned tensor: truncate_idx X max_N x (grid_shape)
            eigvals: spectrum of all eigenfunctions truncated at 99%
            orders: corresonding N order of each solution


        TODO: 
            - Clarify scale of a and how to determine patch_size. 
            - In original code patch_size_func/2 - 1 was used. Why?
    """
    grid = np.arange(-(N-1),N)
    X,Y,Z = np.meshgrid(grid,grid,grid)
    r_tensor = np.sqrt(X**2 + Y**2 + Z**2)
    rho_uniform, idx = np.unique(r_tensor.flatten(),return_inverse=True)

    # Legendre roots for both integrals
    rho_leg, w = scipy.special.roots_legendre(K)
    rho_leg_a =  (a * 0.5) * rho_leg + a / 2.0  
    rho_leg_c =  (c * 0.5)* rho_leg + c / 2
    
    # Truncates the spectra at 99%
    eigval_cumsum = np.cumsum(eigvals / np.sum(eigvals))
    truncate_idx = (eigval_cumsum < 0.99).argmax() + 1

    eigfuncs = eigfuncs[:truncate_idx,...]
    eigvals = eigvals[:truncate_idx,...]
    orders = orders[:truncate_idx,...]

    max_N = orders.max() + 1

    # We interpolate the radial solutions into uniform radial basis
    # using the Fredholm equation (re-expressing new values of R_{N,m} using  
    # values of it at Legendre roots.
    r_grid_uni = np.outer(rho_uniform, rho_leg_c)
    r_grid_leg = np.outer(rho_leg_a, rho_leg_c)

    def Hn(x,N):
        return 4*np.pi * ((1j**N) * spherical_jn(N,x))

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

    sph_harm = np.zeros((max_N,max_N) + theta.shape, dtype=np.complex64)
    for N in range(max_N):
        for m in range(N):
            sph_harm[N,m] = sph_harm(m,N,theta,phi)
    
    templates = sph_harm[orders]*radial_templates[:,None,...]
    return templates, eigvals, orders

    




