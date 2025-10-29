import numpy as np
import scipy
from scipy.special import spherical_jn, sph_harm

def solve_radial_fredholm_equation(Gx: np.ndarry,
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

def create_GPSF_templates(eigfuncs: np.ndarray, 
                          eigvals: np.ndarray, 
                          orders: np.ndarry,
                          G: np.ndaary,
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

    # We interpolate the radial solutions into uniform radial basis
    # using the Fredholm equation (re-expressing new values of R_{N,m} using  
    # values of it at Legendre roots.
    r_grid_uni = np.outer(rho_uniform, rho_leg_c)
    r_grid_leg = np.outer(rho_leg_a, rho_leg_c)

    def Hn_scipy(x,N):
        return 4*np.pi * ((1j**N) * spherical_jn(N,x))

    # Hn evaluated at multiples of uniform radial points in [0,a]
    Hn_uniform = np.array(
        [Hn_scipy(r_grid_uni,N) for N in range(orders.max()+1)]
    )

    # Hn evaluated at multiples Legendre roots in [0,a]
    Hn_leg = np.array(
        [Hn_scipy(r_grid_leg,N) for N in range(orders.max()+1)]
    )

    Hn_leg = Hn_leg[orders]
    Hn_uniform = Hn_uniform[orders]

    sgn = np.where(orders % 2 == 1, -1, 1)
    D = c * 0.5* w * G * (rho_leg_c**2)
    W = a * 0.5 * np.diag(w* rho_leg_a**2)

    H_right = np.swapaxes(sgn[:,None,None]* Hn_leg, -1,-2)
    psi = (Hn_uniform * D[None,None,:]) @ H_right

    # contains eigenfunctions evaluated at uniform radial sampling points
    eigfuncs_uniform = (psi @ W @ eigfuncs.T) / eigvals
    # TODO: Change to 3D using idx such that it contains corresponding
    # radii

    # creates Spherical Harmoonics templates
    theta = np.arctan2(Y, X)
    phi = np.arctan2(Z,np.sqrt(X**2 + Y**2))
    
    #TODO: Figure out how to vecotirze sph_harm. 
    # maybe Jax implementation can help.
    # you only need to calculate up to highest relevant order.
    # and for each such N calculate for |m| < N
    sph_harm = sph_harm(np.arange(orders.max()+1),
                        np.arange(20),
                        theta,phi)