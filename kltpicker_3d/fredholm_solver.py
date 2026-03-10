import numpy as np
import scipy
from scipy.special import spherical_jn


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

    



