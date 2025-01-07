import numpy as np 
import scipy

from scipy.special import legendre
import scipy.integrate as integrate

#from tqdm import tqdm

def solve_eigenfunction_equation(G,N,K=150):
  # TODO: The functio right now is adapted to integrating over [0,1] both integrals.
  #       Needs to be updated to handle both [0,a] and [0,c] and scale the Legendre roots appropriately.
  def Hn(x):
    p = legendre(N)
    if N % 2 == 0:
        return 4 * np.pi * integrate.quad(lambda u: np.cos(x*u) * p(u), 0,1)[0]
    return 4 * np.pi * integrate.quad(lambda u: np.sin(x*u),0,1)[0]
  
  X,w = scipy.special.roots_legendre(K)
  X_scaled = 0.5*X + 0.5

  vv = np.array([[Hn(X_scaled[i]*X_scaled[j]) for j in range(K)] for i in range(K)])
  Gx = G(X_scaled)

  def psi(i,j):
    """ Function uses pre-computed evaluations of H_n at multiples of Legendre roots
        and evaluate the integral in a vectorized way.
    """
    return 0.5 * np.sum(w * vv[j,:] * vv[i,:] * Gx * (X_scaled[j]*X_scaled)**2)

  U = np.array([[w[j]*psi(i,j) for j in range(K)] for i in range(K)])

  eigenvalues, eigenvectors = scipy.linalg.eig(U)
  return eigenvalues,eigenvectors

# G = lambda x: np.exp(-x**2)

# for N in tqdm(range(1,100)):
#     x = solve_eigenfunction_equation(G,N)
