import numpy as np 
import scipy

import jax 
import jax.numpy as jnp
from jax.numpy import einsum

from scipy.special import legendre
import scipy.integrate as integrate

from functools import partial


def gpu_integral_equation_solver(G,a,c,N,K=150):
   """ Solves the Kosambi–Karhunen–Loève integral equation up to order N in parallel.

      Args:
        G: radial power spectrum
        a: particle's diameter 
        c: particle energy function's bandwidth 
        N: the maximum order of the equation we solve for 

      Returns:
        eigenvalues: eigenvalues of the discretized integral equation up to order N 
        eigenfunctions: the solutions of the integral equation
   """
   legendre_roots, w = scipy.special.roots_legendre(K)
   legendre_roots,w = jnp.array(legendre_roots), jnp.array(w)

   rho = c/2 * legendre_roots + c/2
   r = a/2 * legendre_roots + a/2

   # NxK matrix containing Legendre polynomial over [-1,1] 
   # for orders up to N
   p = jnp.array([legendre(k)(legendre_roots) for k in range(N)])
   g_tensor = G(rho)

   # Multiples of inner and outer integral varaibles
   legendre_multiples = r[...,None]*rho[None,...]
   
   even_orders = jnp.arange(2,N,2)
   odd_orders = jnp.arange(1,N,2)
   
   # Broadcasting the input of Hn (scaled Legendre roots multiplied) over the interval [-1,1]
   special_fn_input = einsum("ij,k->ijk",legendre_multiples,legendre_roots)

   p_even = p[even_orders]
   p_odd = p[odd_orders]

   # Tensors of size N//2 x K x K 
   hn_even = batched_Hn_even(special_fn_input,w,p_even) 
   hn_odd = batched_Hn_odd(special_fn_input,w,p_odd)

   hn = jnp.zeros((N,K,K))
   hn = hn.at[even_orders].set(hn_even)
   hn = hn.at[odd_orders].set(hn_odd)

   vv = einsum("bij,bjl->bil", hn, jnp.matrix_transpose(hn))
   jacobian = einsum("i,j->ij",r,rho) ** 2

   batched_psi_matrix = einsum("k,bij,k,ik->bij",w,vv,g_tensor,jacobian)
   
   # As of now there's no GPU-supported eigendecomposition
   # However as we are dealing with relatively small matrices the CPU
   # version is quite fast + the JIT help reduce time as we
   # are calling it for each patch with the same number of eigenvalues
   cpu_device = jax.devices('cpu')[0]
   cpu_psi = jax.device_put(batched_psi_matrix,device=cpu_device)
   eigenvalues, eigenfunctions = batched_eigdecomposition(cpu_psi,N)
   return eigenvalues, eigenfunctions

@jax.jit
def batched_Hn_even(x,w,p): 
  return 4 * jnp.pi * einsum("ijk,bk,k->bij", jnp.cos(x),p,w)
   
@jax.jit
def batched_Hn_odd(x,w,p):
  return 4* jnp.pi * einsum("ijk,bk,k->bij", jnp.sin(x),p,w)

@partial(jax.jit,static_argnames=['N'])
def batched_eigdecomposition(X,N):
    eigenvalues, eigenfunctions = [],[]
    for i in range(N):
        lambdas, eig_fns = jnp.linalg.eig(X[i,...])
        eigenvalues.append(lambdas)
        eigenfunctions.append(eig_fns)
    return eigenvalues,eigenfunctions

def cpu_integral_equation_solver(G,N,K=150):
  # TODO: 
  #   - Handle arbitrary diferent parameters a and c (particle diameter and bandwith)
  #   - Re-write in jax to utilize it's vmap utility; one can calculate the vv matrix for different values of N simulatenosly
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
    """ Function uses pre-computed evaluations of H_n at multiples of Legendre  roots
        and evaluate the integral in a vectorized way.
    """
    return 0.5 * np.sum(w * vv[j,:] * vv[i,:] * Gx * (X_scaled[j]*X_scaled)**2)

  U = np.array([[w[j]*psi(i,j) for j in range(K)] for i in range(K)])

  eigenvalues, eigenvectors = scipy.linalg.eig(U)
  return eigenvalues,eigenvectors

G = lambda x: np.exp(-x**2)
gpu_integral_equation_solver(G,1,1,100)

