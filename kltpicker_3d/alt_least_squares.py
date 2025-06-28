import jax
import jax.numpy as jnp 
from jax.numpy.linalg import norm
from jax.random import uniform


from dataclasses import dataclass 
from functools import partial 

@partial(jax.tree_util.register_dataclass,
         data_fields=['alpha_prev', 'gamma_prev','v_prev',
                      'alpha','gamma','v', 'iter_num'],
         meta_fields=[])
@dataclass
class RPSDFactorization:
    alpha_prev: jnp.ndarray 
    gamma_prev: jnp.ndarray 
    v_prev: jnp.ndarray 

    alpha: jnp.ndarray 
    gamma: jnp.ndarray 
    v: jnp.ndarray

    iter_num: int = 0 
   
key = jax.random.key(1701)

def alternating_least_squares_solver(samples, max_iter, eps):
    M,n = samples.shape
    S = samples.T

    norms_1 = jnp.sum(jnp.abs(S),axis=0)
    v = jnp.abs(S[:,jnp.argmin(norms_1)])
    
    # Faster approximation instead of computing the argmax of l2 error
    norms_infty = jnp.max(jnp.abs(S),axis=1)
    max_infty, min_infty = jnp.argmax(norms_infty), jnp.argmin(norms_infty)
    gamma  = jnp.abs(S[:,max_infty] - S[:,min_infty])

    alpha = jnp.dot(gamma, S - v[...,None]) / jnp.sum(gamma**2)
    alpha = jnp.clip(alpha, 0, 1)

    def alternating_least_squares_convergence(state): 
        v_error = norm(state.v - state.v_prev) / norm(state.v)
        gamma_error = norm(state.gamma - state.gamma_prev) / norm(state.gamma)
        alpha_error = norm(state.alpha - state.alpha_prev) / norm(state.alpha)
        return ((v_error >= eps) | (gamma_error >= eps) | (alpha_error >= eps)) & (state.iter_num < max_iter)

    def alternating_least_squares_iteration(state):
        alpha, gamma, v = state.alpha, state.gamma, state.v 

        alpha = jax.lax.cond(norm(alpha) == 0, 
                         lambda e: uniform(key, M,minval=0,maxval=1),
                         lambda e: e,
                         alpha)

        gamma_new = jnp.dot(alpha, S.T - v[None,...])/ jnp.sum(alpha ** 2)
        gamma_new = jnp.maximum(gamma_new, 0)

        v_new = S - jnp.outer(gamma_new, alpha)
        v_new = jnp.dot(v_new, jnp.ones(M)) / M
        v_new = jnp.maximum(v_new, 0)

        gamma_new = jax.lax.cond(norm(gamma_new) == 0,
                             lambda e: uniform(key, n, minval=0, maxval=1),
                             lambda e: e,
                             gamma_new)

        alpha_new = jnp.dot(gamma_new, S - v_new[...,None]) / jnp.sum(gamma_new ** 2 )
        alpha_new = jnp.clip(alpha_new, 0, 1)


        return RPSDFactorization(alpha,gamma, v,
                             alpha_new, gamma_new, v_new,
                             state.iter_num+1)
    
    init_state = RPSDFactorization(jnp.zeros_like(alpha),
                            jnp.zeros_like(gamma) , 
                            jnp.zeros_like(v),
                            alpha,
                            gamma,
                            v)
    factorization = jax.lax.while_loop(alternating_least_squares_convergence,
                           alternating_least_squares_iteration,
                           init_state)
    return factorization