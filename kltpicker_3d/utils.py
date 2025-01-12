import jax 

import numpy as np
import jax.numpy as jnp

from functools import partial

@jax.jit
def jitted_binarysearch(X,query):
    idx = jnp.searchsorted(X,query,side='left')
    return idx

@partial(jax.jit, static_argnames=['n'])
def gaussian_window(max_d,n):
    l = 2*n-1 
    grid = jnp.arange(l)
    i,j,k = jnp.meshgrid(grid,grid,grid)
    i,j,k = i - (n-1),j-(n-1),k-(n-1)
    xx = i**2 + j**2 + k**2
    alpha = 3.0
    window = jnp.exp(-alpha * (xx ** 2)/ ( 2 * max_d ** 2))
    return window 