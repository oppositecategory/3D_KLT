import jax 

import numpy as np
import jax.numpy as jnp

@jax.jit
def jitted_binarysearch(X,query):
    idx = jnp.searchsorted(X,query,side='left')
    return idx


def create_gaussian_window(k,max_d):
    l = 2*k-1 
    xx = np.array([[[i**2 + j ** 2 + k **2 for i in range(max_d+1)] for j in range(max_d+1)] for k in range(max_d+1)])
    xx = xx - k + 1
    alpha = 3.0
    window = np.exp(-alpha * (xx ** 2)/ ( 2 * max_d ** 2))
    return window 
    