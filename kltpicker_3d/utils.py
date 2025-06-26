import jax 

import numpy as np
import jax.numpy as jnp

from functools import partial

def trigonometric_interpolation(x,y,z):
    """ 
    Vectorized trigonometric interpolation for equidistant points. 

    args:
        - x: equidistant sample points 
        - y: function values at sampled points
        - z: evaluation points for interpolation
    """
    n = x.shape[0]
    
    scale = (x[1] - x[0]) * n / 2 
    x_scaled = (x / scale) * np.pi / 2 
    z_scaled = (z / scale) * np.pi / 2

    delta = z_scaled[:, None] - x_scaled[None, :]

    if n % 2 == 0:
        M = np.sin(n*delta) / (n *np.sin(delta))
    else:
        M = np.sin(n*delta)/ (n*np.tan(delta))
    M[np.isclose(delta,0)] = 1

    p = M @ y 
    return p


@jax.jit
def jitted_binarysearch(X,query):
    idx = jnp.searchsorted(X,query,side='left')
    return idx