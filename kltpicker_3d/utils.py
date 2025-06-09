import jax 

import numpy as np
import jax.numpy as jnp

from functools import partial

@jax.jit
def jitted_binarysearch(X,query):
    idx = jnp.searchsorted(X,query,side='left')
    return idx
