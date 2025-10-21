
import jax
import jax.numpy as jnp 
from jax import lax

from functools import partial

def _spherical_recurrence(x: jnp.ndarray, n_max: int):
    """
    Compute j_n(x) for n=0..n_max on a whole grid x (broadcasted),
    using the 3-term recurrence:
      j_{n+1}(x) = (2n+1)/x * j_n(x) - j_{n-1}(x)
    Returns array with shape (n_max+1, *x.shape).

    NOTE: This implementation is suitable for when x > n. 
    """
    x = x.astype(jnp.float64)
    # Safe divisions at x=0 using limits
    def safe_div(num, den):
        return jnp.where(den == 0.0, 0.0, num / den)

    j0 = jnp.where(x == 0.0, 1.0, jnp.sin(x) / x)
    j1 = jnp.where(x == 0.0, 0.0, jnp.sin(x) / (x * x) - jnp.cos(x) / x)

    if n_max == 0:
        return j0[None, ...]
    if n_max == 1:
        return jnp.stack([j0, j1], axis=0)

    def body(carry, k):
        jm1, jn = carry  # j_{k-1}, j_k
        coef = (2 * k + 1.0) * safe_div(1.0, x)
        jp1 = coef * jn - jm1
        return (jn, jp1), jp1

    (_, _,), ys = lax.scan(body, (j0, j1), jnp.arange(1, n_max))
    # ys holds j2..j_{n_max}
    return jnp.concatenate([j0[None, ...], j1[None, ...], ys], axis=0)


def _jv_power_series(nu : jnp.ndarry, x: jnp.ndarray, max_terms=100, tol=1e-15):
    """
    Cylindrical Bessel J_nu(x) for small |x| via power series.
    Works with scalar or array x, broadcasts over nu/x.

    args:
        nu: Cylindrical Bessel order 
        x: array input 
        max_terms: max numbers of terms to truncate the power series 
        tol: threshold for the magnitude of terms in power series
    """
    x = jnp.asarray(x)
    nu = jnp.asarray(nu)
    # a0 = (x/2)^nu / Gamma(nu+1); compute in log-space for stability
    log_a0 = nu * (jnp.log(jnp.abs(x)) - jnp.log(2.0)) - jax.lax.lgamma(nu + 1.0)
    a0 = jnp.exp(log_a0)
    # carry the sign of x^nu if x<0 and nu non-integer (we’ll assume real x, nu)
    # For real-valued principal branch, (x/2)^nu = exp(nu*log|x/2|) with no extra sign.

    z = (x * 0.5)**2

    def body(carry, k):
        """ Computes the Bessel power function efficiently by computing the ratio a_{k+1}/a_k.
        """
        term, s, done = carry
        # accumulate where not converged
        s = jnp.where(done, s, s + term)
        # next term via ratio
        denom = (k + 1.0) * (k + nu + 1.0)
        next_term = term * (-z) / denom
        # convergence check (relative to current sum)
        done_next = done | (jnp.abs(next_term) <= tol * (1.0 + jnp.abs(s)))
        return (next_term, s, done_next), None

    # initialize
    carry0 = (a0, jnp.zeros_like(a0), jnp.zeros_like(a0, dtype=bool))
    (termN, sumN, doneN), _ = jax.lax.scan(body, carry0, jnp.arange(max_terms))

    # Add the last term (scan adds term before updating; we added inside body already)
    out = sumN

    # Handle x=0 explicitly:
    # J_0(0)=1; J_nu(0)=0 for nu>0 (for Re(nu)>-1). We’ll set J_nu(0)=0 unless nu==0.
    out = jnp.where(x == 0,
                    jnp.where(nu == 0, jnp.ones_like(out), jnp.zeros_like(out)),
                    out)
    return out


@partial(jax.vmap, in_axes=(0,None))
def jn_spherical_small(nu, z, max_terms=100, tol=1e-15):
    return jnp.sqrt(jnp.pi/(2.0*z)) * _jv_power_series(nu+0.5, z, max_terms, tol)


def jn_spherical(z, n_max: int):
    """ Compute Spherical Bessel function up to order n_max on grid z. 

        For each order N the function partitions the grid z into z_1,..z_k,...z_M 
        where z_K is such that floor(z_k) = N. 
        Using this partition of the grid z we extract the stable approximation.


        TODO: Fix bug in mis-match orders. 
    """
    
    mask_ps = (z < jnp.arange(n_max)[...,None]).astype(int)
    mask_recurrence = jnp.abs(1 - mask_ps)
    
    # To reduce repetitive vmap over the corresponding batches 
    # it's faster to call both funcs and extract relevant solutions accoridng 
    # to partitions.
    jn_recurrence = _spherical_recurrence(z, n_max-1)
    jn_ps = jn_spherical_small(jnp.arange(1,n_max+1),z)
    # print(jn_recurrence.shape,jn_ps.shape)
    # print(mask_recurrence.shape,mask_ps.shape)

    output = jn_recurrence * mask_recurrence + jn_ps + mask_ps
    return output