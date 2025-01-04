import numpy as np

#@jit(nopython=True)
def bsearch(x, lower_bound, upper_bound):
    """
    Binary search in a sorted vector.
    
    Binary O(log2(N)) search of the range of indices of all elements of x 
    between LowerBound and UpperBound. If no elements between LowerBound and
    Upperbound are found, the returned lower_index and upper_index are empty.
    The array x is assumed to be sorted from low to high, and is NOT verified
    for such sorting. 
    Based on code from 
    http://stackoverflow.com/questions/20166847/faster-version-of-find-for-sorted-vectors-matlab

    Parameters
    ----------
    x : numpy.ndarray
        A vector of sorted values from low to high.
    lower_bound : float
        Lower boundary on the values of x in the search.
    upper_bound : flo
        Upper boundary on the values of x in the search.

    Returns
    -------
    lower_idx: int
        The smallest index such that LowerBound<=x(index)<=UpperBound.
    upper_idx: int
        The largest index such that LowerBound<=x(index)<=UpperBound.
    """
    if lower_bound > x[-1] or upper_bound < x[0] or upper_bound < lower_bound:
        return None, None
    lower_idx_a = 1
    lower_idx_b = len(x)
    upper_idx_a = 1
    upper_idx_b = len(x)

    while lower_idx_a + 1 < lower_idx_b or upper_idx_a + 1 < upper_idx_b:
        lw = int(np.floor((lower_idx_a + lower_idx_b) / 2))
        if x[lw - 1] >= lower_bound:
            lower_idx_b = lw
        else:
            lower_idx_a = lw
            if upper_idx_a < lw < upper_idx_b:
                upper_idx_a = lw

        up = int(np.ceil((upper_idx_a + upper_idx_b) / 2))
        if x[up - 1] <= upper_bound:
            upper_idx_a = up
        else:
            upper_idx_b = up
            if lower_idx_a < up < lower_idx_b:
                lower_idx_b = up

    if x[lower_idx_a - 1] >= lower_bound:
        lower_idx = lower_idx_a
    else:
        lower_idx = lower_idx_b
    if x[upper_idx_b - 1] <= upper_bound:
        upper_idx = upper_idx_b
    else:
        upper_idx = upper_idx_a

    if upper_idx < lower_idx:
        return None, None

    return lower_idx, upper_idx

def create_gaussian_window(k,max_d):
    l = 2*k-1 
    xx = np.array([[[i**2 + j ** 2 + k **2 for i in range(max_d+1)] for j in range(max_d+1)] for k in range(max_d+1)])
    xx = xx - k + 1
    alpha = 3.0
    window = np.exp(-alpha * (xx ** 2)/ ( 2 * max_d ** 2))
    return window 
    