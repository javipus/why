import numpy as np

from numpy import logical_xor as _xor
from numpy import logical_or as _or

from scipy.stats import binom

import pandas as pd

ber = lambda p, size: binom(p=p, n=1).rvs(size=size)

def problem_4_1(n, t, p, q):
    """
    Take n samples the following model, for k=1,...,t:
        E_k, E'_k ~ Ber(p) iid
        X_0, Y_0 ~ Ber(q) iid
        x_k+1 = (x_k or y_k) xor e_k
        y_k+1 = (x_k or y_k) xor e'_k
    """

    # e.shape = (2, n_samples, n_iters+1)
    e = ber(p=p, size=2*n*(t+1)).reshape([2,n,t+1])

    # Initial conditions
    x0, y0 = ber(p=q, size=n), ber(p=q, size=n)

    # x.shape = y.shape = (n_samples, n_iters+1)
    x, y = x0.reshape([n,1]), y0.reshape([n,1])

    for k in range(1,t+1):
        x = np.hstack([
            x,
            _xor(_or(x[:,-1], y[:,-1]), e[0,:,k]).reshape([n,1])
        ])

        y = np.hstack([
            y,
            _xor(_or(x[:,-1], y[:,-1]), e[1,:,k]).reshape([n,1])
        ])

    return e, x, y