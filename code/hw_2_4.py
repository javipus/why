import numpy as np
from numpy import logical_xor as _xor
from numpy import logical_or as _or
from numpy import logical_and as _and
from numpy import logical_not as _not

from scipy.stats import binom

import pandas as pd

N = int(1e6)
ber = binom(p=1/2, n=1).rvs
u1 = ber
u2 = ber
fs = [
    lambda x, ux, uy: _or(_xor(x,ux),uy),
    lambda x, ux, uy: _xor(_xor(x,ux),uy),
]

def model(f, do=False, N=N):
    ux = u1(size=N)
    uy = u2(size=N)
    x = ber(size=N) if do else ux
    y = f(x, ux, uy)

    return (x,y)

def table(x, y):
    b = False, True
    return np.array([[np.average((x==b1)&(y==b2)) for b1 in b] for b2 in b])

def main():
    for do in (False, True):
        print('\nDO={}:\n'.format(do))
        for i,f in enumerate(fs):
            x, y = model(f, do)
            r = np.corrcoef(x,y)[0,1]
            t = table(x,y)
            print('Model {}: r = {:.3f}'.format(i+1, r))
            with np.printoptions(precision=2):
                print(t)
            print('\n')

if __name__=='__main__':
    main()