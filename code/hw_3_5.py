from functools import reduce
from itertools import product

from sympy import symbols, simplify

from numpy import logical_xor as xor
import pandas as pd
from ic import update

def get_joint(prob, **vars):
    """
    Given
        prob :: vars -> joint_probability
        vars :: {var_name: var_codomain}
    
    Put everything into a dataframe with columns = vars.keys() + 'Pr'
    """

    df = pd.DataFrame(
        product(*vars.values()),
        columns = vars.keys()
    )

    df['Pr'] = df.apply(lambda args: prob(*args), axis=1)

    return df

if __name__=='__main__':
    # Model 1: p=1/2, q=1/2
    # U ~ Ber(p) - True if allergic
    # X ~ Ber(q) - treatment
    # Y = U xor X - recovers
    model1 = lambda u, x, y: [1-p, p][u] * [1-q, q][x] * int(y==xor(u,x))

    # Model 2: p=1/2, q=1/4, r=3/4
    # U ~ Ber(p) - True if allergic
    # X ~ Ber(q if U else r) - treatment
    # Y = U xor X - recovers
    model2 = lambda u, x, y: [1-p, p][u] * [[1-r, 1-q], [r, q]][x][u] * int(y==xor(u,x))
    
    # Model variables
    vars_ = dict(zip(['U', 'X', 'Y'], [[False, True], [False, True], [False, True]]))

    # Model parameters
    p, q, r, = symbols('p q r', real=True, nonnegative=True)

    # Compute
    df1 = get_joint(model1, **vars_)
    df2 = get_joint(model2, **vars_)

    # Update
    post1, post2 = map(lambda d: update(d, 'X', Y=0), [df1, df2])