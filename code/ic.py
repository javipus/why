from functools import reduce, partial
from itertools import chain, combinations, product
import collections
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import chi2
import scipy

def IC(P):
    ## STEP 1: test independences
    # - Generate complete graph
    v = P._vars
    assert all([isinstance(vv, collections.Hashable) for vv in v]), "Variable names must be hashable"
    n = len(v)
    g = nx.complete_graph(v)
    s = {} # dict of cond independences

    # - Test X \indep Y | Z
    for i, a in enumerate(g.nodes):
        for j, b in enumerate(g.nodes):
            if j<=i: continue
            # All nodes != a and b
            z = filter(lambda v: v not in (a, b), g.nodes)
            # Powerset of z
            pz = chain.from_iterable(combinations(z, r) for r in range(n-1))
            # Iterate over pz
            for sab in pz:
                if P.check_independence(a, b, cond=sab):
                    s[(a,b)] = sab # save cond set for next step
                    g.remove_edge(a,b)
                    break

    ## STEP 2: add colliders
    for i, a in enumerate(g.nodes):
        for j, b in enumerate(g.nodes):
            if j<=i: continue
            if g.has_edge(a,b): continue
            z = filter(lambda v: v not in (a, b), g.nodes)
            for c in z:
                if g.has_edge(a,c) and g.has_edge(c,b):
                    if c not in s[(a,b)]:
                        nx.set_edge_attributes(g,
                            {
                                (a,c): {'dir': (a,c)},
                                (b,c): {'dir': (b,c)}
                            })
    
    ## STEP 3: orient the remaining edges
    # TODO debug this step
    # I'm using Dor and Tarsi https://ftp.cs.ucla.edu/pub/stat_ser/r185-dor-tarsi.pdf
    visited = []
    for i, a in enumerate(g.nodes):
        n = [v for v in g.neighbors(a) if v not in visited]
        # check it's sink
        if any([g.get_edge_data(a,b).get('dir', False)==(a,b)]): continue

        for b in n:
            if (dir_:=g.get_edge_data(a,b).get('dir'))==(a,b): # not sink
                break
            elif dir_==None: # undirected
                if not set(g.neighbors(b)).issubset(set(n)): # b not adjacent to all of a's neighbors
                    break

            # if a is a (partial) sink and not a member of a v-structure, direct edge b->a
            nx.set_edge_attributes(g,
            {
                (a,b): {'dir': (b,a)}
            }
            )

            visited += [a]

    return g

class Prob:

    def __init__(self, dom, p):
        self.dom = dom
        self._vars = set(dom.keys())
        self.p = p
        # TODO assert argspec p = dom

    def marginal(self, *args):
        return Prob(
            {v: self.dom[v] for v in self._vars-set(args)},
            lambda **kwds: sum([self.p(**{**kwds, **{args: item}}) for item in product([self.dom[m] for m in args])])
        )

    def conditional(self, **kwds):
        return self[kwds] / self.marginal(self._vars-set(kwds.keys()))[kwds]

    @property
    def joint(self):
        pass

    def _restrict(self, **kwds):
        pass

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, new):
        assert hasattr(new, '__call__'), "Probability must be callable"
        self._p = new

    def __div__(self, other):
        return Prob(self.dom, lambda *args, **kwds: self.p(*args, **kwds)/other)
class Probability:
    """
    Joint empirical probability distribution.
    """

    def __init__(self, df, N):
        """
        Create Probability instance from long-form dataframe with format:
            - Column names are variables X_1,...,X_n together with a column named 'Pr'.
            - Each row contains one value (X_i=x_i)_i=1,...,n and its joint probability Pr[X_1=x_1,...,X_n=x_n].
        
        @param df: DataFrame with format indicated above.
        @param N: Integer - sample size.
        """

        assert 'Pr' in df.columns, "DataFrame must contain probability column"
        self._vars = list([v for v in df.columns if v!='Pr'])
        self._joint = df
        self._N = N

    def check_independence(self, x, y, z, method='chi2'):
        """
        @param X, Y: Variables in the model.
        @param Z: Iterable containing variables in the model.
        @param method: Callable implementing the check.
        @returns True iff X is independent of Y given Z. 
        """

        # TODO I'd like this to be bayesian; ideally:
        # - put a prior on the 2**(n*(n-1)/2) graph topologies
        # - update based on "strength" of independence implied by data
        assert x in self._vars, "{} is not a model variable!".format(x)
        assert y in self._vars, "{} is not a model variable!".format(y)
        assert set(z).issubset(set(self._vars)), "{} are not model variables!".format(z)

        if method=='chi2':
            return self._chi2(x, y, z)
        
        if hasattr(method, '__call__'):
            return method(x, y, z)

        raise NotImplementedError

    def _chi2(self): #, x, y, z):
        # TODO figure out how to use scipy's chi2_contingency

        return scipy.stats.chi2_contingency(self._contingency_table)

    @property
    def _contingency_table(self):
        table = np.zeros(shape=[self._joint[v].nunique() for v in self._vars])
        levels = {v: {val: k for k, val in zip(range(d[v].nunique()), d[v].unique())} for v in d.columns if v!='Pr'}

        for _, row in self._joint.iterrows():
            table[tuple([levels[v][row[v]] for v in self._vars])] = int(self._N*row['Pr'])

        return table

    def _deprecated_chi2(self, x, y, z):
        # NB: see _chi2

        d = self._joint.copy()
        # Wrap vars into lists
        x, y = [x], [y]
        z = list(z) if not isinstance(z, str) else [z]
        w = x + y + z # relevant variables
        v = [v for v in self._vars if v not in w] # marginalize over these

        # Marginal over remaining variables
        pxyz = update(d, *v)

        # Marginals over x, y and z
        pxz = update(pxyz, *y)
        pyz = update(pxyz, *x)
        pz = update(pxyz, *(x+y))

        # degrees of freedom
        rz = pz.shape[0]
        rx, ry = pxz.shape[0]/rz, pyz.shape[0]/rz
        dof = (rx-1)*(ry-1)*rz

        # Compute chi2 
        chi2_ = 0
        print(pxz)

        for _, row in d.iterrows():
            x_, y_, z_ = row[x], row[y], row[z]
            
            data = {**dict((zip(x, x_))), **dict(zip(y, y_)), **dict(zip(z, z_))}

            pxyz_ = conditional(pxyz, {k: data[k] for k in w})['Pr'].values
            pyz_ = conditional(pyz, {k: data[k] for k in y+z})['Pr'].values
            pxz_ = conditional(pxz, {k: data[k] for k in x+z})['Pr'].values
            pz_ = conditional(pz, {k: data[k] for k in z})['Pr'].values
            p0 = safe_div(pyz_*pxz_, pz_)

            #print(row)
            print(x_.values, y_.values, z_.values, pxz_) #, pxz_, pz_)

            chi2_ += (1 - safe_div(pxyz_, p0))**2
        
        chi2_ *= self._N

        return 1-chi2(df=dof).cdf(chi2_)

    def update(self, *marginalize, **condition):
        return update(self._joint, *marginalize, **condition)
    
    def _marginal(self, nuisance):
        return marginal(self._joint, nuisance)

    def _conditional(self, conds):
        return conditional(self._joint, conds)

class Sprinkler(Probability):

    def __init__(self, N=1000, **params):

        self._vars = {
            'se': [0,1],
            'sp': [0,1],
            'r': [0,1],
            'w': [0,1],
        }

        super(Sprinkler, self).__init__(
            get_joint(sprinkler(**params), **self._vars),
            N = N
        )

    def check_independence(self, x, y, z):
        """Manually override independence check to test IC algorithm while I implement the real thing."""

        g = nx.DiGraph()
        g.add_nodes_from(self._vars)
        g.add_edges_from(
            [('se', 'r')],
            [('se', 'sp')],
            [('sp', 'w')],
            [('r', 'w')],
        )

        return d_separates(g, x, y, z)

def d_separates(g, x, y, z):
    for x_ in x:
        for y_ in y:
            paths = nx.all_simple_paths(g, x_, y_)
            for path in paths:
                if path_d_separates(g, path, z):
                    continue
                else:
                    return False
    else:
        return True

def path_d_separates(g, path, z):
    # TODO
    pass

def safe_div(x, y):
    try:
        return x/y
    except ZeroDivisionError as e:
        if x==0:
            return 0
        else:
            raise e

def update(joint, *marginalize, **condition):
    """
    Update P(X,Y,Z) to P(X|Y), where

    @param joint: DataFrame containing P(X,Y,Z) in long format. For N variables (X_i)_i in {1:N}, there are N+1 columns (N for the variables, one named 'Pr' for the probability) and prod_i |cod X_i| rows.

    @param marginalize: Names of variables to integrate out.
    
    @param condition: Dict of variables to condition on, with variable names as keys and iterables of their values as values.
    """

    assert 'Pr' in joint.columns, "No proability column found in joint!"    
    xs = [x for x in joint.columns if x!='Pr']
    assert all([n in xs for n in condition.keys()]), "Conditional variables not in joint!"
    assert all([n in xs for n in marginalize]), "Nuisance variables not in joint!"

    new = joint.copy()

    if marginalize:
        new = marginal(new, marginalize)

    if condition:
        new = conditional(new, condition)

    return new

def marginal(joint, nuisance):
    """
    Integrate out nuisance variables in joint probability to get the marginal on the rest.
    """
    return joint.groupby([x for x in joint.columns if x not in set(nuisance) and x!='Pr']).agg({'Pr': 'sum'}).reset_index()

def conditional(joint, conds):
    """
    Condition a joint probability distribution.
    """

    conds = {k: [v] if not hasattr(v, '__len__') else v for k, v in conds.items()}
    obs_cond = reduce(lambda x, y: x & y, [joint[k].isin(conds[k]) for k in conds])
    new = joint[obs_cond].copy()
    den = new['Pr'].sum()

    if new.empty:# or den==0:
        #print(new)
        #print(den)
        raise ValueError('Can\'t condition on a probability 0 set!')

    new['Pr'] = safe_div(new['Pr'],den)

    return new

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

def sprinkler(p=.5, q0=.25, q1=.75, r0=.75, r1=.25):
    """
    Probability distribution for the sprinkler graph:

        Season -> Rain -> Wet
        Season -> Sprinkler -> Wet

    with all variables in {0,1} and

        P[Season=0] = P[Season=1] = p
        P[Sprinkler=1|Season] ~ Ber(q1*Season + q0*(1-Season))
        P[Rain=1|Season] ~ Ber(r1*Season + r0*(1-Season))
        P[Wet|Sprinkler, Rain] = Sprinkler OR Rain
    """

    pse = lambda se: [1-p, p][se]
    psp = lambda sp, se: [[1-q0, 1-q1], [q0, q1]][sp][se]
    pr = lambda r, se: [[1-r0, 1-r1], [r0, r1]][r][se]
    pw = lambda w, sp, r: 1 if (bool(w) == bool(sp or r)) else 0

    return lambda se, sp, r, w: pse(se)*psp(sp,se)*pr(r,se)*pw(w,sp,r)

if __name__=='__main__':
    
    from sympy import symbols, simplify

    vars_ = {
        'se': [0,1],
        'sp': [0,1],
        'r': [0,1],
        'w': [0,1],
    }

    #params = symbols('p q0 q1 r0 r1', real=True, nonnegative=True)

    params = {
        'p': 1/2,
        'q0': 1/4,
        'q1': 3/4,
        'r0': 3/4,
        'r1': 1/4,
    }

    N = 1000

    d = get_joint(sprinkler(**params), **vars_)

    p = Probability(d, N)

    s = Sprinkler(**params)