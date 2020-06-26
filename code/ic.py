from functools import reduce
from itertools import chain, combinations
import collections
import networkx as nx
from scipy.stats import chi2

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

    def _chi2(self, x, y, z):
        # TODO debug this
        # I think it's not doing what's supposed to
        # Maybe work out some examples yourself and write them as tests

        d = self._joint.copy()
        x, y, z = map(list, (x, y, z)) # wrap in list if they're not already
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

        for _, row in d.iterrows():
            x_, y_, z_ = row[x], row[y], row[z]
            
            data = {**dict((zip(x, x_))), **dict(zip(y, y_)), **dict(zip(z, z_))}

            pxyz_ = conditional(pxyz, {k: data[k] for k in w})['Pr'].values
            pyz_ = conditional(pyz, {k: data[k] for k in y+z})['Pr'].values
            pxz_ = conditional(pxz, {k: data[k] for k in x+z})['Pr'].values
            pz_ = conditional(pz, {k: data[k] for k in z})['Pr'].values
            p0 = pyz_*pxz_/pz_

            print(p0) #, pyz_, pxz_, pz_)

            chi2_ += (1 - pxyz_/p0)**2
        
        chi2_ *= self._N

        return 1-chi2(df=dof).cdf(chi2_)

    def update(self, *marginalize, **condition):
        return update(self._joint, *marginalize, **condition)
    
    def _marginal(self, nuisance):
        return marginal(self._joint, nuisance)

    def _conditional(self, conds):
        return conditional(self._joint, conds)

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

    if new.empty or den==0:
        raise ValueError('Can\'t condition on a probability 0 set!')

    new['Pr'] *= 1/den

    return new