from functools import reduce
from itertools import product
import networkx as nx
from .d_separation import d_separates

class RandomVariable:
    # TODO test

    def __init__(self, domain, pmf, name=None):
        self.domain = domain
        self.pmf = pmf
        if name is not None: self.name = name

class RandomVector:
    # TODO test

    def __init__(self, domains, pmfs, names=None):
        self.domain = product(domains)
        self.pmf = lambda *args: reduce(lambda x, y: x*y, [pmf(arg) for arg, pmf in zip(args, pmfs)])
        if names is not None: self.names = names

    @classmethod
    def from_list(cls, rvs):
        return cls.__init__(
            domains = [rv.domain for rv in rvs],
            pmfs = [rv.domain for rv in rvs],
            names = map(lambda x: None if all([xx is None for xx in x]) else x, [rv.name if hasattr(rv, 'name') else None for rv in rvs]),
        )

class CausalMechanism:

    def __init__(self, x, f):
        self.x = x
        self.f = f

class CausalModel(nx.DiGraph):

    def __init__(self, g, fs, us):
        """
        Create causal model from graph.

        @param g: networkx.DiGraph with RandomVariable instances as nodes.
        @param fs: Dictionary with model variable names as keys and functions as values.
        @param us: Dictionary with model variable names as keys and noise variables as values.
        """

        # Enforce nodes are named RVs
        if not all([isinstance(x, RandomVariable) and\
            hasattr(x, 'name') for x in g.nodes]):
            raise TypeError("Nodes must be named random variables!")

        super(CausalModel, self).__init__(g)

        # Observable variables
        self._model_vars = set(self.nodes)
        self._model_var_names = set([x.name for x in self._model_vars])

        # Add noise variables to model
        for x, pu in us.items():
            if x not in self._model_var_names:
                raise ValueError("Variable {} in noise term but not in model!".format(x))
            
            # Enforce name format
            u = RandomVariable(domain=u.domain, pmf=u.pmf, name='U_{}'.format(x))
            self.add_node(u)
            self.add_edge(u, x)

        # Noise variables
        self._noise_vars = set(self.nodes).difference(self._vars)
        self._noise_var_names = set([u.name for u in self._noise_vars])

        # Markovian condition: U_i \perp U_j if 1!=j
        self._prior = RandomVector(
            domains=[u.domain for u in self._noise_vars],
            pmfs = [u.pmf]
        )

        # Add causal mechanisms
        for x, f in fs.items():
            assert x in g.nodes, "Variable {} in mechanisms but not in model!".format(x)
            nx.set_node_attributes(self, {x: CausalMechanism(x, f)}, 'causal_mechanism')
            # TODO enforce correct type signature
            # `assert parents(x) == fs[x].kwds` or something
            # otoh forcing the fs to have their parameter names tied to a specific graph seems suboptimal - can't reuse functions

    def d_separates(self, x, y, z):
        # TODO wrap args in list
        return d_separates(self, x, y, z)

    @property
    def prob(self):
        """Joint probability distribution."""
        def prob_(xs):
            for x in self.nodes:
                if not (pa:=self.parents(x)):
                    return nx.get_node_attributes(self, 'prob')[x]
                else:
                    f = nx.get_node_attributes(self, 'causal_mechanism')[x]
                    u = [p for p in pa if p not in self._vars]
                    pu = nx.get_node_attributes(self, 'prob')[u]

                    return sum([(xs[x]==f(pai,u=ui))*pu(ui) for pai in pa.domain for ui in u.domain])

    def parents(self, x):
        return set([p for p in nx.all_neighbors(self, x) if (p, x) in self.edges])