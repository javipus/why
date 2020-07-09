import pytest
import networkx as nx
import numpy as np

from functools import reduce
from itertools import product

lproduct = lambda *args, **kwds: list(product(*args, **kwds))

## d-separation ##
from .d_separation import d_separates, path_d_separates

# Kite graph
# 0 = season
# 1 = rain
# 2 = sprinkler
# 3 = wet
# 4 = slippery
kite = nx.DiGraph()
kite.add_nodes_from(range(5))
kite.add_edges_from([
    (0,1), (0,2), (1,3), (2,3), (3,4)
])

# Basic motifs
chain = nx.DiGraph()
chain.add_nodes_from(range(3))
chain.add_edges_from([
    (0,1), (1,2)
])

fork = nx.DiGraph()
fork.add_nodes_from(range(3))
fork.add_edges_from([
    (1,0), (1,2)
])

coll = nx.DiGraph()
coll.add_nodes_from(range(3))
coll.add_edges_from([
    (0,1), (2,1)
])

@pytest.mark.parametrize(
    'g, path, z, blocking_node',
    [
        (kite, [0,1,3], [1,2], 1), # 1 is chain and in [1,2]
        (kite, [0,2,3], [1,2], 2), # 2 is chain and in [1,2]
        (kite, [1,0,2], [0], 0), # 0 is fork and in [0]
        (kite, [1,3,4], [3], 3), # 3 is chain and in [3]
        (kite, [2,3,4], [3], 3), # 3 is chain and in [3]
        (kite, [1,3,2], [0], 3), # 3 is collider and not in [0]
    ]
)
def test_path_blocks(g, path, z, blocking_node):
    blocks, blocking_node_ = path_d_separates(g, path, z, return_node=True)
    assert blocks
    assert blocking_node_ == blocking_node

@pytest.mark.parametrize(
    'g, path, z',
    [
        (kite, [0,1,3], []), # chain
        (kite, [1,3,2], [3]), # 3 is collider and in [3]
        (kite, [1,3,2], [4]), # 4 descends from collider 3 and is in [4]
        (kite, [1,3,2], [3,4]), # both conditions above
        (kite, [1,3,2], [0,3,4]), # adding a fork (0) doesn't fix it
    ]
)
def test_path_not_blocks(g, path, z):
    blocks, blocking_node_ = path_d_separates(g, path, z, return_node=True)
    assert not blocks
    assert blocking_node_ == None

@pytest.mark.parametrize(
    'g, x, y, z',
    [
        (chain, [0], [2], [1]),
        (fork, [0], [2], [1]),
        (coll, [0], [2], []),
        (kite, [0], [3], [1,2]), # 2 chains
        (kite, [0], [4], [1,2]), # 2 chains
        (kite, [1], [4], [3]), # 1 chain
        (kite, [2], [4], [3]), # 1 chain
        (kite, [1], [2], [0]), # 1 fork
    ]
)
def test_blocks(g, x, y, z):
    assert d_separates(g, x, y, z)

@pytest.mark.parametrize(
    'g, x, y, z',
    [
        (chain, [0], [2], []),
        (fork, [0], [2], []),
        (coll, [0], [2], [1]),
        (kite, [1], [2], [3]), # 1 collider
        (kite, [1], [2], [4]), # 1 desc of collider
        (kite, [1], [2], [0,3]), # 1 fork + 1 collider
        (kite, [1], [2], [0,3,4]), # 1 fork + 1 collider + 1 desc of collider
    ]
)
def test_not_blocks(g, x, y, z):
    assert not d_separates(g, x, y, z)


## random variables ##
from .causal_model import RandomVariable as RV
from .causal_model import RandomVector as RVV

domains = {
    '_bool' : (False, True),
    '_cat' : ('a', 'b', 'c'),
    '_set' : set(range(10)),
    '_list' : [x**2 for x in range(5)],
    '_interval' : np.linspace(0, 1, 100),
}

pmfs = {
    'unif_n': lambda n: lambda arg: 1/n,
    'delta': lambda x0: lambda arg: 1 if arg==x0 else 0,
}

eps = 1e-10

@pytest.mark.parametrize(
    'domain, name, pmf',
    [(dom, name, pmfs['unif_n']) for name, dom in domains.items()],
)
def test_unif_rv_normalized(domain, name, pmf, tol=eps):
    rv = RV(domain, pmf(len(domain)), name)
    assert abs(sum(rv.pmf(x) for x in rv.domain)-1) < tol

@pytest.mark.parametrize(
    'domain, name, pmf',
    [(dom, name, pmfs['delta']) for name, dom in domains.items()],
)
def test_delta_rv_normalized(domain, name, pmf, tol=eps):
    for x0 in domain:
        rv = RV(domain, pmf(x0), name)
        assert abs(sum(rv.pmf(x) for x in rv.domain)-1) < tol

@pytest.mark.parametrize(
    'n',
    range(1, 4)
)
def test_unif_vectors(n, tol=eps):
    doms = lproduct(domains.values(), repeat=n)

    for ds in doms:
        rvs = [RV(d, pmfs['unif_n'](len(d))) for d in ds]
        rvv = RVV.from_list(rvs)
        
        _len = lambda it: sum(1 for _ in it)
        _prod = lambda it: reduce(lambda x, y: x*y, it)
        
        assert _len(rvv.domain) == _prod(map(_len, [rv.domain for rv in rvs]))
        assert abs(sum(rvv.pmf(*x) for x in rvv.domain) - 1) < tol
        
        for x in rvv.domain:
            assert rvv.pmf(*x) == _prod(rv.pmf(xx) for xx, rv in zip(x, rvs))

# TODO test CausalModel and CausalMechanism