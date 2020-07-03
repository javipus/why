import pytest
import networkx as nx

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


## causal models ##
from .causal_model import CausalModel, CausalMechanism

CausalModel(
    g = kite,
    fs = {},
    us = {},
)

CausalMechanism(
    x = 1,
    f = lambda x: x,
)