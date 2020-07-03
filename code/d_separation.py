import networkx as nx

def d_separates(g, x, y, z):
    assert nx.is_directed_acyclic_graph(g), "Graph {} must be a DAG!".format(g)
    for x_ in x:
        for y_ in y:
            paths = nx.all_simple_paths(nx.to_undirected(g), x_, y_)
            for path in paths:
                if path_d_separates(g, path, z):
                    continue
                else:
                    return False
    else:
        return True

def path_d_separates(g, path, z, return_node=False):

    for i, m in enumerate(path):
        if i==0 or i==len(path)-1: continue
        l, n = path[i-1], path[i+1]

        # l -> m -> n or l <- m <- n        
        chain = ((l, m) in g.edges and (m, n) in g.edges) or\
            ((n, m) in g.edges and (m, l) in g.edges)

        # l <- m -> n
        fork = (m, l) in g.edges and (m, n) in g.edges
        
        # l -> m <- n
        collider = (l, m) in g.edges and (n, m) in g.edges

        opt1 = (chain or fork) and m in z
        opt2 = collider and m not in z and all([d not in z for d in nx.descendants(g, m)])
        
        if opt1 or opt2:
            return (True if not return_node else (True, m))
    else:
        return (False if not return_node else (False, None))