from math import log

def _get_neighbors_of_neighbors(graph, v):
    """Get the neighbors of the neighbors of node v in a bipartite graph."""
    neighbors_of_neighbors = set()
    first_neighbors = graph.neighbors(v)
    for neighbour in first_neighbors:
        second_neighbors = graph.neighbors(neighbour)
        neighbors_of_neighbors.update(second_neighbors)
    return neighbors_of_neighbors

def jaccard(graph, v, u):
    neighbors_v = set(graph.neighbors(v))
    neighbors_u = _get_neighbors_of_neighbors(graph, u)

    intersection = len(neighbors_u.intersection(neighbors_v))
    union = len(neighbors_u.union(neighbors_v))
    
    if union == 0:
        return 0.0  
    return intersection / union

def adamic_adar(graph, v, u):
    neighbors_v = set(graph.neighbors(v))
    neighbors_u = _get_neighbors_of_neighbors(graph, u)
    common_neighbors = neighbors_u.intersection(neighbors_v)

    score = 0.0
    for z in common_neighbors:
        degree_z = graph.degree(z)
        if degree_z > 0:
            score += 1 / log(degree_z)
    
    return score

def common_neighbor(graph, v, u):
    neighbors_v = set(graph.neighbors(u))
    neighbors_u = _get_neighbors_of_neighbors(graph, u)
    return len(neighbors_u.intersection(neighbors_v))


def preferential_attachment(graph, v, u):
    degree_v = graph.degree(v)
    degree_u = graph.degree(u)

    return degree_v * degree_u
