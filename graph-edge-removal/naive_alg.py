from tqdm import tqdm
import networkx as nx
import random

def naive_approach(graph, gamma):
    """
    Simplifies a weighted graph by pruning edges based on their weights while maintaining connectivity.

    Parameters:
    - graph (networkx.Graph): A weighted undirected graph represented using NetworkX. 
    - gamma (float): A parameter between 0 and 1 that determines the extent of pruning. 
        A value of 0 means no edges are removed, 
        while a value of 1 results in a spanning tree.

    Returns:
    - networkx.Graph: A new graph object that is a simplified version of the input graph 

    Raises:
    - ValueError: If `gamma` is not in the range [0, 1].

    Example:
    >>> import networkx as nx
    >>> G = nx.Graph()
    >>> G.add_weighted_edges_from([(1, 2, 0.5), (2, 3, 0.2), (1, 3, 0.8)])
    >>> simplified_graph = naive_approach(G, gamma=0.5)

    Reference:
    Zhou, F., Mahler, S., & Toivonen, H. (2012). Simplification of Networks by Edge Pruning. In *Network Simplification with Minimal Loss of Connectivity* (pp. 179–198). SpringerLink.
    """

    # Validate gamma
    if not (0 <= gamma <= 1):
        raise ValueError("Parameter 'gamma' must be in the range [0, 1].")

    edges = [(e[0], e[1], e[2]['weight']) for e in graph.edges(data=True)]
    edges.sort(key=lambda x:x[2])
    
    g_temp = graph.copy()
    v_count = graph.number_of_nodes()
    e_count = graph.number_of_edges()
    
    n = int(gamma * (e_count - (v_count-1)))
    
    for i in tqdm(range(n)):
        v, u, w = edges[i]
        g_temp.remove_edge(v, u)
        if nx.has_path(g_temp, v, u) == False:
            g_temp.add_edge(v, u, weight=w)
    
    return g_temp

if __name__ == "__main__":

    random_graph = nx.gnp_random_graph(250, 0.5)
    for u, v in random_graph.edges():
        weight = random.random()
        random_graph[u][v]['weight'] = weight
    
    simplified_graph = naive_approach(random_graph, 0.5)
    print(random_graph)
    print(simplified_graph)