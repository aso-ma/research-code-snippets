import networkx as nx
from math import log

def node_propagation_entropy(graph):
    clustering_dict = nx.clustering(graph)
    cn_dict = dict()
    for node in graph.nodes():
        n1 = len(list(graph.neighbors(node)))
        n2 = len({nj for ni in graph.neighbors(node) for nj in graph.neighbors(ni)})
        cn_dict[node] = (n1+n2)/(1+clustering_dict[node])
    cn_sum = sum(cn_dict.values())
    i_dict = dict()
    for node in graph.nodes():
        i_dict[node] = cn_dict[node] / cn_sum
    pe_dict = dict()
    for node in graph.nodes():
        temp = 0
        for neighbor in graph.neighbors(node):
            temp += i_dict[neighbor] * log(i_dict[neighbor])
        pe_dict[node] = -1 * temp
    return pe_dict