import networkx as nx
from heuristic.ks import ks_if_influence, k_shell
from heuristic.pe import node_propagation_entropy

if __name__ == "__main__":
    g_karate = nx.karate_club_graph()
    
    k_shell_result = k_shell(g_karate)
    ks_if_result = ks_if_influence(g_karate)
    pe_result = node_propagation_entropy(g_karate)

    print("K-Shell decomposition of the Graph:", k_shell_result)