import networkx as nx
from typing import Literal


def edge_removal_strategy(graph, threshold, edge_weight_approach: Literal["distance", "similarity"]):
  """
  Applies an edge removal strategy to the input graph based on a given threshold and whether the edge weights 
  represent 'distance' or 'similarity'. This strategy ensures the output graph remains connected if the input 
  graph is connected, by including the minimum or maximum spanning tree depending on the weight interpretation.
  
  Parameters:
  -----------
  graph : networkx.Graph
      A weighted NetworkX graph where edges have weights representing either distance or similarity.
      
  threshold : float
      A cutoff value for filtering edges. If the edge weights represent distances, only edges with weights 
      less than or equal to the threshold are retained (indicating closer nodes). If the weights represent 
      similarities, edges with weights greater than or equal to the threshold are retained (indicating stronger 
      connections).
      
  edge_weight_approach : Literal["distance", "similarity"]
      Specifies how to interpret the edge weights. If 'distance', the method retains edges with lower weights 
      (closer nodes). If 'similarity', it retains edges with higher weights (stronger correlations).

  Returns:
  --------
  final_graph : networkx.Graph
      A filtered version of the input graph that retains the minimum or maximum spanning tree to guarantee 
      connectivity, and includes additional edges based on the threshold and the weight interpretation.

  Raises:
  -------
  ValueError
      If the input graph is not weighted.

  Notes:
  ------
  This method ensures that the resulting graph is not disconnected if the input graph is connected, 
  by including either the minimum or maximum spanning tree. It was presented in the following paper:
  
  Mafakheri, A., & Sulaimany, S. (2024). Android malware detection through centrality analysis of applications 
  network. Applied Soft Computing, 165, 112058.
  
  When edge weights represent 'distance', the strategy retains edges with lower weights (indicating proximity).
  When edge weights represent 'similarity', the strategy retains edges with higher weights (indicating stronger
  relationships).

  Example:
  >>> G = nx.karate_club_graph()
  >>> G_new = edge_removal_strategy(G, 5, 'similarity')
  >>> print(G_new)
  """
  if not nx.is_weighted(graph):
    raise ValueError("The input graph must be weighted!")
  final_graph = nx.Graph()
  final_graph.name = graph.name
  final_graph.add_nodes_from(graph.nodes())
  # If edge weights represent distance, retain edges with lower weights (closer nodes)
  if(edge_weight_approach == 'distance'):
    mst = nx.minimum_spanning_tree(graph, weight='weight', algorithm='prim')
    final_graph.add_edges_from(mst.edges(data=True))
    final_graph.add_weighted_edges_from([(v, u, data['weight']) 
                                         for v, u, data in graph.edges(data=True) 
                                         if not mst.has_edge(v, u) and data['weight'] <= threshold])
  # If edge weights represent similarity, retain edges with higher weights (stronger connections)
  else:
    mst = nx.maximum_spanning_tree(graph, weight='weight', algorithm='prim')
    final_graph.add_edges_from(mst.edges(data=True))
    final_graph.add_weighted_edges_from([(v, u, data['weight']) 
                                         for v, u, data in graph.edges(data=True) 
                                         if not mst.has_edge(v, u) and data['weight'] >= threshold])
  return final_graph



if __name__ == "__main__":
  g = nx.karate_club_graph()
  g_new = edge_removal_strategy(g, 5, 'similarity')
  g_new = edge_removal_strategy()
  print(g_new)