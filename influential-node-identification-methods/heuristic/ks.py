def k_shell(graph):
  output = dict()
  graph_copy = graph.copy()
  degree_set = set(dict(graph_copy.degree()).values())
  for ks, k in enumerate(degree_set, 1):
    if graph_copy.number_of_nodes() == 0:
      break
    iteration = 1
    iteration_nodes = []
    while(len([d for _, d in graph_copy.degree() if d <= k]) > 0):   
      nodes_to_remove = [node for node, degree in graph_copy.degree() if degree <= k]
      graph_copy.remove_nodes_from(nodes_to_remove)
      for node in nodes_to_remove:
        output[node] = (ks, iteration)
      iteration_nodes.extend(nodes_to_remove)
      iteration+=1
    m = iteration - 1
    for node in iteration_nodes:
      ks, iteration = output[node] 
      output[node] = (ks, iteration, m)
  return output 

def k_shell_iteration_factor(graph, k_shell_dict):
  output = dict()
  for node in graph.nodes():
    ks, iter, max_iter = k_shell_dict[node]
    kif = ks * (1+(iter/max_iter))
    output[node] = kif
  return output

def ks_if_influence(graph):
  ks_dict = k_shell(graph)
  ksif_dict = k_shell_iteration_factor(graph, ks_dict)
  output = dict()
  for node in graph.nodes():
    s = 0
    for neighbor in graph.neighbors(node):
      s += ksif_dict[neighbor] * graph.degree(neighbor)
    output[node] = ksif_dict[node] * graph.degree(node) + s
  return output