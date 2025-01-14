from tqdm import tqdm
from torch_geometric.utils import from_networkx
import torch
import pickle
import pandas as pd 
import numpy as np
import networkx as nx

def generate_subgraph_for(graph, edge):
    v, u = edge
    sub_graph = nx.Graph()

    sub_graph.add_nodes_from([(v, {'bipartite': 0}), (u, {'bipartite': 1})])
    sub_graph.add_edge(v, u)

    # Add neighbors for v
    neighbors_v = graph.neighbors(v)
    sub_graph.add_nodes_from((n, {'bipartite': 1}) for n in neighbors_v)
    sub_graph.add_edges_from((v, n) for n in neighbors_v)

    # Add neighbors for u
    neighbors_u = graph.neighbors(u)
    sub_graph.add_nodes_from((n, {'bipartite': 0}) for n in neighbors_u)
    sub_graph.add_edges_from((u, n) for n in neighbors_u)

    return sub_graph

if __name__ == '__main__':
    path_to_write = './pyg_subgraphs/'

    with open('std_crs_graph_with_features.pkl', 'rb') as f:
        graph = pickle.load(f)

    df_edge_data = pd.read_excel('enrollment_data.xlsx')
    df_node_data = pd.read_pickle('data_node_att.pkl')
    
    edge_data = dict()
    for row in df_edge_data.itertuples():
        edge_att_array = np.fromstring(row.edge_att[1:-1], dtype=float, sep=', ')
        edge_att_tensor = torch.tensor(edge_att_array, dtype=torch.float32).view(1, -1)
        edge_data[(row.username, row.course_id)] = edge_att_tensor
        edge_data[(row.course_id, row.username)] = edge_att_tensor

    node_data = dict()
    for row in df_node_data.itertuples():
        node_data[row.nodes] = torch.tensor(row.node_att, dtype=torch.float32).view(1, -1)

    label_data = dict()
    for row in df_edge_data.itertuples():
        label_data[(row.username, row.course_id)] = row.Labels
        label_data[(row.course_id, row.username)] = row.Labels

    e_count = graph.number_of_edges()


    for edge in tqdm(graph.edges(), total=e_count):
        subgraph = generate_subgraph_for(graph, edge)
        edge_att_data_list = [] 
        node_att_data_list = []

        edge_att_data_list.extend(edge_data[(v, u)] for v, u in subgraph.edges())
        node_att_data_list.extend(node_data[v] for v in subgraph.nodes())

        edge_att_data = torch.vstack(edge_att_data_list)
        node_att_data = torch.vstack(node_att_data_list) 

        pyg_graph = from_networkx(subgraph)
        pyg_graph.edge_attr = edge_att_data
        pyg_graph.x = node_att_data
        pyg_graph.y = label_data[(edge[0], edge[1])]

        with open(f'{path_to_write}/{edge[0]}_{edge[1]}', 'wb') as f:
            pickle.dump(pyg_graph, f)