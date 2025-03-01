from typing import Callable, Optional
import networkx as nx
import numpy as np
from math import atan

# region weight-functions
def y_dist(y_i: float, y_j: float) -> float:
    return y_j - y_i

def x_dist(x_i: int, x_j: int) -> int:
    return x_j - x_i

def slope(y_i: float, y_j: float, x_i: int, x_j: int) -> float:
    return y_dist(y_i, y_j) / x_dist(x_i, x_j)

def angle(y_i: float, y_j: float, x_i: int, x_j: int) -> float:
    return atan(slope(y_i, y_j, x_i, x_j))
# endregion weight-functions

# region visibility-graphs

def get_weighted_edges(edge_list:list, time_series: np.ndarray, weight_func: callable) -> list:
    weighted_edges = []
    for i, j in edge_list:
        match weight_func.__name__:
            case 'y_dist':
                w = weight_func(time_series[i], time_series[j])
            case 'x_dist':
                w = weight_func(i, j)
            case 'slope' | 'angle':
                w = weight_func(time_series[i], time_series[j], i, j)
            case _:
                raise ValueError("Invalid weight function")
        weighted_edges.append((i, j, w))    
    return weighted_edges    

def generate_nvg_from(time_series: np.ndarray, weight_func: Optional[Callable] = None) -> nx.Graph:
    """Generates a natural visibility graph from a time series
    """
    g = nx.Graph()
    n = len(time_series)
    edges = [(i, j)
             for i in range(n)
             for j in range(i + 1, n)
             if all(time_series[k] < time_series[i] + (time_series[j] - time_series[i]) * (k - i) / (j - i)
                    for k in range(i + 1, j))]
    if weight_func == None:
        g.add_edges_from(edges)
    else:
        weighted_edges = get_weighted_edges(edges, time_series, weight_func)
        g.add_weighted_edges_from(weighted_edges)
    return g

def generate_hvg_from(time_series: np.ndarray, weight_func: Optional[Callable] = None) -> nx.Graph:
    """Generates a horizontal visibility graph from a time series
    """
    g = nx.Graph()
    n = len(time_series)
    edges = [(i, j)
             for i in range(n)
             for j in range(i + 1, n)
             if all(time_series[k] < min(time_series[i], time_series[j]) 
                    for k in range(i + 1, j))]
    if weight_func == None:
        g.add_edges_from(edges)
    else:
        weighted_edges = get_weighted_edges(edges, time_series, weight_func)
        g.add_weighted_edges_from(weighted_edges)
    return g

#endregion visibility-graphs

if __name__ == '__main__':
    rnd_ts = np.random.uniform(1, 100, 1000)
    g = generate_hvg_from(rnd_ts, angle)
    print(g)

    