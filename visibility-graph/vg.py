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
    """
    Generates a natural visibility graph from a time series

    Reference:
        Lacasa, L., Luque, B., Ballesteros, F., Luque, J., & Nuno, J. C. (2008).
        From time series to complex networks: The visibility graph.
        Proceedings of the National Academy of Sciences, 105(13), 4972-4975.
        https://doi.org/10.1073/pnas.0709247105
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
    """
    Generates a horizontal visibility graph from a time series
    
    Reference:
        B. Luque, L. Lacasa, F. Ballesteros, and J. Luque, "Horizontal visibility graphs: 
        Exact results for random time series," Phys. Rev. E 80, 046103 (2009).
        DOI: https://doi.org/10.1103/PhysRevE.80.046103
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

def generate_lpvg_from(ts: np.ndarray, penetrable_limit: int = 1, weight_func: Optional[Callable] = None) -> nx.Graph:
    """
    Generates a limited penetrable visibility graph from a time series

    Reference:
        Zhou, T. T., Jin, N. D., Gao, Z. K., & Luo, Y. B. "Limited penetrable visibility graph
        for establishing complex network from time series," Acta Physica Sinica, 61(3), 030506 (2012).
        DOI: 10.7498/aps.61.030506
    """
    g = nx.Graph()
    n = len(ts)

    edges = [(i, j)
             for i in range(n)
             for j in range(i+1, n)
             if all(ts[i+l] < ts[j]+(ts[i]-ts[j])*((j-(i+l))/(j-i))
                    for l in range(1, min(penetrable_limit+1, j-i))) # Ensuring l < j - i
            ]
    if weight_func == None:
        g.add_edges_from(edges)
    else:
        weighted_edges = get_weighted_edges(edges, ts, weight_func)
        g.add_weighted_edges_from(weighted_edges)
    return g

#endregion visibility-graphs

if __name__ == '__main__':
    rnd_ts = np.random.uniform(1, 100, 1000)
    nvg = generate_nvg_from(time_series=rnd_ts)
    hvg = generate_hvg_from(time_series=rnd_ts)
    lpvg = generate_lpvg_from(ts=rnd_ts, penetrable_limit=5)
    print('NVG:', nvg)
    print('HVG:', hvg)
    print('LPVG:', lpvg)



    