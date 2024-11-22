import pandas as pd
import networkx as nx
import gower 

def _gower_similarity_graph(data: pd.DataFrame) -> nx.Graph:
    distance_matrix = gower.gower_matrix(data)
    similarity_matrix = 1 - distance_matrix

    node_count = len(data)
    g = nx.Graph()
    g.add_nodes_from(range(node_count))

    for i in range(node_count):
        for j in range(i + 1, node_count):  
            g.add_edge(i, j, weight=similarity_matrix[i, j])

    return g

def _gaussian_kernel_similarity_graph():
    pass

def _mahalanobis_similarity_graph():
    pass


def generate(data: pd.DataFrame, similarity_method: str) -> nx.Graph:
    """
    Generate a similarity graph based on the specified similarity method.

    Parameters:
    ----------
    data : pd.DataFrame
        A pandas DataFrame containing the data points for which the similarity 
        graph is to be generated. 

    similarity_method : str
        A string indicating the method to use for computing similarities. 
        Supported methods include:
            - 'gower': Uses Gower's similarity coefficient for mixed data 

    Returns:
    -------
    nx.Graph
        A NetworkX Graph object representing the similarity graph.

    Raises:
    ------
    ValueError
        If an unsupported similarity method is specified

    Examples:
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({'feature1': [1, 2], 'feature2': ['A', 'B']})
    >>> graph = generate(data, 'gower')


    """
    match similarity_method:
        case 'gower':
            return _gower_similarity_graph(data)
        case _:
            raise ValueError(f'Unsupported Similarity Method: {similarity_method}')

if __name__ == "__main__":
    df = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
        'Age': [25, 30, 35, 40, 28], 
        'Gender': ['Female', 'Male', 'Male', 'Male', 'Female'], 
        'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], 
        'Income': [50000, 60000, 70000, 80000, 55000],  
        'Employed': [1, 1, 0, 1, 0] 
    })

    graph = generate(df, 'gower')
    print(graph)
    print('Is Weighted?', nx.is_weighted(graph))
    print('Is Complete?', graph.number_of_edges() == graph.number_of_nodes() * (graph.number_of_nodes() - 1) // 2)