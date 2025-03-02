import numpy as np
import pandas as pd
from scipy.stats import kstest

def gower_dist_for(a: pd.Series, b: pd.Series) -> float:
    if len(a) != len(b):
        raise ValueError("Series must have the same number of features.")
    
    n_features = len(a)
    similarities = np.zeros(n_features)
    weights = np.zeros(n_features)

    ranges = {}
    for col in a.index:
        if isinstance(a[col], (int, float)) and isinstance(b[col], (int, float)):
            ranges[col] = max(a[col], b[col]) - min(a[col], b[col])

    for col in a.index:
        if (isinstance(a[col], (int, float)) and 
            isinstance(b[col], (int, float)) and
            is_continuous(a[col]) and
            is_continuous(b[col])):
            r = ranges[col]
            if r == 0:
              similarities[a.index.get_loc(col)] = 0.0
            else:
              similarities[a.index.get_loc(col)] = abs(a[col] - b[col]) / r
            weights[a.index.get_loc(col)] = 1.0
        elif a[col] == b[col]: # a categorical feature in which its values are equal
            similarities[a.index.get_loc(col)] = 1.0
            weights[a.index.get_loc(col)] = 1.0
        else: # a categorical feature in which its values are not equal
            similarities[a.index.get_loc(col)] = 0.0
            weights[a.index.get_loc(col)] = 1.0

    return np.sum(similarities * weights) / np.sum(weights)

def is_continuous(col: pd.Series, threshold: float = 0.7, use_stat_test: bool = False) -> bool:
    if not pd.api.types.is_numeric_dtype(col):
        return False
    
    # Kolmogorov-Smirnov test
    if use_stat_test:
        _, ks_pvalue = kstest(col, 'norm')
        if ks_pvalue > 0.05:
            return True

    # Check for floating-point numbers
    # If the column contains at least one floating-point number
    if (col % 1 != 0).any():
        return True
    # Check unique value ratio
    unique_ratio = col.nunique() / len(col)
    if unique_ratio > threshold:
        return True
    
    return False

def gower_similarity_matrix(dataframe: pd.DataFrame) -> pd.DataFrame:
    distance_df = pd.DataFrame(0, index=dataframe.index, columns=dataframe.index)

    for i in range(len(dataframe)):
        index_i = dataframe.index[i] 
        for j in range(i + 1, len(dataframe)):
            row_i = dataframe.iloc[i]
            row_j = dataframe.iloc[j]
            dist = gower_dist_for(row_i, row_j)
            distance_df.at[index_i, dataframe.index[j]] = dist
            distance_df.at[dataframe.index[j], index_i] = dist

    d_max = distance_df.values.max()
    similarity_df = 1 - (distance_df / d_max)
    return similarity_df


if __name__ == "__main__": 
    pass