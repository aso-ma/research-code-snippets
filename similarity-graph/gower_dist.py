import numpy as np
import pandas as pd
from scipy.stats import kstest, anderson
from io import StringIO
from typing import Dict, Literal, Tuple, Optional

ColumnType = Literal['continuous', 'nominal', 'ordinal']

def is_continuous(col: pd.Series, unique_ratio_threshold: float = 0.7) -> bool:
    if not pd.api.types.is_numeric_dtype(col):
        return False
    
    if (col % 1 != 0).any(): # If the column contains at least one floating-point number
        return True
    
    # Statistical Test
    # Apply both Anderson-Darling and Kolmogorov-Smirnov tests
    distributions = ['norm', 'expon', 'logistic', 'gumbel', 'gamma']
    for dist in distributions:
        try:
            ad_result = anderson(col, dist)
            _, ks_p = kstest(col, dist, args=(col.mean(), col.std()))
            if ad_result.statistic < ad_result.critical_values[2] and ks_p > 0.05: 
                # Check if test statistic is below 5% significance threshold (good fit) 
                # And
                # Fail to reject H_0 (there is no strong evidence against normality)
                # So at least one continuous distribution fits well
                return True 
        except:
            # Some distributions may not work with certain data, skip them
            continue
    
    # Check unique value ratio
    unique_ratio = col.nunique() / len(col)
    if unique_ratio > unique_ratio_threshold:
        return True
    
    return False

def gower_score_for(
    a: pd.Series, 
    b: pd.Series, 
    column_types: Dict[str, ColumnType], 
    range_dict: Dict[str, float], 
    rank_dict: Dict[str, Tuple[pd.Series, int, int]]
) -> float:
    if len(a) != len(b):
        raise ValueError("Series must have the same number of features.")
    
    n_features = len(a)
    similarities = np.zeros(n_features)
    weights = np.zeros(n_features)

    for col in a.index:
        if column_types[col] == 'continuous':
            r = range_dict[col]
            similarities[a.index.get_loc(col)] = 0.0 if r == 0 else 1 - abs(a[col] - b[col]) / r
            weights[a.index.get_loc(col)] = 1.0
        if column_types[col] == 'nominal': 
            similarities[a.index.get_loc(col)] = 1.0 if a[col] == b[col] else 0.0
            weights[a.index.get_loc(col)] = 1.0 
        if column_types[col] == 'ordinal':
            if a[col] == b[col]:
                similarities[a.index.get_loc(col)] = 1.0
            else:
                ranks, count_max, count_min, max_rank, min_rank = rank_dict[col]
                rank_i = ranks.loc[a.name]
                rank_j = ranks.loc[b.name]
                count_rank_i = len(ranks[ranks == rank_i])
                count_rank_j = len(ranks[ranks == rank_j])
                sim = (
                    1 - 
                    (abs(rank_i - rank_j) - ((count_rank_i - 1)/2) - ((count_rank_j-1)/2)) / 
                    (max_rank - min_rank - ((count_max-1)/2) - ((count_min-1)/2))
                )
                similarities[a.index.get_loc(col)] = sim
            weights[a.index.get_loc(col)] = 1.0 

    return np.sum(similarities * weights) / np.sum(weights)

def gower_similarity_matrix(dataframe: pd.DataFrame, column_types: Optional[Dict[str, ColumnType]] = None, dist: bool = False) -> pd.DataFrame:
    """
    This method computes the Gower's similarity matrix for an input dataframe.    

    References:
        - Podani, J.: Extending Gower's general coefficient of similarity to ordinal characters. - Taxon 48: 331-340. 1999. - ISSN 0040-0262. https://doi.org/10.2307/1224438

    Acknowledgments:
        - Parsa Azimi (pazimi730@gmail.com): Reviewed the implementation and provided valuable feedback on the algorithm.
    """
    
    similarity_df = pd.DataFrame(0.0, index=dataframe.index.values, columns=dataframe.index.values)
    np.fill_diagonal(similarity_df.values, 1)


    if column_types == None:
        column_types = dict()
        for col in dataframe.columns:
            column_types[col] = 'continuous' if is_continuous(col) else 'nominal'
        
    range_dict = dict()
    for col in dataframe.columns:
        if column_types[col] == 'continuous':
            range_dict[col] = dataframe[col].max() - dataframe[col].min()

    rank_dict = dict()
    for col in dataframe.columns:
        if column_types[col] == 'ordinal':
            ranks = dataframe[col].rank(method='dense', ascending=True)
            max_rank = ranks.max()
            min_rank = ranks.min()
            max_count = len(ranks[ranks == ranks.max()])
            min_count = len(ranks[ranks == ranks.min()])
            rank_dict[col] = (ranks, max_count, min_count, max_rank, min_rank)

    for i in range(len(dataframe)):
        index_i = dataframe.index[i] 
        for j in range(i + 1, len(dataframe)):
            row_i = dataframe.iloc[i]
            row_j = dataframe.iloc[j]
            score = gower_score_for(row_i, row_j, column_types, range_dict, rank_dict)
            similarity_df.at[index_i, dataframe.index[j]] = score
            similarity_df.at[dataframe.index[j], index_i] = score

    return  np.sqrt(1 - similarity_df) if dist else similarity_df


if __name__ == "__main__": 
    data = """Subject ID,Age,Handedness,Eye Colour,Knows Python
    001,28,Right,Blue,Yes
    002,34,Left,Blue,No
    003,22,Right,Green,Yes
    004,45,Right,Hazel,No
    005,30,Left,Brown,Yes"""

    df = pd.read_csv(StringIO(data), index_col="Subject ID")
    
    col_types: dict[str, ColumnType] = {
        "Age": 'continuous',
        "Handedness": 'nominal',
        "Eye Colour": 'nominal',
        "Knows Python": 'nominal',
    }
    similarity_df = gower_similarity_matrix(df, col_types)
    print(similarity_df)


