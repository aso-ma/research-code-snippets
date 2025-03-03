import numpy as np
import pandas as pd
from scipy.stats import kstest, anderson
from io import StringIO

def gower_score_for(a: pd.Series, b: pd.Series) -> float:
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
        if is_continuous(col):
            r = ranges[col]
            if r == 0:
              similarities[a.index.get_loc(col)] = 0.0
            else:
              similarities[a.index.get_loc(col)] = 1 - abs(a[col] - b[col]) / r
            weights[a.index.get_loc(col)] = 1.0 # comparable
        else: # categorical or binary
            if a[col] == b[col]:
                 similarities[a.index.get_loc(col)] = 1.0
            else:
                 similarities[a.index.get_loc(col)] = 0.0
            weights[a.index.get_loc(col)] = 1.0 # comparable

    return np.sum(similarities * weights) / np.sum(weights)

def is_continuous(col: pd.Series, unique_ratio_threshold: float = 0.8) -> bool:
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

def gower_similarity_matrix(dataframe: pd.DataFrame, dist: bool = False) -> pd.DataFrame:
    similarity_df = pd.DataFrame(0.0, index=dataframe.index, columns=dataframe.index)

    for i in range(len(dataframe)):
        index_i = dataframe.index[i] 
        for j in range(i + 1, len(dataframe)):
            row_i = dataframe.iloc[i]
            row_j = dataframe.iloc[j]
            score = gower_score_for(row_i, row_j)
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
    
    similarity_df = gower_similarity_matrix(df)
    print(similarity_df)

