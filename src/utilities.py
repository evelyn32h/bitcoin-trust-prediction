import numpy as np
import logging
logger = logging.getLogger(__name__)

def print_feature_statistics(features):
    """
    Print statistics (min, max, mean, and count of zeros) for each feature across all edges.
    Parameters:
    features: dict mapping edge tuples to lists of feature values (one list per edge)
    """
    feature_matrix = np.array(list(features.values()))
    n_features = feature_matrix.shape[1]
    print(f"Feature statistics for {n_features} features:")
    for i in range(n_features):
        col = feature_matrix[:, i]
        min_val = np.min(col)
        max_val = np.max(col)
        mean_val = np.mean(col)
        zero_count = np.sum(col == 0)
        print(f"Feature {i+1}: min={min_val}, max={max_val}, mean={mean_val}, zeros={zero_count}/{len(col)}")
