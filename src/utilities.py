import random
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

def sample_edges_with_positive_ratio(G, sample_size, pos_ratio=0.5):
    """
    Sample a subset of edges from a NetworkX graph with a specified ratio of positive to negative edges.
    
    Parameters:
        G (networkx.Graph): Input graph with edge attribute 'weight' indicating sign (+/-).
        sample_size (int): Total number of edges to sample.
        pos_ratio (float): Desired ratio of positive edges (0 < pos_ratio < 1).
    Returns:
        list: Sampled list of (u, v, data) edge tuples.
    """
    edge_list = list(G.edges(data=True))
    # Separate positive and negative edges
    pos_edges = [e for e in edge_list if e[2].get('weight', 1) > 0]
    neg_edges = [e for e in edge_list if e[2].get('weight', 1) < 0]
    
    n_pos = int(round(sample_size * pos_ratio))
    n_neg = sample_size - n_pos
    
    # If not enough edges, take as many as possible
    orig_n_pos, orig_n_neg = n_pos, n_neg
    n_pos = min(n_pos, len(pos_edges))
    n_neg = min(n_neg, len(neg_edges))
    if n_pos < orig_n_pos or n_neg < orig_n_neg:
        logger.warning(f"Requested {orig_n_pos} positive and {orig_n_neg} negative edges, but only {n_pos} positive and {n_neg} negative edges available.")
    
    random.seed(42)
    pos_sample = random.sample(pos_edges, n_pos) if n_pos > 0 else []
    neg_sample = random.sample(neg_edges, n_neg) if n_neg > 0 else []
    
    sample = pos_sample + neg_sample
    random.shuffle(sample)
    return sample

def sample_n_edges(G, sample_size=None):
    """
    Sample a subset of edges from a NetworkX graph without enforcing a positive/negative ratio.
    If sample_size is None, returns all edges. Otherwise, returns a random sample of the given size.
    Prints and returns the number of positive and negative edges in the sample.
    Returns:
        list: Sampled list of (u, v, data) edge tuples.
    """
    import random
    edge_list = list(G.edges(data=True))
    if sample_size is not None and sample_size < len(edge_list):
        random.seed(42)
        sample = random.sample(edge_list, sample_size)
    else:
        sample = edge_list
    pos_edges = [e for e in sample if e[2].get('weight', 1) > 0]
    neg_edges = [e for e in sample if e[2].get('weight', 1) < 0]
    print(f"Sampled {len(sample)} edges: {len(pos_edges)} positive, {len(neg_edges)} negative")
    return sample, pos_edges, neg_edges
