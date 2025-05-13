import networkx as nx
import numpy as np

def map_to_unweighted_graph(G):
    """
    Convert weighted directed graph to signed directed graph and ensure node IDs are integers.
    
    Parameters:
    G: Original weighted directed graph
    
    Returns:
    G_signed: Signed directed graph, edge weights are +1 or -1
    """
    # Relabel nodes to ensure IDs are integers
    G = nx.relabel_nodes(G, lambda x: int(x) if isinstance(x, float) else x)
    
    G_signed = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        # Get original weight
        weight = data['weight']
        
        # Convert to sign: positive → +1, negative → -1
        sign = 1 if weight > 0 else -1
        
        # Add signed edge
        G_signed.add_edge(u, v, weight=sign, time=data['time'])
    
    return G_signed

def ensure_connectivity(G):
    """
    Ensure the graph is weakly connected, if not, keep only the largest weakly connected component
    
    Parameters:
    G: Input graph
    
    Returns:
    largest_cc: Subgraph of the largest weakly connected component
    """
    # Find all weakly connected components
    connected_components = list(nx.weakly_connected_components(G))
    
    # If there's only one connected component, return the original graph
    if len(connected_components) == 1:
        return G
    
    # Find the largest connected component
    largest_cc = max(connected_components, key=len)
    
    # Extract subgraph
    return G.subgraph(largest_cc).copy()

def filter_neutral_edges(G, threshold=1):
    """
    Remove neutral edges with weights close to 0 (absolute value less than threshold)
    
    Parameters:
    G: Input graph
    threshold: Absolute value threshold for edge weights
    
    Returns:
    G_filtered: Filtered graph
    """
    G_filtered = G.copy()
    edges_to_remove = []
    
    for u, v, data in G.edges(data=True):
        if abs(data['weight']) < threshold:
            edges_to_remove.append((u, v))
    
    G_filtered.remove_edges_from(edges_to_remove)
    return G_filtered

def reindex_nodes(G):
    """
    Reindex all nodes in the graph so that they are sequential from 0 to n-1.
    
    Parameters:
    G: Input graph
    
    Returns:
    G_reindexed: Graph with nodes reindexed sequentially
    """
    mapping = {old_index: new_index for new_index, old_index in enumerate(G.nodes())}
    G_reindexed = nx.relabel_nodes(G, mapping)
    return G_reindexed

def normalize_edge_weights(G):
    """
    Normalize all edge weights in the graph to be between -1 and 1.
    
    Parameters:
    G: NetworkX graph
    
    Returns:
    G_normalized: Graph with normalized edge weights
    """
    G_normalized = G.copy()
    weights = [data['weight'] for _, _, data in G.edges(data=True)]
    
    if not weights:  # If there are no edges, return the graph as is
        return G_normalized
    
    max_weight = max(weights)
    min_weight = min(weights)
    
    for u, v, data in G_normalized.edges(data=True):
        if max_weight == min_weight:  # Avoid division by zero if all weights are the same
            data['weight'] = 0
        else:
            data['weight'] = 2 * (data['weight'] - min_weight) / (max_weight - min_weight) - 1
    
    return G_normalized

def filter_by_embeddedness(G, min_embeddedness=1):
    """
    Filter edges by minimum embeddedness threshold
    
    Parameters:
    G: NetworkX graph
    min_embeddedness: Minimum number of common neighbors required
    
    Returns:
    G_filtered: Graph with filtered edges
    """
    # Create a copy of the graph
    G_filtered = nx.DiGraph()
    
    # Create undirected graph to calculate embeddedness
    G_undirected = G.to_undirected()
    
    # Filter edges based on embeddedness
    edges_kept = []
    for u, v, data in G.edges(data=True):
        # Calculate embeddedness (number of common neighbors)
        common_neighbors = set(G_undirected.neighbors(u)) & set(G_undirected.neighbors(v))
        
        # Keep edge if embeddedness is at least min_embeddedness
        if len(common_neighbors) >= min_embeddedness:
            G_filtered.add_edge(u, v, **data)
            edges_kept.append((u, v))
    
    print(f"Original graph: {G.number_of_edges()} edges")
    print(f"Filtered graph: {G_filtered.number_of_edges()} edges")
    print(f"Removed {G.number_of_edges() - G_filtered.number_of_edges()} edges with embeddedness < {min_embeddedness}")
    
    return G_filtered

def create_balanced_dataset(G):
    """
    Create a balanced dataset with equal number of positive and negative edges
    
    Parameters:
    G: NetworkX graph
    
    Returns:
    G_balanced: Graph with balanced positive/negative edges
    """
    # Get positive and negative edges
    pos_edges = [(u, v, data) for u, v, data in G.edges(data=True) if data['weight'] > 0]
    neg_edges = [(u, v, data) for u, v, data in G.edges(data=True) if data['weight'] < 0]
    
    print(f"Original graph: {len(pos_edges)} positive edges, {len(neg_edges)} negative edges")
    print(f"Positive/Negative ratio: {len(pos_edges) / len(neg_edges):.2f}")
    
    # Sample positive edges to match negative edges count
    import random
    random.seed(42)  # For reproducibility
    
    # If we have more positive than negative edges
    if len(pos_edges) > len(neg_edges):
        sampled_pos_edges = random.sample(pos_edges, len(neg_edges))
        final_pos_edges = sampled_pos_edges
        final_neg_edges = neg_edges
    # If we have more negative than positive edges (unlikely in this dataset)
    else:
        sampled_neg_edges = random.sample(neg_edges, len(pos_edges))
        final_pos_edges = pos_edges
        final_neg_edges = sampled_neg_edges
    
    # Create new balanced graph
    G_balanced = nx.DiGraph()
    
    # Add positive edges
    for u, v, data in final_pos_edges:
        G_balanced.add_edge(u, v, **data)
    
    # Add negative edges
    for u, v, data in final_neg_edges:
        G_balanced.add_edge(u, v, **data)
    
    print(f"Balanced graph: {len(final_pos_edges)} positive edges, {len(final_neg_edges)} negative edges")
    
    return G_balanced