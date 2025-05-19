import networkx as nx
import numpy as np
import random

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

def to_undirected(G):
    """
    Convert a directed graph to an undirected graph using NetworkX's to_undirected().
    Parameters:
        G: A NetworkX graph (typically DiGraph)
    Returns:
        An undirected version of G (NetworkX Graph)
    """
    #TODO: Handle assymetric relationships
    return G.to_undirected()

def edge_random_holdout(G, n_edges, random_state=None):
    """
    Hold out a random set of n_edges from the graph as the test set.
    Returns (G_train, G_test) where G_train has the edges removed and G_test contains only the held-out edges.
    
    Parameters:
        G: NetworkX Graph (undirected or directed)
        n_edges: Number of edges to hold out for testing
        random_state: Optional random seed for reproducibility
    Returns:
        G_train: Graph with held-out edges removed
        G_test: Graph with only the held-out edges
    """
    if random_state is not None:
        random.seed(random_state)
    all_edges = list(G.edges(data=True))
    n_edges = min(n_edges, len(all_edges))
    held_out_edges = random.sample(all_edges, n_edges)
    held_out_edge_tuples = [(u, v) for u, v, _ in held_out_edges]
    
    # Create train graph
    G_train = G.copy()
    G_train.remove_edges_from(held_out_edge_tuples)
    
    # Create test graph
    G_test = G.__class__()
    G_test.add_nodes_from(G.nodes(data=True))
    for u, v, data in held_out_edges:
        G_test.add_edge(u, v, **data)
    
    return G_train, G_test

def edge_bfs_holdout_split(G, seed_edge, n_edges):
    """
    Hold out n_edges as close as possible (in edge-space) to the seed_edge using BFS in the line graph.
    Returns (G_train, G_test) where G_train has the edges removed and G_test contains only the held-out edges.
    
    Parameters:
        G: NetworkX Graph (undirected or directed)
        seed_edge: The starting edge (u, v) for BFS in the line graph
        n_edges: Number of edges to hold out for testing
    Returns:
        G_train: Graph with held-out edges removed
        G_test: Graph with only the held-out edges
    """
    L = nx.line_graph(G)
    if seed_edge not in L:
        raise ValueError("seed_edge must be an existing edge tuple")

    # Use networkx's bfs_tree for concise BFS traversal
    bfs_order = list(nx.bfs_tree(L, seed_edge))
    bfs_edges = bfs_order[:n_edges]

    if len(bfs_edges) < n_edges:
        print(f"Warning: Only {len(bfs_edges)} edges found via BFS from seed_edge, but {n_edges} requested. Returning all reachable edges.")
        
    # Prepare held-out edges with data
    held_out_edges = [(u, v, G.get_edge_data(u, v)) for (u, v) in bfs_edges if G.has_edge(u, v)]
    held_out_edge_tuples = [(u, v) for u, v, _ in held_out_edges]

    # Create train graph
    G_train = G.copy()
    G_train.remove_edges_from(held_out_edge_tuples)

    # Create test graph
    G_test = G.__class__()
    for u, v, data in held_out_edges:
        G_test.add_edge(u, v, **data)

    return G_train, G_test

def edge_ego_holdout(G, seed_edge, hops=2):
    """
    Hold out the h-hop neighborhood (in edge-space) around seed_edge.
    Requires undirected graph as input
    """
    L = nx.line_graph(G)
    if seed_edge not in L:
        raise ValueError("seed_edge must be an existing edge tuple")
    test_edges = set(nx.single_source_shortest_path_length(L, seed_edge, cutoff=hops).keys())
    G_train = G.copy()
    G_train.remove_edges_from(test_edges)
    G_test = nx.Graph()
    G_test.add_nodes_from(G.nodes(data=True))
    G_test.add_edges_from(test_edges)
    return G_train, G_test

def sample_random_seed_edges(G, n=5, random_state=None):
    """
    Select n random edges from graph G.
    Parameters:
        G: NetworkX Graph
        n: Number of edges to sample
        random_state: Optional random seed for reproducibility
    Returns:
        List of n randomly sampled edges (as (u, v) tuples)
    """
    import random
    if random_state is not None:
        random.seed(random_state)
    edges = list(G.edges())
    return random.sample(edges, min(n, len(edges)))

