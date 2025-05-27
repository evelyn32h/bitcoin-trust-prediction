import networkx as nx
import numpy as np
import random
from collections import deque
import time

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
    Filter edges by minimum embeddedness threshold using NetworkX's optimized common_neighbors function.
    Assumes the graph is undirected for embeddedness calculation.
    
    Parameters:
    G: NetworkX graph (assumes undirected)
    min_embeddedness: Minimum number of common neighbors required
    
    Returns:
    G_filtered: Graph with filtered edges (same type as input)
    """
    # Create a copy with the same type as input
    G_filtered = G.__class__()
    G_filtered.add_nodes_from(G.nodes(data=True))
    
    # Print warning if graph is directed and convert to undirected for embeddedness calculation
    if G.is_directed():
        print("Warning: Graph is directed. Converting to undirected for embeddedness calculation.")
        G_for_embeddedness = G.to_undirected()
    else:
        G_for_embeddedness = G
    
    # Filter edges based on embeddedness
    edges_kept = []
    for u, v, data in G.edges(data=True):
        # Calculate embeddedness using NetworkX's optimized common_neighbors
        embeddedness = len(list(nx.common_neighbors(G_for_embeddedness, u, v)))
        
        # Keep edge if embeddedness is at least min_embeddedness
        if embeddedness >= min_embeddedness:
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

def optimized_bfs_sampling(G, target_edges, seed_selection='random_moderate_degree', degree_percentile=70):
    """
    Optimized BFS sampling for large graphs following Requirement's strategy.
    Same method as used for Bitcoin OTC sampling but optimized for Epinions scale.
    
    Parameters:
    G: NetworkX directed graph
    target_edges: Target number of edges in the subset
    seed_selection: Method to select seed node ('random_moderate_degree', 'random')
    degree_percentile: Percentile for moderate degree selection (avoid highest degree)
    
    Returns:
    G_subset: Subgraph with approximately target_edges edges
    """
    print(f"Starting optimized BFS sampling...")
    print(f"Target: {target_edges} edges (same strategy as Bitcoin OTC)")
    print(f"Seed selection: {seed_selection}")
    
    start_time = time.time()
    
    # Convert to undirected for BFS (preserves connectivity better)
    if G.is_directed():
        G_undirected = G.to_undirected()
    else:
        G_undirected = G
    
    # Select seed node based on strategy
    if seed_selection == 'random_moderate_degree':
        # Get degree distribution and select from moderate-degree nodes
        # Avoid highest degree nodes as Requirement suggested
        degrees = dict(G_undirected.degree())
        degree_values = list(degrees.values())
        degree_threshold = np.percentile(degree_values, degree_percentile)
        
        # Select nodes with degree >= threshold but not the absolute highest
        candidate_nodes = [node for node, deg in degrees.items() 
                          if deg >= degree_threshold and deg < max(degree_values)]
        
        if not candidate_nodes:
            # Fallback to all nodes if filtering is too strict
            candidate_nodes = list(G_undirected.nodes())
        
        seed_node = random.choice(candidate_nodes)
        seed_degree = degrees[seed_node]
        print(f"Selected seed node {seed_node} with degree {seed_degree} (percentile: {degree_percentile})")
        
    else:  # random selection
        seed_node = random.choice(list(G_undirected.nodes()))
        print(f"Selected random seed node: {seed_node}")
    
    # Optimized BFS implementation
    visited_nodes = set()
    visited_edges = set()
    queue = deque([seed_node])
    visited_nodes.add(seed_node)
    
    print("Running BFS traversal...")
    
    # BFS with edge counting
    while queue and len(visited_edges) < target_edges:
        current_node = queue.popleft()
        
        # Get neighbors and process them
        neighbors = list(G_undirected.neighbors(current_node))
        
        # Shuffle neighbors for randomness
        random.shuffle(neighbors)
        
        for neighbor in neighbors:
            if len(visited_edges) >= target_edges:
                break
                
            # Add edge if not already counted
            edge = tuple(sorted([current_node, neighbor]))
            if edge not in visited_edges:
                visited_edges.add(edge)
                
                # Add neighbor to queue if not visited
                if neighbor not in visited_nodes:
                    visited_nodes.add(neighbor)
                    queue.append(neighbor)
        
        # Progress reporting
        if len(visited_edges) % 5000 == 0:
            elapsed = time.time() - start_time
            print(f"  Progress: {len(visited_edges)}/{target_edges} edges, "
                  f"{len(visited_nodes)} nodes, {elapsed:.1f}s")
    
    # Create subgraph from visited nodes
    print("Creating subgraph from BFS result...")
    subgraph_nodes = visited_nodes
    
    # Extract subgraph from original directed graph
    G_subset = G.subgraph(subgraph_nodes).copy()
    
    # If we have more edges than target, randomly sample to exact target
    if G_subset.number_of_edges() > target_edges:
        print(f"Subgraph has {G_subset.number_of_edges()} edges, sampling to {target_edges}")
        all_edges = list(G_subset.edges(data=True))
        sampled_edges = random.sample(all_edges, target_edges)
        
        # Create new graph with sampled edges
        G_final = G.__class__()
        G_final.add_nodes_from(G_subset.nodes(data=True))
        for u, v, data in sampled_edges:
            G_final.add_edge(u, v, **data)
        G_subset = G_final
    
    elapsed = time.time() - start_time
    
    print(f"BFS sampling completed in {elapsed:.1f} seconds:")
    print(f"  Nodes selected: {G_subset.number_of_nodes()}")
    print(f"  Edges selected: {G_subset.number_of_edges()}")
    print(f"  Target achieved: {G_subset.number_of_edges() / target_edges:.1%}")
    
    # Analyze connectivity preservation
    if G.is_directed():
        original_components = nx.number_weakly_connected_components(G)
        subset_components = nx.number_weakly_connected_components(G_subset)
    else:
        original_components = nx.number_connected_components(G)
        subset_components = nx.number_connected_components(G_subset)
    
    print(f"  Connectivity: {subset_components} components (original: {original_components})")
    
    return G_subset

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

# TASK #1: NEW WEIGHTED FEATURE FUNCTIONS

def label_edges(G):
    """
    Label edges in the graph based on their weights.
    Positive weights are labeled as 1, negative weights as -1, and zero weights as 0.
    
    Parameters:
    G: NetworkX DiGraph with 'weight' edge attribute
    
    Returns:
    G_labeled: Graph with 'label' edge attribute
    """
    G_labeled = G.copy()
    
    for u, v, data in G_labeled.edges(data=True):
        weight = data['weight']
        if weight > 0:
            label = 1
        elif weight < 0:
            label = -1
        else:
            label = 0
        data['label'] = label
    
    return G_labeled

def transform_weights(G, use_weighted_features=True, weight_method='raw', weight_bins=5):
    """
    Transform edge weights for machine learning tasks.
    
    Parameters:
    - G: NetworkX DiGraph with 'weight' and 'time' edge attributes
    - weight_method: 'sign' (binary ±1), 'raw' (preserve weights), 'binned' (discretize)
    - weight_bins: Number of bins for 'binned' method
    
    Returns:
    - G_processed: Graph with transformed weights and preserved original data
    """
    print(f"Processing weights using method: {weight_method}")
    
    # Relabel nodes to ensure IDs are integers
    G = nx.relabel_nodes(G, lambda x: int(x) if isinstance(x, float) else x)
    
    G_processed = nx.DiGraph()
    
    if (not use_weighted_features) or weight_method == 'sign':
        # Original binary method (Task #1 baseline for comparison)
        for u, v, data in G.edges(data=True):
            weight = data['weight']
            sign = 1 if weight > 0 else -1
            G_processed.add_edge(u, v, weight=sign, original_weight=weight, time=data['time'])
            
    elif weight_method == 'raw':
        # New weighted method (Task #1 main implementation)
        for u, v, data in G.edges(data=True):
            weight = data['weight']
            G_processed.add_edge(u, v, weight=weight, original_weight=weight, time=data['time'])
            
    elif weight_method == 'binned':
        # Discretized weighted method (Task #1 alternative)
        weights = [data['weight'] for _, _, data in G.edges(data=True)]
        min_weight, max_weight = min(weights), max(weights)
        
        # Create bins
        bin_edges = np.linspace(min_weight, max_weight, weight_bins + 1)
        
        for u, v, data in G.edges(data=True):
            weight = data['weight']
            # Find which bin this weight belongs to
            bin_idx = np.digitize(weight, bin_edges) - 1
            bin_idx = max(0, min(bin_idx, weight_bins - 1))  # Ensure within bounds
            
            # Convert bin index to weight value (centered in bin)
            bin_center = (bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2
            G_processed.add_edge(u, v, weight=bin_center, original_weight=weight, time=data['time'])
    
    else:
        raise ValueError(f"Unknown weight_method: {weight_method}")
    
    print(f"Weight processing complete:")
    print(f"  Method: {weight_method}")
    print(f"  Nodes: {G_processed.number_of_nodes()}")
    print(f"  Edges: {G_processed.number_of_edges()}")
    
    processed_weights = [data['weight'] for _, _, data in G_processed.edges(data=True)]
    print(f"  Weight range: [{min(processed_weights):.3f}, {max(processed_weights):.3f}]")
    
    return G_processed

def handle_bidirectional_edges_weighted(G, method='max', preserve_weights=True):
    """
    Handle bidirectional edges while preserving weight information for Task #1.
    This is an enhanced version of handle_bidirectional_edges that works with weighted features.
    
    Parameters:
    G: NetworkX directed graph with weights
    method: Method for combining bidirectional edges
    preserve_weights: Whether to preserve original weight information
    
    Returns:
    G_undirected: NetworkX undirected graph with bidirectional edges combined
    stats: Dictionary with conversion statistics
    """
    print(f"Handling bidirectional edges (weighted) using method: {method}")
    
    # Track statistics for analysis
    stats = {
        'original_edges': G.number_of_edges(),
        'bidirectional_pairs': 0,
        'unidirectional_edges': 0,
        'final_edges': 0,
        'method_used': method,
        'sign_consistency_preserved': 0,
        'weight_preservation_quality': 0.0
    }
    
    # Find all bidirectional pairs and unidirectional edges
    bidirectional_pairs = []
    unidirectional_edges = []
    processed_pairs = set()
    
    for u, v, data in G.edges(data=True):
        # Skip if we've already processed this pair
        if (min(u, v), max(u, v)) in processed_pairs:
            continue
            
        if G.has_edge(v, u):  # Bidirectional edge found
            # Get data for both directions
            data_uv = G[u][v]
            data_vu = G[v][u]
            
            bidirectional_pairs.append((u, v, data_uv, data_vu))
            processed_pairs.add((min(u, v), max(u, v)))
            stats['bidirectional_pairs'] += 1
        else:  # Unidirectional edge
            unidirectional_edges.append((u, v, data))
            stats['unidirectional_edges'] += 1
    
    print(f"Found {stats['bidirectional_pairs']} bidirectional pairs and {stats['unidirectional_edges']} unidirectional edges")
    
    # Create new undirected graph
    G_undirected = nx.Graph()
    
    # Add all nodes with their attributes
    G_undirected.add_nodes_from(G.nodes(data=True))
    
    # Process bidirectional edges according to the specified method
    sign_consistency_count = 0
    
    for u, v, data_uv, data_vu in bidirectional_pairs:
        weight_uv = data_uv.get('weight', 0)
        weight_vu = data_vu.get('weight', 0)
        time_uv = data_uv.get('time', 0)
        time_vu = data_vu.get('time', 0)
        orig_uv = data_uv.get('original_weight', weight_uv)
        orig_vu = data_vu.get('original_weight', weight_vu)
        
        # Track sign consistency
        if (weight_uv > 0) == (weight_vu > 0):
            sign_consistency_count += 1
        
        if method == 'average':
            combined_weight = (weight_uv + weight_vu) / 2
            combined_time = (time_uv + time_vu) / 2
            combined_orig = (orig_uv + orig_vu) / 2
            
        elif method == 'sum':
            combined_weight = weight_uv + weight_vu
            combined_time = min(time_uv, time_vu)
            combined_orig = orig_uv + orig_vu
            
        elif method == 'max':
            if abs(weight_uv) >= abs(weight_vu):
                combined_weight = weight_uv
                combined_time = time_uv
                combined_orig = orig_uv
            else:
                combined_weight = weight_vu
                combined_time = time_vu
                combined_orig = orig_vu
                
        elif method == 'min':
            if abs(weight_uv) <= abs(weight_vu):
                combined_weight = weight_uv
                combined_time = time_uv
                combined_orig = orig_uv
            else:
                combined_weight = weight_vu
                combined_time = time_vu
                combined_orig = orig_vu
                
        elif method == 'stronger':
            if abs(weight_uv) > abs(weight_vu):
                combined_weight = weight_uv
                combined_time = time_uv
                combined_orig = orig_uv
            elif abs(weight_vu) > abs(weight_uv):
                combined_weight = weight_vu
                combined_time = time_vu
                combined_orig = orig_vu
            else:
                # If equal strength, average them
                combined_weight = (weight_uv + weight_vu) / 2
                combined_time = (time_uv + time_vu) / 2
                combined_orig = (orig_uv + orig_vu) / 2
                
        else:
            raise ValueError(f"Unknown bidirectional handling method: {method}")
        
        # Add the combined edge with preserved weight information
        edge_attrs = {'weight': combined_weight, 'time': combined_time}
        if preserve_weights:
            edge_attrs['original_weight'] = combined_orig
            
        G_undirected.add_edge(u, v, **edge_attrs)
    
    # Add unidirectional edges as-is
    for u, v, data in unidirectional_edges:
        G_undirected.add_edge(u, v, **data)
    
    # Calculate final statistics
    stats['final_edges'] = G_undirected.number_of_edges()
    stats['sign_consistency_preserved'] = sign_consistency_count
    
    print(f"Weighted conversion complete: {stats['original_edges']} → {stats['final_edges']} edges")
    print(f"  - {stats['bidirectional_pairs']} bidirectional pairs combined")
    print(f"  - {stats['unidirectional_edges']} unidirectional edges preserved")
    print(f"  - {sign_consistency_count}/{stats['bidirectional_pairs']} bidirectional pairs had consistent signs")
    
    return G_undirected, stats

# EXISTING BIDIRECTIONAL EDGE HANDLING FUNCTIONS (Task #2 - COMPLETED)

def handle_bidirectional_edges_binary(G, method='average'):
    """
    Handle bidirectional edges in a directed graph by combining them into undirected edges.
    This addresses teacher feedback about asymmetric relationships and implements 
    explicit methods for dealing with bidirectional edges as mentioned in the paper.
    
    Parameters:
    G: NetworkX directed graph
    method: str, method for combining bidirectional edges
        - 'average': Average the weights (recommended for trust networks)
        - 'sum': Sum the weights  
        - 'max': Take maximum absolute weight (keeping original sign)
        - 'min': Take minimum absolute weight (keeping original sign)
        - 'first': Keep the first edge encountered (deterministic but arbitrary)
        - 'stronger': Keep the edge with larger absolute weight
    
    Returns:
    G_undirected: NetworkX undirected graph with bidirectional edges combined
    stats: Dictionary with conversion statistics
    """
    print(f"Handling bidirectional edges using method: {method}")
    
    # Track statistics for analysis
    stats = {
        'original_edges': G.number_of_edges(),
        'bidirectional_pairs': 0,
        'unidirectional_edges': 0,
        'final_edges': 0,
        'method_used': method,
        'sign_consistency_preserved': 0,
        'weight_preservation_quality': 0.0
    }
    
    # Find all bidirectional pairs and unidirectional edges
    bidirectional_pairs = []
    unidirectional_edges = []
    processed_pairs = set()
    
    for u, v, data in G.edges(data=True):
        # Skip if we've already processed this pair
        if (min(u, v), max(u, v)) in processed_pairs:
            continue
            
        if G.has_edge(v, u):  # Bidirectional edge found
            # Get data for both directions
            data_uv = G[u][v]
            data_vu = G[v][u]
            
            bidirectional_pairs.append((u, v, data_uv, data_vu))
            processed_pairs.add((min(u, v), max(u, v)))
            stats['bidirectional_pairs'] += 1
        else:  # Unidirectional edge
            unidirectional_edges.append((u, v, data))
            stats['unidirectional_edges'] += 1
    
    print(f"Found {stats['bidirectional_pairs']} bidirectional pairs and {stats['unidirectional_edges']} unidirectional edges")
    
    # Create new undirected graph
    G_undirected = nx.Graph()
    
    # Add all nodes with their attributes
    G_undirected.add_nodes_from(G.nodes(data=True))
    
    # Process bidirectional edges according to the specified method
    sign_consistency_count = 0
    
    for u, v, data_uv, data_vu in bidirectional_pairs:
        weight_uv = data_uv.get('weight', 0)
        weight_vu = data_vu.get('weight', 0)
        time_uv = data_uv.get('time', 0)
        time_vu = data_vu.get('time', 0)
        
        # Track sign consistency
        if (weight_uv > 0) == (weight_vu > 0):
            sign_consistency_count += 1
        
        if method == 'average':
            combined_weight = (weight_uv + weight_vu) / 2
            combined_time = (time_uv + time_vu) / 2
            
        elif method == 'sum':
            combined_weight = weight_uv + weight_vu
            combined_time = min(time_uv, time_vu)  # Use earlier time
            
        elif method == 'max':
            if abs(weight_uv) >= abs(weight_vu):
                combined_weight = weight_uv
                combined_time = time_uv
            else:
                combined_weight = weight_vu
                combined_time = time_vu
                
        elif method == 'min':
            if abs(weight_uv) <= abs(weight_vu):
                combined_weight = weight_uv
                combined_time = time_uv
            else:
                combined_weight = weight_vu
                combined_time = time_vu
                
        elif method == 'first':
            # Use lexicographically first edge (deterministic)
            if (u, v) < (v, u):
                combined_weight = weight_uv
                combined_time = time_uv
            else:
                combined_weight = weight_vu
                combined_time = time_vu
                
        elif method == 'stronger':
            if abs(weight_uv) > abs(weight_vu):
                combined_weight = weight_uv
                combined_time = time_uv
            elif abs(weight_vu) > abs(weight_uv):
                combined_weight = weight_vu
                combined_time = time_vu
            else:
                # If equal strength, average them
                combined_weight = (weight_uv + weight_vu) / 2
                combined_time = (time_uv + time_vu) / 2
                
        else:
            raise ValueError(f"Unknown bidirectional handling method: {method}")
        
        # Add the combined edge
        G_undirected.add_edge(u, v, weight=combined_weight, time=combined_time)
    
    # Add unidirectional edges as-is
    for u, v, data in unidirectional_edges:
        G_undirected.add_edge(u, v, **data)
    
    # Calculate final statistics
    stats['final_edges'] = G_undirected.number_of_edges()
    stats['sign_consistency_preserved'] = sign_consistency_count
    
    print(f"Conversion complete: {stats['original_edges']} → {stats['final_edges']} edges")
    print(f"  - {stats['bidirectional_pairs']} bidirectional pairs combined")
    print(f"  - {stats['unidirectional_edges']} unidirectional edges preserved")
    print(f"  - {sign_consistency_count}/{stats['bidirectional_pairs']} bidirectional pairs had consistent signs")
    
    return G_undirected, stats

def to_undirected(G, method='average', use_weighted_features=True, preserve_original_weights=True):
    """
    Convert directed graph to undirected while properly handling bidirectional edges.
    This replaces the simple to_undirected() function with bidirectional edge handling.
    Addresses the TODO comment about handling asymmetric relationships.
    
    Parameters:
    G: NetworkX directed graph
    method: Method for handling bidirectional edges
    
    Returns:
    G_undirected: Undirected graph with bidirectional edges properly handled
    """
    if not G.is_directed():
        return G  # Already undirected
    
    # Use the weighted-aware bidirectional handling
    if use_weighted_features:
        G_undirected, stats = handle_bidirectional_edges_weighted(G, method=method, preserve_weights=preserve_original_weights)
    else:
        G_undirected, stats = handle_bidirectional_edges_binary(G, method=method)
    
    # Print impact analysis
    original_pos = sum(1 for _, _, d in G.edges(data=True) if d.get('weight', 0) > 0)
    original_neg = sum(1 for _, _, d in G.edges(data=True) if d.get('weight', 0) < 0)
    converted_pos = sum(1 for _, _, d in G_undirected.edges(data=True) if d.get('weight', 0) > 0)
    converted_neg = sum(1 for _, _, d in G_undirected.edges(data=True) if d.get('weight', 0) < 0)
    
    print(f"\nBidirectional conversion impact:")
    print(f"  Edge count change: {stats['final_edges'] - stats['original_edges']} ({stats['final_edges']/stats['original_edges']:.3f}x)")
    print(f"  Positive edges: {original_pos} → {converted_pos}")
    print(f"  Negative edges: {original_neg} → {converted_neg}")  
    
    return G_undirected