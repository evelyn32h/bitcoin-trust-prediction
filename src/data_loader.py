import json
import networkx as nx
import pandas as pd
import numpy as np
import os
import joblib
import random
import yaml
from collections import defaultdict

def save_metrics_to_json(metrics, save_path):
    """
    Save metrics dictionary to JSON file
    
    Parameters:
    metrics: Dictionary containing metrics
    save_path: Path to save the JSON file
    """
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    metrics_serializable = convert_numpy_types(metrics)
    
    with open(save_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    
    print(f"Metrics saved to {save_path}")

def load_bitcoin_data(filepath, enable_subset_sampling=False, subset_config=None):
    """
    Load dataset (Bitcoin OTC or Epinions) and convert to NetworkX directed weighted graph
    Auto-detects format based on file extension or content
    Enhanced with BFS subset sampling support following project strategy
    
    Parameters:
    filepath: Path to the data file
    enable_subset_sampling: Whether to apply subset sampling
    subset_config: Dictionary with subset sampling configuration
    
    Returns:
    G: NetworkX directed graph
    df: DataFrame of the original data
    """
    print(f"Loading data from {filepath}...")
    
    # Auto-detect dataset format
    if 'epinions' in filepath.lower() or filepath.endswith('.txt'):
        # Epinions format: tab-separated, may have comments
        print("Detected Epinions format (tab-separated)")
        df = pd.read_csv(filepath, sep='\t', header=None, 
                        names=['source', 'target', 'rating'],
                        comment='#')  # Skip lines starting with #
        # Add time column for compatibility (use row index as proxy for time)
        df['time'] = range(len(df))
    else:
        # Bitcoin OTC format: comma-separated
        print("Detected Bitcoin OTC format (comma-separated)")
        df = pd.read_csv(filepath, sep=',', header=None, 
                        names=['source', 'target', 'rating', 'time'])
    
    print(f"Data loaded successfully with {len(df)} edges.")
    
    # Apply subset sampling if enabled and dataset is large
    if enable_subset_sampling and subset_config and len(df) > 50000:
        print("Large dataset detected. Applying subset sampling...")
        df = apply_subset_sampling(df, subset_config)
        print(f"Subset sampling complete. New size: {len(df)} edges.")
    elif enable_subset_sampling and len(df) <= 50000:
        print("Dataset is already small, skipping subset sampling.")
    
    # Basic statistics
    print(f"Dataset statistics:")
    print(f"  Nodes: {len(set(df['source']) | set(df['target']))} unique")
    print(f"  Edges: {len(df)}")
    
    # Edge sign distribution
    pos_edges = len(df[df['rating'] > 0])
    neg_edges = len(df[df['rating'] < 0])
    zero_edges = len(df[df['rating'] == 0])
    print(f"  Positive edges: {pos_edges} ({pos_edges/len(df)*100:.1f}%)")
    print(f"  Negative edges: {neg_edges} ({neg_edges/len(df)*100:.1f}%)")
    if zero_edges > 0:
        print(f"  Zero edges: {zero_edges} ({zero_edges/len(df)*100:.1f}%)")
    
    # Create directed weighted graph
    G = nx.DiGraph()
    
    # Add edges and weights
    for _, row in df.iterrows():
        G.add_edge(row['source'], row['target'], 
                   weight=row['rating'], 
                   time=row['time'])
    
    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G, df

def apply_subset_sampling(df, subset_config):
    """
    Apply subset sampling to large datasets to make them comparable in size.
    Enhanced with BFS sampling method following project strategy.
    
    Parameters:
    df: Original DataFrame
    subset_config: Dictionary with sampling configuration
    
    Returns:
    df_subset: Sampled subset DataFrame
    """
    sampling_method = subset_config.get('subset_sampling_method', 'time_based')
    target_edge_count = subset_config.get('target_edge_count', 40000)
    preserve_structure = subset_config.get('subset_preserve_structure', True)
    
    print(f"Applying {sampling_method} sampling...")
    print(f"Target: ~{target_edge_count} edges (comparable to Bitcoin OTC)")
    
    if sampling_method == 'bfs_sampling':
        # NEW: BFS sampling following project strategy
        df_subset = sample_by_bfs_method(df, target_edge_count, subset_config)
        print(f"BFS sampling: {len(df_subset)} edges")
        
    elif sampling_method == 'community_detection':
        # Community detection sampling - best for preserving network structure
        df_subset = sample_by_community_detection(df, target_edge_count)
        print(f"Community detection sampling: {len(df_subset)} edges")
        
    elif sampling_method == 'embeddedness_based':
        # Embeddedness-based sampling to preserve high-embeddedness edges
        df_subset = sample_by_embeddedness_priority(df, target_edge_count)
        print(f"Embeddedness-priority sampling: {len(df_subset)} edges")
        
    elif sampling_method == 'embeddedness_stratified':
        # Stratified sampling to preserve embeddedness distribution
        df_subset = sample_by_embeddedness_distribution(df, target_edge_count)
        print(f"Embeddedness-distribution sampling: {len(df_subset)} edges")
        
    elif sampling_method == 'time_based':
        # Original time-based sampling
        df_sorted = df.sort_values('time', ascending=False)
        df_subset = df_sorted.head(target_edge_count).copy()
        print(f"Selected most recent {len(df_subset)} edges")
        
    elif sampling_method == 'random':
        # Random sampling with seed for reproducibility
        random.seed(42)
        df_subset = df.sample(n=min(target_edge_count, len(df)), random_state=42).copy()
        print(f"Random sampling of {len(df_subset)} edges")
        
    elif sampling_method == 'high_degree':
        # Sample subgraph around high-degree nodes to preserve structure
        df_subset = sample_high_degree_subgraph(df, target_edge_count)
        print(f"High-degree subgraph sampling: {len(df_subset)} edges")
        
    else:
        print(f"Unknown sampling method: {sampling_method}, using time_based")
        df_sorted = df.sort_values('time', ascending=False)
        df_subset = df_sorted.head(target_edge_count).copy()
    
    # Reset time column to be sequential for consistency
    df_subset = df_subset.copy()
    df_subset['time'] = range(len(df_subset))
    
    # Report sampling results
    original_nodes = len(set(df['source']) | set(df['target']))
    subset_nodes = len(set(df_subset['source']) | set(df_subset['target']))
    
    print(f"Sampling completed:")
    print(f"  Method: {sampling_method}")
    print(f"  Original: {len(df)} edges, {original_nodes} nodes")
    print(f"  Subset: {len(df_subset)} edges, {subset_nodes} nodes")
    print(f"  Edge compression: {len(df_subset)/len(df):.3f}")
    print(f"  Node compression: {subset_nodes/original_nodes:.3f}")
    
    return df_subset

def sample_by_bfs_method(df, target_edge_count, subset_config):
    """
    Apply BFS sampling following project strategy.
    Same method as used for Bitcoin OTC but optimized for large datasets.
    
    Parameters:
    df: Original DataFrame
    target_edge_count: Target number of edges
    subset_config: Configuration dictionary with BFS settings
    
    Returns:
    df_subset: Subset DataFrame from BFS sampling
    """
    from src.preprocessing import optimized_bfs_sampling
    
    print("Applying BFS sampling (same strategy as Bitcoin OTC)...")
    
    # Create graph from DataFrame
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['source'], row['target'], 
                   weight=row['rating'], time=row['time'])
    
    print(f"Created graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Get BFS configuration
    seed_selection = subset_config.get('bfs_seed_selection', 'random_moderate_degree')
    degree_percentile = subset_config.get('bfs_degree_percentile', 70)
    
    # Apply BFS sampling
    G_subset = optimized_bfs_sampling(
        G, 
        target_edge_count, 
        seed_selection=seed_selection,
        degree_percentile=degree_percentile
    )
    
    # Convert back to DataFrame
    subset_edges = []
    for u, v, data in G_subset.edges(data=True):
        subset_edges.append({
            'source': u,
            'target': v, 
            'rating': data['weight'],
            'time': data['time']
        })
    
    df_subset = pd.DataFrame(subset_edges)
    
    print(f"BFS sampling completed:")
    print(f"  Nodes: {G_subset.number_of_nodes()}")
    print(f"  Edges: {len(df_subset)}")
    print(f"  Target achievement: {len(df_subset)/target_edge_count:.1%}")
    
    return df_subset

def sample_by_community_detection(df, target_edge_count):
    """
    Sample edges using community detection to preserve natural network structure.
    This is the most scientifically sound approach for maintaining network properties.
    
    Parameters:
    df: Original DataFrame
    target_edge_count: Target number of edges
    
    Returns:
    df_subset: Subset DataFrame preserving community structure
    """
    try:
        import community as community_louvain  # python-louvain
    except ImportError:
        print("Error: python-louvain not installed. Install with: pip install python-louvain")
        print("Falling back to time-based sampling...")
        return df.sort_values('time', ascending=False).head(target_edge_count).copy()
    
    print("Computing community structure using Louvain algorithm...")
    
    # Build undirected graph for community detection (communities are undirected)
    G_undirected = nx.Graph()
    for _, row in df.iterrows():
        # Use absolute weight for community detection (structure matters, not sign)
        G_undirected.add_edge(row['source'], row['target'], weight=abs(row['rating']))
    
    print(f"  Graph built: {G_undirected.number_of_nodes()} nodes, {G_undirected.number_of_edges()} edges")
    
    # Run Louvain community detection
    print("  Running Louvain community detection...")
    partition = community_louvain.best_partition(G_undirected, weight='weight', resolution=1.0, random_state=42)
    
    # Analyze community structure
    communities = defaultdict(list)
    for node, comm_id in partition.items():
        communities[comm_id].append(node)
    
    print(f"  Found {len(communities)} communities")
    
    # Sort communities by size (largest first)
    sorted_communities = sorted(communities.items(), 
                               key=lambda x: len(x[1]), 
                               reverse=True)
    
    # Display top communities
    print("  Top communities by size:")
    for i, (comm_id, nodes) in enumerate(sorted_communities[:5]):
        # Count edges within this community
        comm_edges = 0
        for _, row in df.iterrows():
            if row['source'] in nodes and row['target'] in nodes:
                comm_edges += 1
        print(f"    Community {comm_id}: {len(nodes)} nodes, ~{comm_edges} internal edges")
    
    # Select communities to reach target edge count
    selected_nodes = set()
    selected_edges_count = 0
    
    print("  Selecting communities to reach target edge count:")
    
    for comm_id, nodes in sorted_communities:
        # Calculate edges if we add this community
        candidate_nodes = selected_nodes.union(set(nodes))
        
        # Count edges in the candidate subgraph
        candidate_edges = []
        for _, row in df.iterrows():
            if row['source'] in candidate_nodes and row['target'] in candidate_nodes:
                candidate_edges.append(row)
        
        candidate_edge_count = len(candidate_edges)
        
        # Decision: add this community if it doesn't exceed target by too much
        if candidate_edge_count <= target_edge_count * 1.15:  # Allow 15% overshoot
            selected_nodes = candidate_nodes
            selected_edges_count = candidate_edge_count
            print(f"    Added community {comm_id}: {len(nodes)} nodes")
            print(f"      Total: {len(selected_nodes)} nodes, {selected_edges_count} edges")
            
            # Stop if we're close to target
            if selected_edges_count >= target_edge_count * 0.9:  # At least 90% of target
                break
        else:
            # If adding whole community overshoots, try partial addition
            if selected_edges_count < target_edge_count * 0.8:  # If we're still far from target
                remaining_target = target_edge_count - selected_edges_count
                
                # Add highest-degree nodes from this community
                node_degrees = [(node, G_undirected.degree(node)) for node in nodes]
                node_degrees.sort(key=lambda x: x[1], reverse=True)
                
                # Add nodes incrementally until we approach target
                added_from_community = 0
                for node, degree in node_degrees:
                    test_nodes = selected_nodes.union({node})
                    test_edges = [row for _, row in df.iterrows() 
                                 if row['source'] in test_nodes and row['target'] in test_nodes]
                    
                    if len(test_edges) <= target_edge_count:
                        selected_nodes.add(node)
                        selected_edges_count = len([row for _, row in df.iterrows() 
                                                   if row['source'] in selected_nodes and row['target'] in selected_nodes])
                        added_from_community += 1
                    else:
                        break
                
                if added_from_community > 0:
                    print(f"    Partially added community {comm_id}: {added_from_community}/{len(nodes)} nodes")
                    print(f"      Total: {len(selected_nodes)} nodes, {selected_edges_count} edges")
            break
    
    # Extract final subset
    final_edges = []
    for _, row in df.iterrows():
        if row['source'] in selected_nodes and row['target'] in selected_nodes:
            final_edges.append(row)
    
    df_subset = pd.DataFrame(final_edges)
    
    # Analyze network structure preservation
    if len(df_subset) > 0:
        # Calculate clustering coefficient preservation
        try:
            original_clustering = nx.average_clustering(G_undirected)
            subset_graph = nx.Graph()
            for _, row in df_subset.iterrows():
                subset_graph.add_edge(row['source'], row['target'])
            subset_clustering = nx.average_clustering(subset_graph)
            
            print(f"  Network structure preservation:")
            print(f"    Original avg clustering: {original_clustering:.4f}")
            print(f"    Subset avg clustering: {subset_clustering:.4f}")
            print(f"    Clustering preservation: {subset_clustering/original_clustering:.3f}")
        except:
            print(f"  Could not calculate clustering coefficients")
        
        # Calculate final community distribution in subset
        subset_partition = {}
        for node in selected_nodes:
            if node in partition:
                subset_partition[node] = partition[node]
        
        subset_communities = defaultdict(int)
        for comm_id in subset_partition.values():
            subset_communities[comm_id] += 1
        
        print(f"  Final subset contains nodes from {len(subset_communities)} communities")
    
    print(f"Community detection sampling completed:")
    print(f"  Selected: {len(selected_nodes)} nodes")
    print(f"  Final edges: {len(df_subset)}")
    print(f"  Target was: {target_edge_count}")
    print(f"  Achievement: {len(df_subset)/target_edge_count:.1%} of target")
    
    return df_subset

def sample_by_embeddedness_priority(df, target_edge_count):
    """
    Sample edges prioritizing high embeddedness to preserve HOC feature quality.
    
    Parameters:
    df: Original DataFrame
    target_edge_count: Target number of edges
    
    Returns:
    df_subset: Subset DataFrame with high-embeddedness edges prioritized
    """
    print("Computing embeddedness for all edges...")
    
    # Build temporary graph to calculate embeddedness
    G_temp = nx.Graph()  # Use undirected for embeddedness calculation
    for _, row in df.iterrows():
        G_temp.add_edge(row['source'], row['target'])
    
    print(f"  Graph built: {G_temp.number_of_nodes()} nodes, {G_temp.number_of_edges()} edges")
    
    # Calculate embeddedness for each edge
    edge_embeddedness = []
    
    for idx, row in df.iterrows():
        u, v = row['source'], row['target']
        
        if G_temp.has_edge(u, v):
            # Get neighbors
            u_neighbors = set(G_temp.neighbors(u))
            v_neighbors = set(G_temp.neighbors(v))
            
            # Calculate common neighbors (embeddedness)
            common_neighbors = u_neighbors & v_neighbors
            embeddedness = len(common_neighbors)
        else:
            embeddedness = 0
        
        edge_embeddedness.append((idx, embeddedness, row))
        
        if len(edge_embeddedness) % 50000 == 0:
            print(f"  Progress: {len(edge_embeddedness)}/{len(df)} edges processed")
    
    print(f"  Embeddedness calculation complete")
    
    # Sort by embeddedness (highest first) and select top edges
    edge_embeddedness.sort(key=lambda x: x[1], reverse=True)
    selected_edges = edge_embeddedness[:target_edge_count]
    
    # Create subset DataFrame
    selected_rows = [edge[2] for edge in selected_edges]
    df_subset = pd.DataFrame(selected_rows)
    
    # Report embeddedness statistics
    embeddedness_values = [edge[1] for edge in selected_edges]
    avg_embeddedness = np.mean(embeddedness_values)
    zero_embed_ratio = sum(1 for e in embeddedness_values if e == 0) / len(embeddedness_values)
    high_embed_ratio = sum(1 for e in embeddedness_values if e >= 3) / len(embeddedness_values)
    
    print(f"  Selected edges embeddedness:")
    print(f"    Average embeddedness: {avg_embeddedness:.2f}")
    print(f"    Zero embeddedness: {zero_embed_ratio:.1%}")
    print(f"    High embeddedness (≥3): {high_embed_ratio:.1%}")
    
    return df_subset

def sample_by_embeddedness_distribution(df, target_edge_count):
    """
    Sample edges preserving the original embeddedness distribution of full Epinions.
    This maintains the same proportion of high/low embeddedness edges as the complete dataset.
    
    Parameters:
    df: Original DataFrame
    target_edge_count: Target number of edges
    
    Returns:
    df_subset: Subset DataFrame preserving embeddedness distribution
    """
    print("Computing embeddedness distribution for stratified sampling...")
    
    # Build temporary graph to calculate embeddedness
    G_temp = nx.Graph()  # Use undirected for embeddedness calculation
    for _, row in df.iterrows():
        G_temp.add_edge(row['source'], row['target'])
    
    print(f"  Graph built: {G_temp.number_of_nodes()} nodes, {G_temp.number_of_edges()} edges")
    
    # Group edges by embeddedness level
    embeddedness_groups = defaultdict(list)
    
    for idx, row in df.iterrows():
        u, v = row['source'], row['target']
        
        if G_temp.has_edge(u, v):
            # Get neighbors
            u_neighbors = set(G_temp.neighbors(u))
            v_neighbors = set(G_temp.neighbors(v))
            
            # Calculate common neighbors (embeddedness)
            common_neighbors = u_neighbors & v_neighbors
            embeddedness = len(common_neighbors)
        else:
            embeddedness = 0
        
        # Group by embeddedness level
        if embeddedness == 0:
            group = 'embed_0'
        elif embeddedness == 1:
            group = 'embed_1'
        elif embeddedness == 2:
            group = 'embed_2'
        elif embeddedness <= 5:
            group = 'embed_3-5'
        elif embeddedness <= 10:
            group = 'embed_6-10'
        else:
            group = 'embed_10+'
            
        embeddedness_groups[group].append((idx, embeddedness, row))
        
        if (idx + 1) % 50000 == 0:
            print(f"  Progress: {idx + 1}/{len(df)} edges processed")
    
    print(f"  Embeddedness grouping complete")
    
    # Report original distribution
    total_edges = len(df)
    print(f"  Original embeddedness distribution:")
    for group, edges in embeddedness_groups.items():
        ratio = len(edges) / total_edges
        print(f"    {group}: {len(edges)} edges ({ratio:.1%})")
    
    # Sample from each group proportionally
    selected_edges = []
    random.seed(42)  # For reproducibility
    
    for group, edges in embeddedness_groups.items():
        if not edges:
            continue
            
        # Calculate how many edges to sample from this group
        original_ratio = len(edges) / total_edges
        target_count = int(target_edge_count * original_ratio)
        
        # Ensure we don't sample more than available
        sample_count = min(target_count, len(edges))
        
        if sample_count > 0:
            # Random sample from this embeddedness group
            sampled = random.sample(edges, sample_count)
            selected_edges.extend(sampled)
            
            print(f"    Sampling {group}: {sample_count}/{len(edges)} edges")
    
    # If we haven't reached target, fill with remaining high-embeddedness edges
    if len(selected_edges) < target_edge_count:
        remaining_needed = target_edge_count - len(selected_edges)
        
        # Get high-embeddedness edges not yet selected
        selected_indices = {edge[0] for edge in selected_edges}
        high_embed_edges = []
        
        for group in ['embed_10+', 'embed_6-10', 'embed_3-5']:
            if group in embeddedness_groups:
                for edge in embeddedness_groups[group]:
                    if edge[0] not in selected_indices:
                        high_embed_edges.append(edge)
        
        if high_embed_edges:
            additional = random.sample(high_embed_edges, 
                                     min(remaining_needed, len(high_embed_edges)))
            selected_edges.extend(additional)
            print(f"    Added {len(additional)} additional high-embeddedness edges")
    
    # Create subset DataFrame
    selected_rows = [edge[2] for edge in selected_edges[:target_edge_count]]
    df_subset = pd.DataFrame(selected_rows)
    
    # Report final embeddedness statistics
    final_embeddedness = [edge[1] for edge in selected_edges[:target_edge_count]]
    avg_embeddedness = np.mean(final_embeddedness)
    zero_embed_ratio = sum(1 for e in final_embeddedness if e == 0) / len(final_embeddedness)
    high_embed_ratio = sum(1 for e in final_embeddedness if e >= 3) / len(final_embeddedness)
    
    print(f"  Final subset embeddedness:")
    print(f"    Average embeddedness: {avg_embeddedness:.2f}")
    print(f"    Zero embeddedness: {zero_embed_ratio:.1%}")
    print(f"    High embeddedness (≥3): {high_embed_ratio:.1%}")
    
    return df_subset

def sample_high_degree_subgraph(df, target_edge_count):
    """
    Sample a subgraph around high-degree nodes to preserve network structure.
    
    Parameters:
    df: Original DataFrame
    target_edge_count: Target number of edges
    
    Returns:
    df_subset: Subset DataFrame
    """
    # Build temporary graph to calculate degrees
    G_temp = nx.DiGraph()
    for _, row in df.iterrows():
        G_temp.add_edge(row['source'], row['target'])
    
    # Calculate degree centrality (total degree)
    degree_dict = dict(G_temp.degree())
    
    # Select high-degree nodes (top 30% by degree)
    sorted_nodes = sorted(degree_dict.keys(), key=lambda x: degree_dict[x], reverse=True)
    num_seed_nodes = max(100, len(sorted_nodes) // 10)  # At least 100 or 10% of nodes
    seed_nodes = set(sorted_nodes[:num_seed_nodes])
    
    # Expand to include neighbors to maintain connectivity
    expanded_nodes = set(seed_nodes)
    for node in seed_nodes:
        # Add immediate neighbors
        neighbors = set(G_temp.neighbors(node)) | set(G_temp.predecessors(node))
        expanded_nodes.update(list(neighbors)[:5])  # Add up to 5 neighbors per high-degree node
    
    # Filter edges to only include selected nodes
    df_subset = df[
        (df['source'].isin(expanded_nodes)) & 
        (df['target'].isin(expanded_nodes))
    ].copy()
    
    # If still too many edges, take the most recent ones
    if len(df_subset) > target_edge_count:
        df_subset = df_subset.sort_values('time', ascending=False).head(target_edge_count)
    
    return df_subset

def load_undirected_graph_from_csv(filepath):
    """
    Load an undirected NetworkX graph from a CSV file with columns: source, target, label, weight, time.
    Returns:
        G: NetworkX undirected graph
        df: DataFrame of the original data
    """
    # Try to read with expected format first
    try:
        df = pd.read_csv(filepath, header=None, names=['source', 'target', 'label', 'weight', 'time'])
    except Exception as e:
        print(f"Warning: Could not read with expected format, trying alternative: {e}")
        # Try reading with headers
        df = pd.read_csv(filepath)
        # Ensure required columns exist
        required_cols = ['source', 'target']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in CSV")
    
    # Create undirected graph
    G = nx.Graph()
    
    for _, row in df.iterrows():
        # Get edge attributes
        edge_attrs = {}
        if 'weight' in row and not pd.isna(row['weight']):
            edge_attrs['weight'] = row['weight']
        if 'label' in row and not pd.isna(row['label']):
            edge_attrs['label'] = row['label']
        if 'time' in row and not pd.isna(row['time']):
            edge_attrs['time'] = row['time']
        
        G.add_edge(row['source'], row['target'], **edge_attrs)
    
    print(f"Loaded graph from {filepath}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G, df

def save_graph_to_csv(G, filepath):
    """
    Save a NetworkX graph to a CSV file with columns: source, target, label, weight, time.
    The order and field names match the expected format.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert graph to edge list
    edge_data = []
    for u, v, data in G.edges(data=True):
        edge_record = {
            'source': u,
            'target': v,
            'label': data.get('label', 0),
            'weight': data.get('weight', 1),
            'time': data.get('time', 0)
        }
        edge_data.append(edge_record)
    
    # Create DataFrame and save
    df = pd.DataFrame(edge_data)
    df = df[['source', 'target', 'label', 'weight', 'time']]  # Ensure column order
    df.to_csv(filepath, index=False, header=False)
    print(f"Graph saved to {filepath}: {len(df)} edges")

def save_model(model, scaler, out_dir, fold):
    """
    Saves the model and scaler to out_dir for the given fold.
    """
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, f'model_fold_{fold}.joblib')
    scaler_path = os.path.join(out_dir, f'scaler_fold_{fold}.joblib')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"Saved model and scaler for fold {fold} to {out_dir}")
    print(f"  Model: {model_path}")
    print(f"  Scaler: {scaler_path}")
    
def load_models(training_dir, n_folds):
    """
    Loads models and scalers from training_dir, one per fold.
    Returns a list of (model, scaler) tuples.
    """
    models_and_scalers = []
    for i in range(n_folds):
        model_path = os.path.join(training_dir, f'model_fold_{i}.joblib')
        scaler_path = os.path.join(training_dir, f'scaler_fold_{i}.joblib')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            models_and_scalers.append((model, scaler))
        else:
            print(f"Warning: Model or scaler files not found for fold {i}")
            print(f"  Expected: {model_path}")
            print(f"  Expected: {scaler_path}")
            models_and_scalers.append((None, None))
    return models_and_scalers
    
def save_prediction_results(true_labels, predicted_labels, predicted_probabilities, out_dir):
    """
    Save prediction results (true labels, predicted labels, predicted probabilities) to a single CSV file in the specified directory.
    Each column represents one parameter.
    """
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame({
        "true_label": true_labels,
        "predicted_label": predicted_labels,
        "predicted_probability": predicted_probabilities
    })
    filepath = os.path.join(out_dir, "prediction_results.csv")
    df.to_csv(filepath, index=False)
    print(f"Prediction results saved to {filepath}")

def load_prediction_results(out_dir):
    """
    Load prediction results from a single CSV file in the specified directory.
    Returns true_labels, predicted_labels, predicted_probabilities as numpy arrays.
    """
    file_path = os.path.join(out_dir, "prediction_results.csv")
    df = pd.read_csv(file_path)
    true_labels = df["true_label"].to_numpy()
    predicted_labels = df["predicted_label"].to_numpy()
    predicted_probabilities = df["predicted_probability"].to_numpy()
    return true_labels, predicted_labels, predicted_probabilities

def save_metrics(metrics, out_dir):
    """
    Save a (possibly nested) metrics dictionary as a JSON file at the given path.
    """
    os.makedirs(out_dir, exist_ok=True)
    file_path = os.path.join(out_dir, "metrics.json")
    save_metrics_to_json(metrics, file_path)
        
def load_metrics(metrics_path):
    """
    Load metrics from JSON file.
    """
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    return metrics

def save_config(config_dict, out_dir, filename="config_used.yaml"):
    """
    Save a configuration dictionary as a YAML file in the specified directory.
    """
    os.makedirs(out_dir, exist_ok=True)
    config_path = os.path.join(out_dir, filename)
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    print(f"Config saved to {config_path}")

if __name__ == "__main__":
    # Test BFS sampling
    import sys
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        # Default test paths
        bitcoin_path = os.path.join('..', 'data', 'soc-sign-bitcoinotc.csv')
        epinions_path = os.path.join('..', 'data', 'soc-sign-epinions.txt')
        
        # Test whichever exists
        if os.path.exists(epinions_path):
            data_path = epinions_path
        else:
            data_path = bitcoin_path
    
    # Test with BFS sampling (following project strategy)
    subset_config = {
        'subset_sampling_method': 'bfs_sampling',
        'target_edge_count': 35000,
        'subset_preserve_structure': True,
        'bfs_seed_selection': 'random_moderate_degree',
        'bfs_degree_percentile': 70
    }
    
    G, df = load_bitcoin_data(data_path, enable_subset_sampling=True, subset_config=subset_config)
    print("Data sample:")
    print(df.head())