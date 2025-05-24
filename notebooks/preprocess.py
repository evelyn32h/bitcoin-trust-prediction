import os
import sys
import argparse
import pandas as pd
import yaml
import networkx as nx
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set project root as working directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(PROJECT_ROOT)

# Add project root to sys.path for src imports
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
from src.data_loader import load_bitcoin_data, save_graph_to_csv, save_config
from src.preprocessing import (label_edges, map_to_unweighted_graph, ensure_connectivity, to_undirected, reindex_nodes, 
                              edge_bfs_holdout_split, sample_random_seed_edges, to_undirected,
                              transform_weights, handle_bidirectional_edges_weighted)

# Load config from YAML
with open(os.path.join(PROJECT_ROOT, 'config.yaml'), 'r') as f:
    config = yaml.safe_load(f)

# Make it clear these values come from config
N_FOLDS = config['default_n_folds']
DEFAULT_EXPERIMENT_NAME = config['default_experiment_name']
DATA_PATH = config['data_path']
N_TEST_EDGES = config['num_test_edges']
N_VALIDATION_EDGES = config['num_validation_edges']

def create_splits(G, n_folds, n_test_edges, n_validation_edges):
    """
    Splits the graph into a single test split and multiple train/validation splits in parallel.
    Returns: (G_test, [(G_train, G_val), ...])
    """
    # Extract a single test split first
    test_seed_edge = sample_random_seed_edges(G, n=1, random_state=123)[0]
    G_trainval, G_test = edge_bfs_holdout_split(G, test_seed_edge, n_test_edges)

    # Now split the remaining graph into training/validation splits in parallel
    seed_edges = sample_random_seed_edges(G_trainval, n=n_folds, random_state=42)
    splits = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(edge_bfs_holdout_split, G_trainval, seed_edge, n_validation_edges) for seed_edge in seed_edges]
        for future in as_completed(futures):
            G_train, G_val = future.result()
            splits.append((G_train, G_val))
    return G_test, splits

def save_splits(G_test, splits, out_dir):
    """
    Saves the test split and train/validation splits to disk using save_graph_to_csv.
    Also saves the configuration variables used to a config_used.yaml file in the same directory.
    """
    os.makedirs(out_dir, exist_ok=True)
    test_path = os.path.join(out_dir, 'test.csv')
    save_graph_to_csv(G_test, test_path)
    print(f"Saved test split to {test_path}")
    for i, (G_train, G_val) in enumerate(splits):
        train_path = os.path.join(out_dir, f'fold_{i}_train.csv')
        val_path = os.path.join(out_dir, f'fold_{i}_val.csv')
        save_graph_to_csv(G_train, train_path)
        save_graph_to_csv(G_val, val_path)
    

def preprocess_graph(G, bidirectional_method='max', use_weighted_features=False, weight_method='sign', 
                     weight_bins=5, preserve_original_weights=True):
    """
    Applies all preprocessing steps to the input graph and returns the processed graph.
    Now supports both binary (original) and weighted features.
    
    Parameters:
    G: Input directed graph
    bidirectional_method: Method for handling bidirectional edges
    use_weighted_features: Whether to use weighted features 
    weight_method: How to process weights ('sign', 'raw', 'binned')
    weight_bins: Number of bins if using binned method
    preserve_original_weights: Whether to preserve original weight information
    
    Returns:
    G_processed: Processed undirected graph
    """
    print("Preprocessing graph...")
    print(f"Original graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Task #1 - Use weighted features: {use_weighted_features}")
    print(f"Weight method: {weight_method}")
    
    # Step 1: Process weights according to method
    G = transform_weights(G, use_weighted_features=use_weighted_features, weight_method=weight_method, weight_bins=weight_bins)
    print(f"After weight processing: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
    # Step 2: Ensure connectivity (keep largest weakly connected component)
    G = ensure_connectivity(G)
    print(f"After connectivity filtering: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Step 3: Handle bidirectional edges and convert to undirected
    G = to_undirected(G, method=bidirectional_method, use_weighted_features=use_weighted_features,
                     preserve_original_weights=preserve_original_weights)
        
    print(f"After bidirectional handling: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Step 4: Add labels (sign) to edges
    G = label_edges(G)
    
    # Step 5: Reindex nodes to be sequential
    G = reindex_nodes(G)
    print(f"Final graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    return G

def main():
    parser = argparse.ArgumentParser(description="Preprocess and split the graph, saving train/test splits.")
    parser.add_argument('--name', type=str, default=DEFAULT_EXPERIMENT_NAME, 
                       help="Name for the results directory (from config.yaml)")
    parser.add_argument('--bidirectional_method', type=str, default=None,
                       help="Method for handling bidirectional edges (overrides config)")
    parser.add_argument('--use_weighted_features', action='store_true',
                       help="Enable weighted features (Task #1)")
    parser.add_argument('--weight_method', type=str, default=None,
                       help="Weight processing method: sign, raw, binned")
    args = parser.parse_args()

    # Load parameters from config or command line
    bidirectional_method = args.bidirectional_method or config.get('bidirectional_method', 'max')
    use_weighted_features = args.use_weighted_features or config.get('use_weighted_features', False)
    weight_method = args.weight_method or config.get('weight_method', 'sign')
    weight_bins = config.get('weight_bins', 5)
    preserve_original_weights = config.get('preserve_original_weights', True)
    
    print(f"Task #1 Configuration:")
    print(f"  Using bidirectional edge handling method: {bidirectional_method}")
    print(f"  Using weighted features: {use_weighted_features}")
    print(f"  Weight method: {weight_method}")
    print(f"  Experiment name: {args.name}")

    # Load data
    G, df = load_bitcoin_data(DATA_PATH)

    # Preprocess with Task #1 weighted features support
    G = preprocess_graph(G, bidirectional_method=bidirectional_method,
                        use_weighted_features=use_weighted_features,
                        weight_method=weight_method,
                        weight_bins=weight_bins,
                        preserve_original_weights=preserve_original_weights)

    # Create splits
    G_test, splits = create_splits(G, N_FOLDS, N_TEST_EDGES, N_VALIDATION_EDGES)
    
    # Save splits
    out_dir = os.path.join('results', args.name, 'preprocess')
    save_splits(G_test, splits, out_dir)
    
    # Save config variables used using save_config
    save_config({
        'n_folds': N_FOLDS,
        'num_test_edges': N_TEST_EDGES,
        'num_validation_edges': N_VALIDATION_EDGES,
        'data_path': DATA_PATH,
        'experiment_name': args.name,
        'bidirectional_method': bidirectional_method,
        'use_weighted_features': use_weighted_features,
        'weight_method': weight_method,
        'weight_bins': weight_bins,
        'preserve_original_weights': preserve_original_weights
    }, out_dir)

if __name__ == "__main__":
    main()