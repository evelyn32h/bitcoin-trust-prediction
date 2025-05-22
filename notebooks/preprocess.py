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
from src.preprocessing import map_to_unweighted_graph, ensure_connectivity, to_undirected, reindex_nodes, edge_bfs_holdout_split, sample_random_seed_edges

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
    

def preprocess_graph(G):
    """
    Applies all preprocessing steps to the input graph and returns the processed graph.
    """
    print("Preprocessing graph...")
    G = map_to_unweighted_graph(G)
    G = ensure_connectivity(G)
    G = to_undirected(G)
    G = reindex_nodes(G)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def main():
    parser = argparse.ArgumentParser(description="Preprocess and split the graph, saving train/test splits.")
    parser.add_argument('--name', type=str, default=DEFAULT_EXPERIMENT_NAME, help="Name for the results directory (from config.yaml)")
    args = parser.parse_args()

    # Load data
    G, df = load_bitcoin_data(DATA_PATH)

    # Preprocess 
    G = preprocess_graph(G)

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
        'experiment_name': DEFAULT_EXPERIMENT_NAME
    }, out_dir)

if __name__ == "__main__":
    main()
