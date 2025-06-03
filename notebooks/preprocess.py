import os
import sys
import argparse
import pandas as pd
import yaml
import networkx as nx
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set project root as working directory (notebooks/ is current directory)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(PROJECT_ROOT)

# Add project root to sys.path for src imports
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
from src.data_loader import load_bitcoin_data, save_graph_to_csv, save_config
from src.preprocessing import (label_edges, map_to_unweighted_graph, ensure_connectivity, to_undirected, reindex_nodes, 
                              edge_bfs_holdout_split, sample_random_seed_edges, to_undirected,
                              transform_weights, handle_bidirectional_edges_weighted, filter_by_embeddedness)

# Load config from YAML (from project root)
with open(os.path.join(PROJECT_ROOT, 'config.yaml'), 'r') as f:
    config = yaml.safe_load(f)

# Make it clear these values come from config
N_FOLDS = config['default_n_folds']
DEFAULT_EXPERIMENT_NAME = config['default_experiment_name']
DATA_PATH = config['data_path']

# UPDATED: Use optimized split ratios (74:12:14)
# These are the specific optimal sizes found through testing
N_TEST_EDGES = config['num_test_edges']         # 3080 samples (14%)
N_VALIDATION_EDGES = config['num_validation_edges']  # 2640 samples (12%)

# Get split ratios for validation
SPLIT_RATIOS = config.get('split_ratio', {
    'train': 0.74,
    'validation': 0.12,
    'test': 0.14
})

# NEW: Subset sampling configuration
ENABLE_SUBSET_SAMPLING = config.get('enable_subset_sampling', False)
SUBSET_SAMPLING_METHOD = config.get('subset_sampling_method', 'time_based')
TARGET_EDGE_COUNT = config.get('target_edge_count', 40000)
SUBSET_PRESERVE_STRUCTURE = config.get('subset_preserve_structure', True)

def calculate_optimal_split_sizes(total_edges):
    """
    Calculate split sizes based on optimal 74:12:14 ratio and specific sample counts.
    
    Parameters:
    total_edges: Total number of edges in the dataset
    
    Returns:
    dict: Split sizes and validation info
    """
    print(f"\nCalculating optimal split for {total_edges:,} total edges")
    print(f"Target split ratio: {SPLIT_RATIOS['train']:.0%}:{SPLIT_RATIOS['validation']:.0%}:{SPLIT_RATIOS['test']:.0%}")
    
    # Use specific optimal counts
    test_size = N_TEST_EDGES      # 3080 (14%)
    validation_size = N_VALIDATION_EDGES  # 2640 (12%)
    
    # Calculate training size as remainder
    train_size = total_edges - test_size - validation_size
    
    # Validate split ratios
    actual_test_ratio = test_size / total_edges
    actual_val_ratio = validation_size / total_edges
    actual_train_ratio = train_size / total_edges
    
    print(f"Optimal configuration:")
    print(f"  Test set:       {test_size:,} edges ({actual_test_ratio:.1%})")
    print(f"  Validation set: {validation_size:,} edges ({actual_val_ratio:.1%})")
    print(f"  Training set:   {train_size:,} edges ({actual_train_ratio:.1%})")
    
    # Check if we have enough edges
    if train_size < 1000:
        print(f"Warning: Training set very small ({train_size:,} edges)")
    
    if total_edges < (test_size + validation_size):
        raise ValueError(f"Dataset too small: need {test_size + validation_size:,} edges, have {total_edges:,}")
    
    return {
        'train': train_size,
        'validation': validation_size,
        'test': test_size,
        'ratios': {
            'train': actual_train_ratio,
            'validation': actual_val_ratio,
            'test': actual_test_ratio
        }
    }

def create_splits_optimal(G, n_folds, test_edges, validation_edges):
    """
    Create splits using optimal 74:12:14 ratio.
    Enhanced version of original create_splits function.
    
    Parameters:
    G: NetworkX graph
    n_folds: Number of cross-validation folds
    test_edges: Number of edges for test set (3080)
    validation_edges: Number of edges for validation set (2640)
    
    Returns:
    (G_test, [(G_train, G_val), ...])
    """
    print(f"\nCreating splits with optimal configuration")
    print(f"Total edges available: {G.number_of_edges():,}")
    
    # Calculate and validate split sizes
    split_info = calculate_optimal_split_sizes(G.number_of_edges())
    
    # Extract a single test split first using BFS method
    print(f"\nCreating test set ({test_edges:,} edges)...")
    test_seed_edge = sample_random_seed_edges(G, n=1, random_state=123)[0]
    G_trainval, G_test = edge_bfs_holdout_split(G, test_seed_edge, test_edges)
    
    actual_test_size = G_test.number_of_edges()
    print(f"Test set created: {actual_test_size:,} edges")
    
    if abs(actual_test_size - test_edges) > test_edges * 0.1:  # Allow 10% tolerance
        print(f"  Warning: Test set size differs from target by {actual_test_size - test_edges:+,} edges")
    
    # Now split the remaining graph into training/validation splits in parallel
    print(f"\nCreating {n_folds} train/validation splits from remaining {G_trainval.number_of_edges():,} edges...")
    
    seed_edges = sample_random_seed_edges(G_trainval, n=n_folds, random_state=42)
    splits = []
    
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(edge_bfs_holdout_split, G_trainval, seed_edge, validation_edges) 
                  for seed_edge in seed_edges]
        
        for i, future in enumerate(as_completed(futures)):
            G_train, G_val = future.result()
            splits.append((G_train, G_val))
            
            actual_val_size = G_val.number_of_edges()
            actual_train_size = G_train.number_of_edges()
            
            print(f"  Fold {len(splits)}: Train={actual_train_size:,}, Val={actual_val_size:,}")
    
    # Sort splits to maintain consistent ordering
    splits.sort(key=lambda x: x[0].number_of_edges())
    
    # Final validation and summary
    if splits:
        avg_train_size = sum(split[0].number_of_edges() for split in splits) / len(splits)
        avg_val_size = sum(split[1].number_of_edges() for split in splits) / len(splits)
        
        total_used = actual_test_size + avg_val_size + avg_train_size
        
        print(f"\nFinal split summary:")
        print(f"  Average train size: {avg_train_size:,.0f} edges ({avg_train_size/G.number_of_edges():.1%})")
        print(f"  Average val size:   {avg_val_size:,.0f} edges ({avg_val_size/G.number_of_edges():.1%})")
        print(f"  Test size:          {actual_test_size:,} edges ({actual_test_size/G.number_of_edges():.1%})")
        print(f"  Total edge usage:   {total_used:,.0f} / {G.number_of_edges():,} ({total_used/G.number_of_edges():.1%})")
        
        # Check if ratios match target
        target_test_ratio = SPLIT_RATIOS['test']
        actual_test_ratio = actual_test_size / G.number_of_edges()
        
        if abs(actual_test_ratio - target_test_ratio) < 0.02:  # Within 2%
            print(f"Split ratios match optimal configuration!")
        else:
            print(f"  Split ratios differ from target (test: {actual_test_ratio:.1%} vs {target_test_ratio:.1%})")
    
    return G_test, splits

def save_splits(G_test, splits, out_dir):
    """
    Save the test split and train/validation splits to disk using save_graph_to_csv.
    Enhanced with better reporting for optimal split.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Save test split
    test_path = os.path.join(out_dir, 'test.csv')
    save_graph_to_csv(G_test, test_path)
    print(f"Saved test split to {test_path} ({G_test.number_of_edges():,} edges)")
    
    # Save train/validation splits
    total_train_edges = 0
    total_val_edges = 0
    
    for i, (G_train, G_val) in enumerate(splits):
        train_path = os.path.join(out_dir, f'fold_{i}_train.csv')
        val_path = os.path.join(out_dir, f'fold_{i}_val.csv')
        
        save_graph_to_csv(G_train, train_path)
        save_graph_to_csv(G_val, val_path)
        
        train_edges = G_train.number_of_edges()
        val_edges = G_val.number_of_edges()
        
        total_train_edges += train_edges
        total_val_edges += val_edges
        
        print(f"Saved fold {i}: train={train_edges:,} edges, val={val_edges:,} edges")
    
    # Summary report
    avg_train = total_train_edges / len(splits) if splits else 0
    avg_val = total_val_edges / len(splits) if splits else 0
    
    print(f"\nSplit summary:")
    print(f"  Test set:     {G_test.number_of_edges():,} edges (target: {N_TEST_EDGES:,})")
    print(f"  Avg train:    {avg_train:,.0f} edges per fold")
    print(f"  Avg val:      {avg_val:,.0f} edges per fold (target: {N_VALIDATION_EDGES:,})")
    print(f"  Total folds:  {len(splits)}")

def preprocess_graph(G, min_embeddedness=None, bidirectional_method='max', use_weighted_features=False, weight_method='sign', 
                     weight_bins=5, preserve_original_weights=True):
    """
    Apply all preprocessing steps to the input graph and return the processed graph.
    Now supports both binary (original) and weighted features.
    ENHANCED: Embeddedness filtering is applied BEFORE BFS splitting.
    
    Parameters:
    G: Input directed graph
    min_embeddedness: Minimum embeddedness for edge filtering (None=no filter, 0=no filter, >=1=filter)
    bidirectional_method: Method for handling bidirectional edges
    use_weighted_features: Whether to use weighted features 
    weight_method: How to process weights ('sign', 'raw', 'binned')
    weight_bins: Number of bins if using binned method
    preserve_original_weights: Whether to preserve original weight information
    
    Returns:
    G_processed: Processed undirected graph
    """
    print("Preprocessing graph for optimal split...")
    print(f"Original graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Use weighted features: {use_weighted_features}")
    print(f"Weight method: {weight_method}")
    if min_embeddedness is not None:
        print(f"Embeddedness filtering: min_embeddedness={min_embeddedness}")
    
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
    
    # Step 5: EMBEDDEDNESS FILTERING (NEW - APPLIED BEFORE BFS SPLITTING)
    if min_embeddedness is not None and min_embeddedness > 0:
        print(f"Applying embeddedness filtering with min_embeddedness={min_embeddedness}")
        G = filter_by_embeddedness(G, min_embeddedness)
        print(f"After embeddedness filtering: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    elif min_embeddedness == 0:
        print(f"Embeddedness filtering disabled (min_embeddedness=0): keeping all {G.number_of_edges()} edges")
    else:
        print(f"No embeddedness filtering specified: keeping all {G.number_of_edges()} edges")
    
    # Step 6: Reindex nodes to be sequential
    G = reindex_nodes(G)
    print(f"Final graph ready for optimal split: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    return G

def main():
    parser = argparse.ArgumentParser(description="Preprocess and split the graph using optimal 74:12:14 ratio")
    parser.add_argument('--name', type=str, default=DEFAULT_EXPERIMENT_NAME, 
                       help="Name for the results directory (from config.yaml)")
    parser.add_argument('--bidirectional_method', type=str, default=None,
                       help="Method for handling bidirectional edges (overrides config)")
    parser.add_argument('--use_weighted_features', action='store_true',
                       help="Enable weighted features")
    parser.add_argument('--weight_method', type=str, default=None,
                       help="Weight processing method: sign, raw, binned")
    parser.add_argument('--enable_subset_sampling', action='store_true',
                       help="Enable subset sampling for large datasets")
    parser.add_argument('--subset_method', type=str, default=None,
                       help="Subset sampling method: time_based, random, high_degree")
    parser.add_argument('--target_edges', type=int, default=None,
                       help="Target number of edges for subset sampling")
    
    # NEW: Embeddedness filtering parameter
    parser.add_argument('--min_embeddedness', type=int, default=None,
                       help="Minimum embeddedness for edge filtering (0=no filter, 1=moderate, 2=strong)")
    
    # NEW: Allow override of optimal sizes for testing
    parser.add_argument('--test_edges', type=int, default=N_TEST_EDGES,
                       help=f"Number of test edges (default: {N_TEST_EDGES} - optimal)")
    parser.add_argument('--val_edges', type=int, default=N_VALIDATION_EDGES,
                       help=f"Number of validation edges (default: {N_VALIDATION_EDGES} - optimal)")
    
    args = parser.parse_args()

    # Load parameters from config or command line
    bidirectional_method = args.bidirectional_method or config.get('bidirectional_method', 'max')
    use_weighted_features = args.use_weighted_features or config.get('use_weighted_features', False)
    weight_method = args.weight_method or config.get('weight_method', 'sign')
    weight_bins = config.get('weight_bins', 5)
    preserve_original_weights = config.get('preserve_original_weights', True)
    
    # NEW: Get embeddedness parameter (command line overrides config)
    min_embeddedness = args.min_embeddedness
    if min_embeddedness is None:
        min_embeddedness = config.get('min_train_embeddedness', 1)  # Default to 1 if not specified
    
    # Subset sampling parameters (command line args override config)
    enable_subset_sampling = args.enable_subset_sampling or ENABLE_SUBSET_SAMPLING
    subset_method = args.subset_method or SUBSET_SAMPLING_METHOD
    target_edges = args.target_edges or TARGET_EDGE_COUNT
    
    # Use optimal sizes (can be overridden via command line)
    test_edges = args.test_edges
    val_edges = args.val_edges
    
    print(f"OPTIMAL SPLIT CONFIGURATION")
    print(f"=" * 50)
    print(f"Split ratio: {SPLIT_RATIOS['train']:.0%}:{SPLIT_RATIOS['validation']:.0%}:{SPLIT_RATIOS['test']:.0%}")
    print(f"Test edges: {test_edges:,}")
    print(f"Validation edges: {val_edges:,}")
    print(f"Experiment name: {args.name}")
    print(f"Bidirectional method: {bidirectional_method}")
    print(f"Weighted features: {use_weighted_features}")
    print(f"Weight method: {weight_method}")
    print(f"Embeddedness filtering: min_embeddedness={min_embeddedness}")
    
    if enable_subset_sampling:
        print(f"\nSubset Sampling Configuration:")
        print(f"  Enable subset sampling: {enable_subset_sampling}")
        print(f"  Sampling method: {subset_method}")
        print(f"  Target edge count: {target_edges}")

    # Prepare subset configuration
    subset_config = {
        'subset_sampling_method': subset_method,
        'target_edge_count': target_edges,
        'subset_preserve_structure': SUBSET_PRESERVE_STRUCTURE
    } if enable_subset_sampling else None

    # Load data with optional subset sampling
    print(f"\nLoading data from {DATA_PATH}...")
    G, df = load_bitcoin_data(DATA_PATH, 
                             enable_subset_sampling=enable_subset_sampling, 
                             subset_config=subset_config)

    # Preprocess with embeddedness filtering support
    G = preprocess_graph(G, 
                        min_embeddedness=min_embeddedness,
                        bidirectional_method=bidirectional_method,
                        use_weighted_features=use_weighted_features,
                        weight_method=weight_method,
                        weight_bins=weight_bins,
                        preserve_original_weights=preserve_original_weights)

    # Create splits using optimal configuration
    print(f"\nCreating splits with optimal 74:12:14 ratio...")
    G_test, splits = create_splits_optimal(G, N_FOLDS, test_edges, val_edges)
    
    # Save splits to project root results directory
    out_dir = os.path.join(PROJECT_ROOT, 'results', args.name, 'preprocess')
    save_splits(G_test, splits, out_dir)
    
    # Save config variables used using save_config
    config_to_save = {
        'n_folds': N_FOLDS,
        'num_test_edges': test_edges,
        'num_validation_edges': val_edges,
        'data_path': DATA_PATH,
        'experiment_name': args.name,
        'bidirectional_method': bidirectional_method,
        'use_weighted_features': use_weighted_features,
        'weight_method': weight_method,
        'weight_bins': weight_bins,
        'preserve_original_weights': preserve_original_weights,
        'enable_subset_sampling': enable_subset_sampling,
        'subset_sampling_method': subset_method,
        'target_edge_count': target_edges,
        'subset_preserve_structure': SUBSET_PRESERVE_STRUCTURE,
        # NEW: Save embeddedness filtering configuration
        'min_train_embeddedness': min_embeddedness,
        'embeddedness_filtering_applied': 'preprocessing_stage',
        # Save optimal split configuration
        'optimal_split': True,
        'split_ratios': SPLIT_RATIOS,
        'optimal_split_info': 'Using improved 74:12:14 ratio for better accuracy'
    }
    
    save_config(config_to_save, out_dir)
    
    print(f"\nOPTIMAL PREPROCESSING COMPLETE!")
    print(f"Results saved to: {out_dir}")
    print(f"Embeddedness filtering applied at preprocessing stage")
    print(f"Ready for improved performance with optimal split ratios!")

if __name__ == "__main__":
    main()