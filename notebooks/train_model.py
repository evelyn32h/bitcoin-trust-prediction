import os
import sys
import argparse
import pandas as pd
import networkx as nx
import joblib
import yaml

# Add project root to sys.path for src imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_bitcoin_data, load_undirected_graph_from_csv, save_model, save_config
from src.preprocessing import map_to_unweighted_graph, ensure_connectivity, to_undirected, reindex_nodes
from src.feature_extraction import feature_matrix_from_graph
from src.models import train_edge_sign_classifier, scale_training_features

# Load config from YAML
with open(os.path.join(PROJECT_ROOT, 'config.yaml'), 'r') as f:
    config = yaml.safe_load(f)

N_FOLDS = config['default_n_folds']
CYCLE_LENGTH = config['cycle_length'] 
DEFAULT_EXPERIMENT_NAME = config['default_experiment_name']

def load_training_sets(training_dir, n_folds):
    """
    Loads training sets from CSV files in training_dir using load_undirected_graph_from_csv.
    Returns a list of G_train graphs.
    """
    training_sets = []
    for i in range(n_folds):
        train_path = os.path.join(training_dir, f'fold_{i}_train.csv')
        G_train, _ = load_undirected_graph_from_csv(train_path)
        training_sets.append(G_train)
    return training_sets

def load_experiment_config(experiment_name):
    """
    Load the configuration used for preprocessing this experiment.
    This is needed to extract features with the same settings.
    """
    config_path = os.path.join(PROJECT_ROOT, 'results', experiment_name, 'preprocess', 'config_used.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        print(f"Warning: No config found at {config_path}, using defaults")
        return {}

def train_model_on_set(G_train, cycle_length=4, use_weighted_features=False, weight_aggregation='product'):
    """
    Trains a model on the given training graph and returns the trained model and scaler.
    Now supports weighted features.
    """
    # Print some info about edge weights before reindexing
    G_train = reindex_nodes(G_train)
    
    # Extract features with Task #1 support
    X_train, y_train, _ = feature_matrix_from_graph(
        G_train, 
        k=cycle_length,
        use_weighted_features=use_weighted_features,
        weight_aggregation=weight_aggregation
    )
    
    print(f"Unique labels in y_train: {pd.Series(y_train).unique()}")
    
    # Print stats about features and labels
    print(f"Feature matrix shape: {X_train.shape}")
    print(f"Using weighted features: {use_weighted_features}")
    if use_weighted_features:
        print(f"Weight aggregation method: {weight_aggregation}")
    
    X_train_scaled, scaler = scale_training_features(X_train)
    model = train_edge_sign_classifier(X_train_scaled, y_train)
    return model, scaler

def train_all_models(training_sets, cycle_length, use_weighted_features=False, weight_aggregation='product'):
    """
    Trains a model and scaler for each training set and returns a list of (model, scaler) tuples.
    Now supports weighted features (Task #1).
    """
    results = []
    for fold, G_train in enumerate(training_sets):
        print(f"Training model for training set {fold} ...")
        model, scaler = train_model_on_set(
            G_train, 
            cycle_length=cycle_length,
            use_weighted_features=use_weighted_features,
            weight_aggregation=weight_aggregation
        )
        results.append((model, scaler))
    return results

def save_all_models(models_and_scalers, out_dir):
    """
    Saves all models and scalers to out_dir, one per fold.
    """
    for fold, (model, scaler) in enumerate(models_and_scalers):
        save_model(model, scaler, out_dir, fold)

def main():
    parser = argparse.ArgumentParser(description="Train models for each training set and save them.")
    parser.add_argument('--name', type=str, default=DEFAULT_EXPERIMENT_NAME, help="Name for the results directory (from preprocess)")
    parser.add_argument('--n_folds', type=int, default=N_FOLDS, help=f"Number of folds (default: {N_FOLDS})")
    parser.add_argument('--cycle_length', type=int, default=CYCLE_LENGTH, help=f"Cycle length for feature extraction (default: {CYCLE_LENGTH})")
    args = parser.parse_args()

    training_dir = os.path.join(PROJECT_ROOT, 'results', args.name, 'preprocess')
    out_dir = os.path.join(PROJECT_ROOT, 'results', args.name, 'training')

    # Make sure previous preprocessing step was run
    if not os.path.exists(training_dir):
        print(f"WARNING: The preprocess directory '{training_dir}' does not exist. Please run preprocess.py first.")
        return

    # Load experiment configuration (Task #1 support)
    exp_config = load_experiment_config(args.name)
    use_weighted_features = exp_config.get('use_weighted_features', False) # TODO: should be program args
    weight_aggregation = exp_config.get('weight_aggregation', 'product')
    
    print(f"Configuration loaded:")
    print(f"  Use weighted features: {use_weighted_features}")
    print(f"  Weight aggregation: {weight_aggregation}")

    print(f"Loading training sets from {training_dir} ...")
    training_sets = load_training_sets(training_dir, args.n_folds)

    models_and_scalers = train_all_models(
        training_sets, 
        args.cycle_length,
        use_weighted_features=use_weighted_features,
        weight_aggregation=weight_aggregation
    )
    save_all_models(models_and_scalers, out_dir)

    # Save config variables used using save_config
    save_config({
        'n_folds': args.n_folds,
        'cycle_length': args.cycle_length,
        'experiment_name': args.name,
        'use_weighted_features': use_weighted_features,
        'weight_aggregation': weight_aggregation
    }, out_dir)
    
    print("\n✓ All models trained and saved.")

if __name__ == "__main__":
    main()