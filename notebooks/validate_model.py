import os
import sys
import argparse
import pandas as pd
import numpy as np
import joblib
import yaml
from sklearn.metrics import accuracy_score
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import time

# Add project root to sys.path for src imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_undirected_graph_from_csv, save_metrics, save_prediction_results, load_models, save_config
from src.preprocessing import reindex_nodes
from src.feature_extraction import feature_matrix_from_graph
from src.models import predict_edge_signs, scale_test_features
from src.evaluation import calculate_evaluation_metrics, evaluate_sign_predictor
from src.utilities import sample_edges_with_positive_ratio

# Load config from YAML
with open(os.path.join(PROJECT_ROOT, 'config.yaml'), 'r') as f:
    config = yaml.safe_load(f)

N_FOLDS = config['default_n_folds']
CYCLE_LENGTH = config['cycle_length']
DEFAULT_EXPERIMENT_NAME = config['default_experiment_name']
PREDICTIONS_PER_FOLD = config.get('nr_val_predictions', 200)
POS_EDGES_RATIO = config.get('pos_edges_ratio', 0.5)

def load_validation_sets(validation_dir, n_folds):
    """
    Loads validation sets from CSV files in validation_dir using load_undirected_graph_from_csv.
    Returns a list of G_val graphs (one per fold).
    """
    validation_sets = []
    for i in range(n_folds):
        val_path = os.path.join(validation_dir, f'fold_{i}_val.csv')
        G_val, _ = load_undirected_graph_from_csv(val_path)
        validation_sets.append(G_val)
    return validation_sets

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

def validate_model_on_fold(args):
    """
    Validate a single fold: predict edge signs for a sampled subset of validation edges.
    Removes each edge before prediction to avoid data leakage (like inductive_evalution.py).
    Returns true labels, predicted labels, predicted probabilities, fold index, and fold runtime.
    Now supports weighted features.
    """
    fold, G_val, model, scaler, cycle_length, predictions_per_fold, pos_edges_ratio, use_weighted_features, weight_aggregation = args

    fold_start = time.time()
    G_val = reindex_nodes(G_val)

    # Sample a subset of validation edges to predict, with a controlled positive/negative ratio
    validation_edges_sample = sample_edges_with_positive_ratio(G_val, sample_size=predictions_per_fold, pos_ratio=pos_edges_ratio)

    true_label, predicted_label, predicted_probabilities = [], [], []
    edge_times = []
    pred_start = time.time()
    
    for idx, (u, v, data) in enumerate(validation_edges_sample):
        edge_start = time.time()
        
        if not G_val.has_edge(u, v):
            logging.warning(f"Edge ({u}, {v}) not found in validation. This should not happen.")
        
        # Remove current edge to avoid data leakage
        edge_data = G_val[u][v].copy()
        G_val.remove_edge(u, v)
        
        # Extract and scale features for this edge with Task #1 support
        X_test, y_test, _ = feature_matrix_from_graph(
            G_val, 
            edges=[(u, v, data)], 
            k=cycle_length,
            use_weighted_features=use_weighted_features,
            weight_aggregation=weight_aggregation
        )
        X_test_scaled = scale_test_features(X_test, scaler)
        
        # Predict edge sign
        pred, prob = predict_edge_signs(model, X_test_scaled, threshold=0.90)
        
        # Save results
        true_label.append(y_test[0])
        predicted_label.append(pred[0])
        predicted_probabilities.append(prob[0])
        
        # Restore the edge
        G_val.add_edge(u, v, **edge_data)
        
        # Timing and progress reporting
        edge_time = time.time() - edge_start
        edge_times.append(edge_time)
        if (idx + 1) % 50 == 0 or (idx + 1) == len(validation_edges_sample):
            elapsed = time.time() - pred_start
            rate = (idx + 1) / elapsed
            remaining = (len(validation_edges_sample) - idx - 1) / rate
            print(f"  Fold {fold}: Progress: {idx+1}/{len(validation_edges_sample)} ({rate:.1f} edges/sec, ~{remaining/60:.1f} min remaining)")
    fold_time = time.time() - fold_start
    
    return true_label, predicted_label, predicted_probabilities, fold, fold_time

def validate_all_models(validation_sets, models_and_scalers, cycle_length, predictions_per_fold, pos_edges_ratio, use_weighted_features, weight_aggregation):
    """
    Runs validation for all folds in parallel and returns all true labels, predicted labels, and predicted probabilities.
    Now supports weighted features.
    """
    all_true_labels = []
    all_predicted_labels = []
    all_predicted_probabilities = []
    n_folds = len(validation_sets)
    
    # Prepare arguments for each fold
    fold_args = [
        (fold, G_val, model, scaler, cycle_length, predictions_per_fold, pos_edges_ratio, use_weighted_features, weight_aggregation)
        for fold, (G_val, (model, scaler)) in enumerate(zip(validation_sets, models_and_scalers))
    ]
    
    print(f"Running validation for {n_folds} folds in parallel...")
    print(f"settings: weighted_features={use_weighted_features}, aggregation={weight_aggregation}")
    
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(validate_model_on_fold, args) for args in fold_args]
        for future in as_completed(futures):
            y_true, y_pred, y_prob, fold, fold_time = future.result()
            all_true_labels.extend(y_true)
            all_predicted_labels.extend(y_pred)
            all_predicted_probabilities.extend(y_prob)
            print(f"Fold {fold} validation completed in {fold_time/60:.1f} minutes.")
            
    return all_true_labels, all_predicted_labels, all_predicted_probabilities

def main():
    parser = argparse.ArgumentParser(description="Validate models on validation sets and save results.")
    parser.add_argument('--name', type=str, default=DEFAULT_EXPERIMENT_NAME, help="Name for the results directory (from preprocess)")
    parser.add_argument('--n_folds', type=int, default=N_FOLDS, help=f"Number of folds (default: {N_FOLDS})")
    parser.add_argument('--cycle_length', type=int, default=CYCLE_LENGTH, help=f"Cycle length for feature extraction (default: {CYCLE_LENGTH})")
    parser.add_argument('--predictions_per_fold', type=int, default=PREDICTIONS_PER_FOLD, help="Number of validation edges to predict per fold")
    parser.add_argument('--pos_edges_ratio', type=float, default=POS_EDGES_RATIO, help="Desired positive edge ratio in validation sample")
    args = parser.parse_args()

    # Set up paths for validation, training, and output directories
    validation_dir = os.path.join(PROJECT_ROOT, 'results', args.name, 'preprocess')
    training_dir = os.path.join(PROJECT_ROOT, 'results', args.name, 'training')
    out_dir = os.path.join(PROJECT_ROOT, 'results', args.name, 'validation')

    if not os.path.exists(validation_dir):
        print(f"WARNING: The preprocess directory '{validation_dir}' does not exist. Please run preprocess.py first.")
        return
    if not os.path.exists(training_dir):
        print(f"WARNING: The training directory '{training_dir}' does not exist. Please run train_model.py first.")
        return

    # Load experiment configuration (Task #1 support)
    exp_config = load_experiment_config(args.name)
    use_weighted_features = exp_config.get('use_weighted_features', False)
    weight_aggregation = exp_config.get('weight_aggregation', 'product')

    print(f"Loading validation sets from {validation_dir} ...")
    validation_sets = load_validation_sets(validation_dir, args.n_folds)
    print(f"Loading models and scalers from {training_dir} ...")
    models_and_scalers = load_models(training_dir, args.n_folds)

    # Run validation for all folds in parallel
    all_true_labels, all_predicted_labels, all_predicted_probabilities = validate_all_models(
        validation_sets, models_and_scalers, args.cycle_length, args.predictions_per_fold, args.pos_edges_ratio,
        use_weighted_features, weight_aggregation
    )

    # Compute and print evaluation metrics
    metrics = calculate_evaluation_metrics(all_true_labels, all_predicted_labels, all_predicted_probabilities)
    print("\nValidation Results")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.4f}")

    # Save results to output directory
    save_prediction_results(all_true_labels, all_predicted_labels, all_predicted_probabilities, out_dir)
    save_metrics(metrics, out_dir)
    
    # Save config variables used using save_config
    save_config({
        'n_folds': args.n_folds,
        'cycle_length': args.cycle_length,
        'predictions_per_fold': args.predictions_per_fold,
        'pos_edges_ratio': args.pos_edges_ratio,
        'experiment_name': args.name,
        'use_weighted_features': use_weighted_features,
        'weight_aggregation': weight_aggregation
    }, out_dir)
    print(f"\n✓ Validation results saved to {out_dir}")

if __name__ == "__main__":
    main()