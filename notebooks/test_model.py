import os
import sys
import argparse
import pandas as pd
import numpy as np
import joblib
import yaml
import json
import time
import logging


# Add project root to sys.path for src imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utilities import sample_n_edges
from src.data_loader import load_models, load_undirected_graph_from_csv, load_metrics, save_prediction_results, save_metrics, save_config
from src.preprocessing import reindex_nodes
from src.feature_extraction import feature_matrix_from_graph
from src.models import predict_edge_signs, scale_test_features
from src.evaluation import calculate_evaluation_metrics, calculate_test_metrics

# Load config from YAML
with open(os.path.join(PROJECT_ROOT, 'config.yaml'), 'r') as f:
    config = yaml.safe_load(f)

N_FOLDS = config['default_n_folds']
CYCLE_LENGTH = config['cycle_length']
DEFAULT_EXPERIMENT_NAME = config['default_experiment_name']
THRESHOLD_TYPE = config.get('threshold_type', 'default_threshold')  # e.g., 'default_threshold', 'best_f1_threshold', 'best_accuracy_threshold'
N_TEST_PREDICTIONS = config.get('n_test_predictions', 200)  # Number of test predictions to sample

def load_test_set(test_path):
    G_test, _ = load_undirected_graph_from_csv(test_path)
    return G_test


def test_model_on_set(G_test, model, scaler, cycle_length, threshold):
    """
    Predict edge signs for all edges in the test set, removing each edge before prediction to avoid data leakage.
    Returns true labels, predicted labels, predicted probabilities, and edge list.
    """
    
    G_test = reindex_nodes(G_test)
    true_label, predicted_label, predicted_probabilities = [], [], []
    edge_times = []
    pred_start = time.time()
    
    # Sample a subset of edges to predict
    test_edges_sample, _, _ = sample_n_edges(G_test, sample_size=N_TEST_PREDICTIONS)
    
    for idx, (u, v, data) in enumerate(test_edges_sample):
        edge_start = time.time()
        if not G_test.has_edge(u, v):
            logging.warning(f"Edge ({u}, {v}) not found in test. This should not happen.")
        
        # Remove current edge to avoid data leakage
        edge_data = G_test[u][v].copy()
        G_test.remove_edge(u, v)
        
        # Extract and scale features for this edge
        X_test, y_test, _ = feature_matrix_from_graph(G_test, edges=[(u, v, data)], k=cycle_length)
        X_test_scaled = scale_test_features(X_test, scaler)
        
        # Predict edge sign
        pred, prob = predict_edge_signs(model, X_test_scaled, threshold=threshold)
        
        # Save results
        true_label.append(y_test[0])
        predicted_label.append(pred[0])
        predicted_probabilities.append(prob[0])
        
        # Restore the edge
        G_test.add_edge(u, v, **edge_data)
        
        # Timing and progress reporting
        edge_time = time.time() - edge_start
        edge_times.append(edge_time)
        if (idx + 1) % 50 == 0 or (idx + 1) == len(test_edges_sample):
            elapsed = time.time() - pred_start
            rate = (idx + 1) / elapsed
            remaining = (len(test_edges_sample) - idx - 1) / rate
            print(f"  Test: Progress: {idx+1}/{len(test_edges_sample)} ({rate:.1f} edges/sec, ~{remaining/60:.1f} min remaining)")
    return true_label, predicted_label, predicted_probabilities


def main():
    parser = argparse.ArgumentParser(description="Test the model on the test set and save results.")
    parser.add_argument('--name', type=str, default=DEFAULT_EXPERIMENT_NAME, help="Name for the results directory (from preprocess)")
    parser.add_argument('--n_folds', type=int, default=N_FOLDS, help=f"Number of folds (default: {N_FOLDS})")
    parser.add_argument('--cycle_length', type=int, default=CYCLE_LENGTH, help=f"Cycle length for feature extraction (default: {CYCLE_LENGTH})")
    parser.add_argument('--threshold_type', type=str, default=THRESHOLD_TYPE, help="Which threshold type to use for testing")
    args = parser.parse_args()

    test_path = os.path.join(PROJECT_ROOT, 'results', args.name, 'preprocess', 'test.csv')
    training_dir = os.path.join(PROJECT_ROOT, 'results', args.name, 'training')
    validation_metrics_path = os.path.join(PROJECT_ROOT, 'results', args.name, 'validation', 'metrics.csv')
    out_dir = os.path.join(PROJECT_ROOT, 'results', args.name, 'testing')
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(test_path):
        print(f"WARNING: The test set '{test_path}' does not exist. Please run preprocess.py first.")
        return
    if not os.path.exists(training_dir):
        print(f"WARNING: The training directory '{training_dir}' does not exist. Please run train_model.py first.")
        return
    if not os.path.exists(validation_metrics_path):
        print(f"WARNING: The validation metrics '{validation_metrics_path}' do not exist. Please run validate_model.py first.")
        return

    print(f"Loading test set from {test_path} ...")
    G_test = load_test_set(test_path)
    print(f"Loading models and scalers from {training_dir} ...")
    models_and_scalers = load_models(training_dir, args.n_folds)
    print(f"Loading validation metrics from {validation_metrics_path} ...")
    metrics = load_metrics(validation_metrics_path)

    # Select threshold
    threshold = metrics.get(args.threshold_type, 0.5)
    print(f"Using threshold {threshold} (type: {args.threshold_type})")

    # For simplicity, use the first model/scaler (or could ensemble)
    # TODO use aggregate of models?
    model, scaler = models_and_scalers[0]
    y_true, y_pred, y_prob = test_model_on_set(G_test, model, scaler, args.cycle_length, threshold)

    # Compute and print evaluation metrics
    test_metrics = calculate_test_metrics(y_true, y_pred, y_prob)
    print("\nTest Results")
    for key, value in test_metrics.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.4f}")

    # Save results
    save_prediction_results(y_true, y_pred, y_prob, out_dir)
    save_metrics(test_metrics, out_dir)
    
    # Save config variables used using save_config
    save_config({
        'n_folds': args.n_folds,
        'cycle_length': args.cycle_length,
        'threshold_type': args.threshold_type,
        'experiment_name': args.name
    }, out_dir)
    print(f"\n✓ Test results saved to {out_dir}")

if __name__ == "__main__":
    main()
