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
from src.data_loader import load_undirected_graph_from_csv, save_metrics, save_prediction_results, save_config
from src.preprocessing import reindex_nodes
from src.feature_extraction import feature_matrix_from_graph
from src.models import predict_edge_signs, scale_testing_features
from src.evaluation import calculate_comparative_test_metrics

# Load config from YAML
with open(os.path.join(PROJECT_ROOT, 'config.yaml'), 'r') as f:
    config = yaml.safe_load(f)

N_FOLDS = config['default_n_folds']
CYCLE_LENGTH = config['cycle_length']
DEFAULT_EXPERIMENT_NAME = config['default_experiment_name']
THRESHOLD_TYPE = config.get('threshold_type', 'default_threshold')
N_TEST_PREDICTIONS = config.get('n_test_predictions', 300)

def load_test_set(test_path):
    """Load test set from CSV file."""
    G_test, _ = load_undirected_graph_from_csv(test_path)
    return G_test

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

def load_model_and_scaler(training_dir, fold=0):
    """Load trained model and scaler for a specific fold."""
    model_path = os.path.join(training_dir, f'model_fold_{fold}.joblib')
    scaler_path = os.path.join(training_dir, f'scaler_fold_{fold}.joblib')
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Model or scaler not found for fold {fold}")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def load_validation_metrics(validation_metrics_path):
    """Load validation metrics from JSON file."""
    with open(validation_metrics_path, 'r') as f:
        return json.load(f)

def test_model_on_set(G_test, model, scaler, cycle_length, threshold, 
                     use_weighted_features=False, weight_aggregation='product'):
    """
    Test edge sign prediction on the test set.
    NOTE: Embeddedness filtering was already applied during preprocessing.
    """
    print(f"Testing model on test set...")
    print(f"Configuration: weighted_features={use_weighted_features}, aggregation={weight_aggregation}")
    
    G_test = reindex_nodes(G_test)
    true_labels, predicted_labels, predicted_probabilities = [], [], []
    
    # Sample test edges - no embeddedness filtering here as it was done in preprocessing
    test_edges_sample = sample_n_edges(
        G_test, 
        sample_size=N_TEST_PREDICTIONS,
        pos_ratio=config.get('pos_test_edges_ratio', None)
        # NOTE: min_embeddedness removed - filtering done in preprocessing
    )
    
    print(f"Testing on {len(test_edges_sample)} edges...")
    
    for idx, edge_tuple in enumerate(test_edges_sample):
        if len(edge_tuple) == 2:
            u, v = edge_tuple
        elif len(edge_tuple) == 3:
            u, v, data = edge_tuple
        else:
            continue
            
        if not G_test.has_edge(u, v):
            print(f"Warning: Edge ({u}, {v}) not found in test set")
            continue
        
        # Remove current edge to avoid data leakage
        edge_data = G_test[u][v].copy()
        G_test.remove_edge(u, v)
        
        try:
            # Extract features for this edge
            X_test, y_test, _ = feature_matrix_from_graph(
                G_test, 
                edges=[(u, v, edge_data)], 
                k=cycle_length,
                use_weighted_features=use_weighted_features,
                weight_aggregation=weight_aggregation
            )
            
            if X_test.shape[0] == 0:
                print(f"Warning: No features extracted for edge ({u}, {v})")
                G_test.add_edge(u, v, **edge_data)
                continue
            
            # Scale features and predict
            X_test_scaled = scale_testing_features(X_test, scaler)
            pred, prob = predict_edge_signs(model, X_test_scaled, threshold=threshold)
            
            # Save results
            true_labels.append(y_test[0])
            predicted_labels.append(pred[0])
            predicted_probabilities.append(prob[0])
            
        except Exception as e:
            print(f"Error processing edge ({u}, {v}): {e}")
        finally:
            # Restore the edge
            G_test.add_edge(u, v, **edge_data)
        
        # Progress reporting
        if (idx + 1) % 50 == 0:
            print(f"  Progress: {idx+1}/{len(test_edges_sample)} edges processed")
    
    print(f"Testing completed: {len(true_labels)} predictions made")
    return true_labels, predicted_labels, predicted_probabilities

def main():
    parser = argparse.ArgumentParser(description="Test the model on the test set and save results.")
    parser.add_argument('--name', type=str, default=DEFAULT_EXPERIMENT_NAME, 
                       help="Name for the results directory")
    parser.add_argument('--n_folds', type=int, default=N_FOLDS, 
                       help=f"Number of folds (default: {N_FOLDS})")
    parser.add_argument('--cycle_length', type=int, default=CYCLE_LENGTH, 
                       help=f"Cycle length for feature extraction (default: {CYCLE_LENGTH})")
    parser.add_argument('--threshold_type', type=str, default=THRESHOLD_TYPE, 
                       help="Which threshold type to use for testing")
    args = parser.parse_args()

    # Set up paths
    test_path = os.path.join(PROJECT_ROOT, 'results', args.name, 'preprocess', 'test.csv')
    training_dir = os.path.join(PROJECT_ROOT, 'results', args.name, 'training')
    validation_metrics_path = os.path.join(PROJECT_ROOT, 'results', args.name, 'validation', 'metrics.json')
    out_dir = os.path.join(PROJECT_ROOT, 'results', args.name, 'testing')
    
    # Check prerequisites
    if not os.path.exists(test_path):
        print(f"ERROR: Test set not found: {test_path}")
        print("Please run preprocessing first.")
        return
    if not os.path.exists(training_dir):
        print(f"ERROR: Training directory not found: {training_dir}")
        print("Please run training first.")
        return
    if not os.path.exists(validation_metrics_path):
        print(f"ERROR: Validation metrics not found: {validation_metrics_path}")
        print("Please run validation first.")
        return

    os.makedirs(out_dir, exist_ok=True)

    # Load experiment configuration
    exp_config = load_experiment_config(args.name)
    use_weighted_features = exp_config.get('use_weighted_features', False)
    weight_aggregation = exp_config.get('weight_aggregation', 'product')

    print(f"Testing configuration:")
    print(f"  Experiment: {args.name}")
    print(f"  Weighted features: {use_weighted_features}")
    print(f"  Weight aggregation: {weight_aggregation}")
    print(f"  Threshold type: {args.threshold_type}")
    
    # Note about embeddedness filtering
    if exp_config.get('embeddedness_filtering_applied') == 'preprocessing_stage':
        min_embeddedness_used = exp_config.get('min_train_embeddedness', 'unknown')
        print(f"  Embeddedness filtering: Applied in preprocessing (level={min_embeddedness_used})")

    # Load test set, model, and validation metrics
    print(f"Loading test set from {test_path}...")
    G_test = load_test_set(test_path)
    
    print(f"Loading model and scaler from {training_dir}...")
    model, scaler = load_model_and_scaler(training_dir, fold=0)
    
    print(f"Loading validation metrics from {validation_metrics_path}...")
    validation_metrics = load_validation_metrics(validation_metrics_path)

    # Select threshold from validation results
    if 'actual' in validation_metrics:
        threshold = validation_metrics['actual'].get(args.threshold_type, 0.5)
    else:
        threshold = 0.5
        print(f"Warning: Could not find threshold in validation metrics, using default 0.5")
    
    print(f"Using threshold: {threshold} (type: {args.threshold_type})")

    # Run testing
    y_true, y_pred, y_prob = test_model_on_set(
        G_test, model, scaler, args.cycle_length, threshold,
        use_weighted_features=use_weighted_features, 
        weight_aggregation=weight_aggregation
    )

    if not y_true:
        print("ERROR: No predictions were made")
        return

    # Calculate test metrics
    test_metrics = calculate_comparative_test_metrics(y_true, y_pred, y_prob)
    
    # Print results
    print(f"\nTest Results:")
    print(f"  Accuracy: {test_metrics['actual']['accuracy']:.4f}")
    print(f"  F1 Score: {test_metrics['actual']['f1_score']:.4f}")
    print(f"  Precision: {test_metrics['actual']['precision']:.4f}")
    print(f"  Recall: {test_metrics['actual']['recall']:.4f}")
    if 'roc_auc' in test_metrics['actual']:
        print(f"  ROC AUC: {test_metrics['actual']['roc_auc']:.4f}")

    # Save results
    save_prediction_results(y_true, y_pred, y_prob, out_dir)
    save_metrics(test_metrics, out_dir)
    
    # Save configuration
    save_config({
        'n_folds': args.n_folds,
        'cycle_length': args.cycle_length,
        'threshold_type': args.threshold_type,
        'experiment_name': args.name,
        'use_weighted_features': use_weighted_features,
        'weight_aggregation': weight_aggregation,
        'threshold_used': threshold,
        'testing_completed': True
    }, out_dir)
    
    print(f"\nTest results saved to {out_dir}")

if __name__ == "__main__":
    main()