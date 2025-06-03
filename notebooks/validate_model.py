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
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add project root to sys.path for src imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_undirected_graph_from_csv, save_metrics, save_prediction_results, save_config
from src.preprocessing import reindex_nodes
from src.feature_extraction import feature_matrix_from_graph
from src.models import predict_edge_signs, scale_testing_features
from src.evaluation import calculate_comparative_evaluation_metrics
from src.utilities import sample_n_edges

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
    Load validation sets from CSV files.
    Returns a list of G_val graphs (one per fold).
    """
    validation_sets = []
    for i in range(n_folds):
        val_path = os.path.join(validation_dir, f'fold_{i}_val.csv')
        if os.path.exists(val_path):
            G_val, _ = load_undirected_graph_from_csv(val_path)
            validation_sets.append(G_val)
        else:
            print(f"Warning: Validation file not found: {val_path}")
            validation_sets.append(None)
    return validation_sets

def load_experiment_config(experiment_name):
    """
    Load the configuration used for preprocessing this experiment.
    """
    config_path = os.path.join(PROJECT_ROOT, 'results', experiment_name, 'preprocess', 'config_used.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        print(f"Warning: No config found at {config_path}, using defaults")
        return {}

def load_models_and_scalers(training_dir, n_folds):
    """Load trained models and scalers for all folds."""
    models_and_scalers = []
    for i in range(n_folds):
        model_path = os.path.join(training_dir, f'model_fold_{i}.joblib')
        scaler_path = os.path.join(training_dir, f'scaler_fold_{i}.joblib')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            models_and_scalers.append((model, scaler))
        else:
            print(f"Warning: Model or scaler not found for fold {i}")
            models_and_scalers.append((None, None))
    return models_and_scalers

def validate_model_on_fold(args):
    """
    Validate a single fold by predicting edge signs for validation edges.
    NOTE: Embeddedness filtering was already applied during preprocessing.
    """
    (fold, G_val, model, scaler, cycle_length, predictions_per_fold, 
     pos_edges_ratio, use_weighted_features, weight_aggregation) = args

    if G_val is None or model is None or scaler is None:
        print(f"Skipping fold {fold} - missing data or model")
        return [], [], [], fold, 0

    fold_start = time.time()
    G_val = reindex_nodes(G_val)

    # Sample validation edges - no embeddedness filtering here as it was done in preprocessing
    validation_edges_sample = sample_n_edges(
        G_val, 
        sample_size=predictions_per_fold,
        pos_ratio=pos_edges_ratio
        # NOTE: min_embeddedness removed - filtering done in preprocessing
    )

    true_labels, predicted_labels, predicted_probabilities = [], [], []
    
    print(f"Fold {fold}: Validating on {len(validation_edges_sample)} edges...")
    
    for idx, edge_tuple in enumerate(validation_edges_sample):
        if len(edge_tuple) == 2:
            u, v = edge_tuple
        elif len(edge_tuple) == 3:
            u, v, data = edge_tuple
        else:
            continue
            
        if not G_val.has_edge(u, v):
            print(f"Warning: Edge ({u}, {v}) not found in validation set")
            continue
        
        # Remove current edge to avoid data leakage
        edge_data = G_val[u][v].copy()
        G_val.remove_edge(u, v)
        
        try:
            # Extract features for this edge
            X_test, y_test, _ = feature_matrix_from_graph(
                G_val, 
                edges=[(u, v, edge_data)], 
                k=cycle_length,
                use_weighted_features=use_weighted_features,
                weight_aggregation=weight_aggregation
            )
            
            if X_test.shape[0] == 0:
                G_val.add_edge(u, v, **edge_data)
                continue
            
            # Scale features and predict
            X_test_scaled = scale_testing_features(X_test, scaler)
            pred, prob = predict_edge_signs(model, X_test_scaled, threshold=0.5)
            
            # Save results
            true_labels.append(y_test[0])
            predicted_labels.append(pred[0])
            predicted_probabilities.append(prob[0])
            
        except Exception as e:
            print(f"Error processing edge ({u}, {v}) in fold {fold}: {e}")
        finally:
            # Restore the edge
            G_val.add_edge(u, v, **edge_data)
        
        # Progress reporting
        if (idx + 1) % 50 == 0:
            elapsed = time.time() - fold_start
            rate = (idx + 1) / elapsed
            remaining = (len(validation_edges_sample) - idx - 1) / rate
            print(f"  Fold {fold}: Progress: {idx+1}/{len(validation_edges_sample)} "
                  f"({rate:.1f} edges/sec, ~{remaining/60:.1f} min remaining)")

    fold_time = time.time() - fold_start
    print(f"Fold {fold} validation completed in {fold_time/60:.1f} minutes.")
    
    return true_labels, predicted_labels, predicted_probabilities, fold, fold_time

def validate_all_models(validation_sets, models_and_scalers, cycle_length, 
                       predictions_per_fold, pos_edges_ratio, 
                       use_weighted_features, weight_aggregation):
    """
    Run validation for all folds and return all results.
    """
    all_true_labels = []
    all_predicted_labels = []
    all_predicted_probabilities = []
    n_folds = len(validation_sets)
    
    # Prepare arguments for each fold
    fold_args = [
        (fold, G_val, model, scaler, cycle_length, predictions_per_fold, 
         pos_edges_ratio, use_weighted_features, weight_aggregation)
        for fold, (G_val, (model, scaler)) in enumerate(zip(validation_sets, models_and_scalers))
    ]
    
    print(f"Running validation for {n_folds} folds...")
    print(f"Configuration: weighted_features={use_weighted_features}, aggregation={weight_aggregation}")
    
    # Run validation for each fold
    for args in fold_args:
        y_true, y_pred, y_prob, fold, fold_time = validate_model_on_fold(args)
        all_true_labels.extend(y_true)
        all_predicted_labels.extend(y_pred)
        all_predicted_probabilities.extend(y_prob)
            
    return all_true_labels, all_predicted_labels, all_predicted_probabilities

def main():
    parser = argparse.ArgumentParser(description="Validate models on validation sets and save results.")
    parser.add_argument('--name', type=str, default=DEFAULT_EXPERIMENT_NAME, 
                       help="Name for the results directory")
    parser.add_argument('--n_folds', type=int, default=N_FOLDS, 
                       help=f"Number of folds (default: {N_FOLDS})")
    parser.add_argument('--cycle_length', type=int, default=CYCLE_LENGTH, 
                       help=f"Cycle length for feature extraction (default: {CYCLE_LENGTH})")
    parser.add_argument('--predictions_per_fold', type=int, default=PREDICTIONS_PER_FOLD, 
                       help="Number of validation edges to predict per fold")
    parser.add_argument('--pos_edges_ratio', type=float, default=POS_EDGES_RATIO, 
                       help="Desired positive edge ratio in validation sample")
    args = parser.parse_args()

    # Set up paths
    validation_dir = os.path.join(PROJECT_ROOT, 'results', args.name, 'preprocess')
    training_dir = os.path.join(PROJECT_ROOT, 'results', args.name, 'training')
    out_dir = os.path.join(PROJECT_ROOT, 'results', args.name, 'validation')

    # Check prerequisites
    if not os.path.exists(validation_dir):
        print(f"ERROR: Preprocessing directory not found: {validation_dir}")
        print("Please run preprocessing first.")
        return
    if not os.path.exists(training_dir):
        print(f"ERROR: Training directory not found: {training_dir}")
        print("Please run training first.")
        return

    os.makedirs(out_dir, exist_ok=True)

    # Load experiment configuration
    exp_config = load_experiment_config(args.name)
    use_weighted_features = exp_config.get('use_weighted_features', False)
    weight_aggregation = exp_config.get('weight_aggregation', 'product')

    print(f"Validation configuration:")
    print(f"  Experiment: {args.name}")
    print(f"  Weighted features: {use_weighted_features}")
    print(f"  Weight aggregation: {weight_aggregation}")
    print(f"  Predictions per fold: {args.predictions_per_fold}")
    print(f"  Positive edges ratio: {args.pos_edges_ratio}")
    
    # Note about embeddedness filtering
    if exp_config.get('embeddedness_filtering_applied') == 'preprocessing_stage':
        min_embeddedness_used = exp_config.get('min_train_embeddedness', 'unknown')
        print(f"  Embeddedness filtering: Applied in preprocessing (level={min_embeddedness_used})")

    # Load data and models
    print(f"Loading validation sets from {validation_dir}...")
    validation_sets = load_validation_sets(validation_dir, args.n_folds)
    
    print(f"Loading models and scalers from {training_dir}...")
    models_and_scalers = load_models_and_scalers(training_dir, args.n_folds)

    # Run validation
    all_true_labels, all_predicted_labels, all_predicted_probabilities = validate_all_models(
        validation_sets, models_and_scalers, args.cycle_length, 
        args.predictions_per_fold, args.pos_edges_ratio,
        use_weighted_features, weight_aggregation
    )

    if not all_true_labels:
        print("ERROR: No validation predictions were made")
        return

    # Calculate metrics
    metrics = calculate_comparative_evaluation_metrics(
        all_true_labels, all_predicted_labels, all_predicted_probabilities
    )
    
    # Print results
    print(f"\nValidation Results:")
    if 'actual' in metrics:
        actual = metrics['actual']
        print(f"  Best F1 Score: {actual.get('best_f1', 0):.4f}")
        print(f"  Best Accuracy: {actual.get('best_accuracy', 0):.4f}")
        print(f"  ROC AUC: {actual.get('roc_auc', 0):.4f}")
    
    # Save results
    save_prediction_results(all_true_labels, all_predicted_labels, all_predicted_probabilities, out_dir)
    save_metrics(metrics, out_dir)
    
    # Save configuration
    save_config({
        'n_folds': args.n_folds,
        'cycle_length': args.cycle_length,
        'predictions_per_fold': args.predictions_per_fold,
        'pos_edges_ratio': args.pos_edges_ratio,
        'experiment_name': args.name,
        'use_weighted_features': use_weighted_features,
        'weight_aggregation': weight_aggregation,
        'validation_completed': True
    }, out_dir)
    
    print(f"\nValidation results saved to {out_dir}")

if __name__ == "__main__":
    main()