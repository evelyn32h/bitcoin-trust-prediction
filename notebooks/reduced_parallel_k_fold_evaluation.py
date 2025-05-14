# Add parent directory to sys.path for src imports to work in both script and notebook contexts
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import networkx as nx
from sklearn.model_selection import KFold
import logging
import time

from src.utilities import sample_edges_with_positive_ratio

from src.data_loader import load_bitcoin_data
from src.preprocessing import filter_neutral_edges, map_to_unweighted_graph, ensure_connectivity, reindex_nodes
from src.feature_extraction import feature_matrix_from_graph
from src.models import train_edge_sign_classifier, predict_edge_signs, scale_training_features, scale_test_features
from src.evaluation import evaluate_sign_predictor, plot_roc_curve, plot_confusion_matrix

# Set logging to reduce noise
logging.basicConfig(level=logging.WARNING)



def save_and_plot_results(all_y_true, all_y_pred, all_y_prob, cycle_length, total_start):
    """
    Convert results to arrays, evaluate, save, and plot.
    """
    # Convert to arrays
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_prob = np.array(all_y_prob)

    # Evaluate
    metrics = evaluate_sign_predictor(all_y_true, all_y_pred, all_y_prob)

    # Save results
    results_dir = os.path.join('..', 'results', 'strict_evaluation')
    os.makedirs(results_dir, exist_ok=True)

    # Generate plots
    plot_roc_curve(all_y_true, all_y_prob, 
                   save_path=os.path.join(results_dir, f'strict_roc_k{cycle_length}.png'))
    plot_confusion_matrix(all_y_true, all_y_pred,
                         save_path=os.path.join(results_dir, f'strict_cm_k{cycle_length}.png'))

    return metrics



def process_fold(args):
    (fold, train_idx, test_idx, all_edges, G, cycle_length, predictions_per_fold, pos_edges_ratio) = args
    fold_start = time.time()
    
    # Create training graph and preprocess
    train_edges = [all_edges[i] for i in train_idx]
    G_train = G.edge_subgraph(train_edges).copy()
    G_train = ensure_connectivity(G_train)
    G_train = reindex_nodes(G_train)
    
    # Create test graph and preprocess
    test_edges = [all_edges[i] for i in test_idx]
    G_test = G.edge_subgraph(test_edges).copy()
    G_test = ensure_connectivity(G_test)
    G_test = reindex_nodes(G_test)
    
    # Extract training features
    X_train, y_train, _ = feature_matrix_from_graph(G_train, k=cycle_length)
    X_train_scaled, scaler = scale_training_features(X_train)
    model = train_edge_sign_classifier(X_train_scaled, y_train)
    
    # Sample subset of test edges to predict
    test_edges_sample = sample_edges_with_positive_ratio(G_test, sample_size=predictions_per_fold, pos_ratio=pos_edges_ratio)
    
    # Per-edge prediction (serial, can be parallelized as before if needed)
    y_true_fold, y_pred_fold, y_prob_fold = [], [], []
    edge_times = []
    pred_start = time.time()
    for idx, (u, v, data) in enumerate(test_edges_sample):
        if not G_test.has_edge(u, v):
            logging.warning(f"Edge ({u}, {v}) not found in test. This should not happen.")
        edge_start = time.time()
        
        # Remove current edge to avoid data leakage
        edge_data = G_test[u][v].copy()
        G_test.remove_edge(u, v)
        
        # Exract and scale features features
        X_test, y_test, _ = feature_matrix_from_graph(G_test, edges=[(u, v, data)], k=cycle_length)
        X_test_scaled = scale_test_features(X_test, scaler)
        
        # Predict edge sign
        pred, prob = predict_edge_signs(model, X_test_scaled, threshold=0.90)
        
        # Save results
        y_true_fold.append(y_test[0])
        y_pred_fold.append(pred[0])
        y_prob_fold.append(prob[0])
        
        # Restore the edge
        G_test.add_edge(u, v, **edge_data)
        
        # Timing and progress
        edge_time = time.time() - edge_start
        edge_times.append(edge_time)
        if (idx + 1) % 50 == 0 or (idx + 1) == len(test_edges_sample):
            avg_time = np.mean(edge_times[-50:]) if len(edge_times) >= 50 else np.mean(edge_times)
            elapsed = time.time() - pred_start
            rate = (idx + 1) / elapsed
            remaining = (len(test_edges_sample) - idx - 1) / rate
            print(f"  Fold {fold}: Progress: {idx+1}/{len(test_edges_sample)} ({rate:.1f} edges/sec, ~{remaining/60:.1f} min remaining)")
            
    fold_time = time.time() - fold_start
    return y_true_fold, y_pred_fold, y_prob_fold, fold, fold_time



def reduced_evaluation(G, n_folds=10, cycle_length=3, predictions_per_fold=20, pos_edges_ratio=0.5):
    """
    Strict evaluation method to avoid data leakage - Optimized version
    """
    print(f"Running strict {n_folds}-fold evaluation (k={cycle_length})...")
    # Get all edges
    all_edges = list(G.edges(data=False))
    n_edges = len(all_edges)
    edge_indices = np.arange(n_edges)
    
    # Create folds
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    # Prepare arguments for each fold
    fold_args = [
        (fold, train_idx, test_idx, all_edges, G, cycle_length, predictions_per_fold, pos_edges_ratio)
        for fold, (train_idx, test_idx) in enumerate(kf.split(edge_indices), 1)
    ]
    
    print(f"Running {n_folds} folds in parallel...")
    total_start = time.time()
    all_y_true, all_y_pred, all_y_prob = [], [], []
    fold_times = [0] * n_folds
    
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_fold, args) for args in fold_args]
        for future in as_completed(futures):
            y_true_fold, y_pred_fold, y_prob_fold, fold, fold_time = future.result()
            
            all_y_true.extend(y_true_fold)
            all_y_pred.extend(y_pred_fold)
            all_y_prob.extend(y_prob_fold)
            
            fold_times[fold-1] = fold_time
            print(f"Fold {fold} completed in {fold_time/60:.1f} minutes")
    
    total_time = time.time() - total_start
    print(f"\nTotal evaluation time: {total_time/60:.1f} minutes")
    metrics = save_and_plot_results(all_y_true, all_y_pred, all_y_prob, cycle_length, total_start)
    return metrics



def main():
    # Load data
    data_path = os.path.join('..', 'data', 'soc-sign-bitcoinotc.csv')
    G, df = load_bitcoin_data(data_path)
    
    # Preprocess
    print("Preprocessing graph...")
    G = filter_neutral_edges(G)
    G_signed = map_to_unweighted_graph(G)
    G_connected = ensure_connectivity(G_signed)
    
    print(f"Graph: {G_connected.number_of_nodes()} nodes, {G_connected.number_of_edges()} edges")
    
    # Run strict evaluation
    print("\n" + "="*50)
    print("Running quick evaluation...")
    
    # k=3
    metrics_k3 = reduced_evaluation(G_connected, n_folds=4, cycle_length=4, predictions_per_fold=500, pos_edges_ratio=0.2)
    
    print("\nStrict Evaluation Results (k=3):")
    for key, value in metrics_k3.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.4f}")
    
    print("\n✓ evaluation completed!")

if __name__ == "__main__":
    main()