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
    # TODO: select subset of nodes instead of edges, this will make adjacency matrix smaller
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    
    total_start = time.time()
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(edge_indices), 1):
        fold_start = time.time()
        print(f"\nFold {fold}/{n_folds} ({len(test_idx)} test edges)...")
        
        # Split edges
        #train_edges = [(all_edges[i][0], all_edges[i][1]) for i in train_idx]
                
        # Create training graph and preprocess
        train_edges = [all_edges[i] for i in train_idx]
        G_train = G.edge_subgraph(train_edges).copy()
        G_train = ensure_connectivity(G_train)
        G_train = reindex_nodes(G_train)
        print(f"Training Graph: {G_train.number_of_nodes()} nodes, {G_train.number_of_edges()} edges")
        
        # Create test graph and preprocess
        test_edges = [all_edges[i] for i in test_idx]
        G_test = G.edge_subgraph(test_edges).copy()
        G_test = ensure_connectivity(G_test)
        G_test = reindex_nodes(G_test)
        print(f"Test Graph: {G_test.number_of_nodes()} nodes, {G_test.number_of_edges()} edges")
                
        # Extract training features
        print("Extracting training features...")
        train_start = time.time()
        X_train, y_train, _ = feature_matrix_from_graph(G_train, k=cycle_length)
        train_time = time.time() - train_start
        print(f"Training feature extraction: {train_time:.2f}s")
        
        # Scale features
        X_train_scaled, scaler = scale_training_features(X_train)
        
        # Train model
        print("Training model...")
        model = train_edge_sign_classifier(X_train_scaled, y_train)
        
        # Sample subset of test edges to predict (to decrease computation time)
        test_edges_sample = sample_edges_with_positive_ratio(G_test, sample_size=predictions_per_fold, pos_ratio=pos_edges_ratio)
        
        print(f"Predicting {len(test_edges_sample)} test edges...")
        pred_start = time.time()
        edge_times = []
        
        # Testing
        for idx, (u, v, data) in enumerate(test_edges_sample):
            edge_start = time.time()
            
            if G_test.number_of_edges() == 0:
                logging.warning("Test graph is empty. This should not happen.")
            
            if not G_test.has_edge(u, v):
                logging.warning(f"Edge ({u}, {v}) not found in test. This should not happen.")               
            
            # Remove current edge from test graph
            # TODO: instead of removing and adding backm we can set weight to 0
            edge_data = G_test[u][v].copy()
            G_test.remove_edge(u, v)
            
            # TODO: should we run reindex and ensure connectivity?
            # Note, reindex does ruin the previous sampling, so we don't do it here 
             
            # Extract features from TEST graph
            X_test, y_test, _ = feature_matrix_from_graph(
                G_test,  # Changed from G_train to G_test
                edges=[(u, v, data)],  # 0 is the same as removing the edge for feature extraction
                k=cycle_length
            )
            
            # Scale features
            X_test_scaled = scale_test_features(X_test, scaler)
            
            # Predict
            # TODO: find right threshold
            pred, prob = predict_edge_signs(model, X_test_scaled, threshold=0.90)
            all_y_true.append(y_test[0])    # label
            all_y_pred.append(pred[0])      # prediction
            all_y_prob.append(prob[0])      # probability
            
            # Add the edge back to the test graph
            G_test.add_edge(u, v, **edge_data)
            
            # Timing and progress
            edge_time = time.time() - edge_start
            edge_times.append(edge_time)
            
            # Update progress every 50 edges
            if (idx + 1) % 50 == 0 or (idx + 1) == len(test_edges_sample):
                avg_time = np.mean(edge_times[-50:]) if len(edge_times) >= 50 else np.mean(edge_times)
                elapsed = time.time() - pred_start
                rate = (idx + 1) / elapsed
                remaining = (len(test_edges_sample) - idx - 1) / rate
                
                print(f"  Progress: {idx+1}/{len(test_edges)} ({rate:.1f} edges/sec, "
                      f"~{remaining/60:.1f} min remaining)")
        
        fold_time = time.time() - fold_start
        print(f"Fold {fold} completed in {fold_time/60:.1f} minutes")
    
    # Total time
    total_time = time.time() - total_start
    print(f"\nTotal evaluation time: {total_time/60:.1f} minutes")
    
    # Convert to arrays, evaluate, save, and plot
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
    metrics_k3 = reduced_evaluation(G_connected, n_folds=3, cycle_length=3, predictions_per_fold=20, pos_edges_ratio=0.5)
    
    print("\nStrict Evaluation Results (k=3):")
    for key, value in metrics_k3.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.4f}")
    
    print("\n✓ evaluation completed!")

if __name__ == "__main__":
    main()