import sys
import os
import numpy as np
import networkx as nx
from sklearn.model_selection import KFold
import logging
import time

sys.path.append(os.path.join('..'))

from src.data_loader import load_bitcoin_data
from src.preprocessing import filter_neutral_edges, map_to_unweighted_graph, ensure_connectivity, reindex_nodes
from src.feature_extraction import feature_matrix_from_graph
from src.models import train_edge_sign_classifier, predict_edge_signs, scale_training_features, scale_test_features
from src.evaluation import evaluate_sign_predictor, plot_roc_curve, plot_confusion_matrix

# Set logging to reduce noise
logging.basicConfig(level=logging.WARNING)

def strict_evaluation(G, n_folds=10, cycle_length=3):
    """
    Strict evaluation method to avoid data leakage - Optimized version
    """
    print(f"Running strict {n_folds}-fold evaluation (k={cycle_length})...")
    
    # Get all edges
    all_edges = list(G.edges(data=True))
    n_edges = len(all_edges)
    edge_indices = np.arange(n_edges)
    
    # Create folds
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    
    total_start = time.time()
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(edge_indices), 1):
        fold_start = time.time()
        print(f"\nFold {fold}/{n_folds} ({len(test_idx)} test edges)...")
        
        # Split edges
        train_edges = [all_edges[i] for i in train_idx]
        test_edges = [all_edges[i] for i in test_idx]
        
        # Create training graph (only training edges)
        G_train = nx.DiGraph()
        for u, v, data in train_edges:
            G_train.add_edge(u, v, **data)
        
        # Ensure connectivity and reindex
        G_train = ensure_connectivity(G_train)
        G_train = reindex_nodes(G_train)
        
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
        
        # Process test edges (no need to filter anymore as per Vide)
        print(f"Predicting {len(test_edges)} test edges...")
        pred_start = time.time()
        edge_times = []
        
        for idx, (u, v, data) in enumerate(test_edges):
            edge_start = time.time()
            
            # True label
            true_sign = data['weight']
            all_y_true.append(true_sign)
            
            # Create test graph without current edge
            G_test = nx.DiGraph()
            for j, (u2, v2, data2) in enumerate(test_edges):
                if j != idx:  # Skip current edge
                    G_test.add_edge(u2, v2, **data2)
            
            # Ensure connectivity and reindex for test graph
            if G_test.number_of_edges() > 0:
                G_test = ensure_connectivity(G_test)
                G_test = reindex_nodes(G_test)
            
            # Check if nodes exist in test graph
            if G_test.number_of_edges() == 0 or u not in G_test.nodes() or v not in G_test.nodes():
                all_y_pred.append(1)  # Default positive
                all_y_prob.append(0.5)
                continue
            
            try:
                # Extract features from TEST graph (not training!) - This is the key fix
                X_test, _, _ = feature_matrix_from_graph(
                    G_test,  # Changed from G_train to G_test
                    edges=[(u, v, {'weight': 1})],
                    k=cycle_length
                )
                
                # Scale features
                X_test_scaled = scale_test_features(X_test, scaler)
                
                # Predict
                pred, prob = predict_edge_signs(model, X_test_scaled)
                all_y_pred.append(pred[0])
                all_y_prob.append(prob[0])
                
            except Exception as e:
                # Default prediction on error
                all_y_pred.append(1)
                all_y_prob.append(0.5)
            
            # Timing and progress
            edge_time = time.time() - edge_start
            edge_times.append(edge_time)
            
            # Update progress every 50 edges
            if (idx + 1) % 50 == 0 or (idx + 1) == len(test_edges):
                avg_time = np.mean(edge_times[-50:]) if len(edge_times) >= 50 else np.mean(edge_times)
                elapsed = time.time() - pred_start
                rate = (idx + 1) / elapsed
                remaining = (len(test_edges) - idx - 1) / rate
                
                print(f"  Progress: {idx+1}/{len(test_edges)} ({rate:.1f} edges/sec, "
                      f"~{remaining/60:.1f} min remaining)")
        
        fold_time = time.time() - fold_start
        print(f"Fold {fold} completed in {fold_time/60:.1f} minutes")
    
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
    
    total_time = time.time() - total_start
    print(f"\nTotal evaluation time: {total_time/60:.1f} minutes")
    
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
    print("Running strict evaluation to avoid data leakage...")
    print("Using Vide's optimized feature extraction")
    
    # k=3
    metrics_k3 = strict_evaluation(G_connected, n_folds=10, cycle_length=3)
    
    print("\nStrict Evaluation Results (k=3):")
    for key, value in metrics_k3.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.4f}")
    
    # k=4 (if requested)
    if input("\nRun with k=4? (slower but better accuracy) (y/n): ").lower() == 'y':
        metrics_k4 = strict_evaluation(G_connected, n_folds=10, cycle_length=4)
        
        print("\nStrict Evaluation Results (k=4):")
        for key, value in metrics_k4.items():
            if isinstance(value, (int, float)):
                print(f"{key}: {value:.4f}")
    
    print("\n✓ Strict evaluation completed!")

if __name__ == "__main__":
    main()