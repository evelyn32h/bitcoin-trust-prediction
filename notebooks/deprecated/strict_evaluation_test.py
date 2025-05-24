import sys
import os
import numpy as np
import networkx as nx
from sklearn.model_selection import KFold
import logging
import time
import random

sys.path.append(os.path.join('..'))

from src.data_loader import load_bitcoin_data
from src.preprocessing import filter_neutral_edges, map_to_unweighted_graph, ensure_connectivity, reindex_nodes
from src.feature_extraction import feature_matrix_from_graph
from src.models import train_edge_sign_classifier, predict_edge_signs, scale_training_features, scale_test_features
from src.evaluation import evaluate_sign_predictor, plot_roc_curve, plot_confusion_matrix

logging.basicConfig(level=logging.WARNING)

def strict_evaluation_test(G, n_folds=3, cycle_length=3, edges_per_fold=20):
    """
    Vide's suggested test version - faster evaluation for optimization
    """
    print(f"Running test evaluation ({n_folds} folds, {edges_per_fold} edges per fold, k={cycle_length})...")
    
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
        
        # Split edges
        train_edges = [all_edges[i] for i in train_idx]
        test_edges = [all_edges[i] for i in test_idx]
        
        # Sample test edges (ensure 50% negative edges as per Vide)
        pos_test_edges = [(i, e) for i, e in enumerate(test_edges) if e[2]['weight'] > 0]
        neg_test_edges = [(i, e) for i, e in enumerate(test_edges) if e[2]['weight'] < 0]
        
        # Sample 50/50 positive/negative
        n_pos_sample = edges_per_fold // 2
        n_neg_sample = edges_per_fold // 2
        
        # Handle case where we don't have enough negative edges
        if len(neg_test_edges) < n_neg_sample:
            n_neg_sample = len(neg_test_edges)
            n_pos_sample = edges_per_fold - n_neg_sample
        
        sampled_pos = random.sample(pos_test_edges, min(n_pos_sample, len(pos_test_edges)))
        sampled_neg = random.sample(neg_test_edges, min(n_neg_sample, len(neg_test_edges)))
        sampled_test_edges = sampled_pos + sampled_neg
        
        print(f"\nFold {fold}/{n_folds} (sampled {len(sampled_test_edges)} test edges: {len(sampled_pos)} pos, {len(sampled_neg)} neg)...")
        
        # Create training graph
        G_train = nx.DiGraph()
        for u, v, data in train_edges:
            G_train.add_edge(u, v, **data)
        
        # Ensure connectivity and reindex
        G_train = ensure_connectivity(G_train)
        G_train = reindex_nodes(G_train)
        
        # Extract training features
        print("Extracting training features...")
        X_train, y_train, _ = feature_matrix_from_graph(G_train, k=cycle_length)
        
        # Scale features
        X_train_scaled, scaler = scale_training_features(X_train)
        
        # Train model
        print("Training model...")
        model = train_edge_sign_classifier(X_train_scaled, y_train)
        
        # Process sampled test edges
        print(f"Predicting {len(sampled_test_edges)} sampled test edges...")
        
        for idx_in_test, (u, v, data) in sampled_test_edges:
            # True label
            true_sign = data['weight']
            all_y_true.append(true_sign)
            
            # Create test graph without current edge
            G_test = nx.DiGraph()
            for j, (u2, v2, data2) in enumerate(test_edges):
                if j != idx_in_test:  # Skip current edge
                    G_test.add_edge(u2, v2, **data2)
            
            # Ensure connectivity and reindex
            if G_test.number_of_edges() > 0:
                G_test = ensure_connectivity(G_test)
                G_test = reindex_nodes(G_test)
            
            # Check if nodes exist in test graph
            if G_test.number_of_edges() == 0 or u not in G_test.nodes() or v not in G_test.nodes():
                all_y_pred.append(1)
                all_y_prob.append(0.5)
                continue
            
            try:
                # Extract features from TEST graph (not training!) - This is the key fix
                X_test, _, _ = feature_matrix_from_graph(
                    G_test,  # Changed from G_train to G_test
                    edges=[(u, v, {'weight': 1})],
                    k=cycle_length
                )
                
                # Scale and predict
                X_test_scaled = scale_test_features(X_test, scaler)
                pred, prob = predict_edge_signs(model, X_test_scaled)
                all_y_pred.append(pred[0])
                all_y_prob.append(prob[0])
                
            except Exception as e:
                all_y_pred.append(1)
                all_y_prob.append(0.5)
        
        fold_time = time.time() - fold_start
        print(f"Fold {fold} completed in {fold_time:.1f} seconds")
    
    # Convert to arrays
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_prob = np.array(all_y_prob)
    
    # Evaluate
    metrics = evaluate_sign_predictor(all_y_true, all_y_pred, all_y_prob)
    
    total_time = time.time() - total_start
    print(f"\nTotal evaluation time: {total_time:.1f} seconds")
    
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
    
    # Run test evaluation
    print("\n" + "="*50)
    print("Running Vide's test evaluation (3 folds, 20 edges per fold, 50% negative)...")
    
    metrics = strict_evaluation_test(G_connected, n_folds=3, cycle_length=3, edges_per_fold=20)
    
    print("\nTest Results:")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.4f}")
    
    print("\nNote: These are approximate results for testing purposes")
    print("Full evaluation is running in strict_evaluation.py")

if __name__ == "__main__":
    main()