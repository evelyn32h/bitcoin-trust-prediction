import sys
import os
import numpy as np
import pandas as pd

# Add src directory to Python path
sys.path.append(os.path.join('..'))

# Import custom modules
from src.data_loader import load_bitcoin_data
from src.preprocessing import (filter_neutral_edges, map_to_unweighted_graph, 
                              ensure_connectivity, reindex_nodes_sequentially,
                              filter_by_embeddedness, create_balanced_dataset)
from src.feature_extraction import feature_matrix_from_graph
from src.models import train_edge_sign_classifier, predict_edge_signs
from src.evaluation import evaluate_sign_predictor

def experiment_with_preprocessing(G, experiment_name, k=3):
    """
    Run experiment with specific preprocessing
    """
    print(f"\nRunning experiment: {experiment_name}")
    
    # Extract features
    X, y, edges = feature_matrix_from_graph(G, k=k)
    
    # Simple train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    
    # Train and evaluate
    model = train_edge_sign_classifier(X_train, y_train)
    y_pred, y_prob = predict_edge_signs(model, X_test)
    metrics = evaluate_sign_predictor(y_test, y_pred, y_prob)
    
    print(f"Results for {experiment_name}:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"AUC: {metrics.get('auc', 'N/A'):.4f}")
    print(f"False Positive Rate: {metrics['false_positive_rate']:.4f}")
    
    return metrics

def main():
    # Load data
    data_path = os.path.join('..', 'data', 'soc-sign-bitcoinotc.csv')
    G, df = load_bitcoin_data(data_path)
    
    # Base preprocessing
    G = filter_neutral_edges(G)
    G_signed = map_to_unweighted_graph(G)
    G_connected = ensure_connectivity(G_signed)
    G_processed = reindex_nodes_sequentially(G_connected)
    
    results = {}
    
    # 1. Original dataset
    results['original'] = experiment_with_preprocessing(
        G_processed, "Original dataset", k=3
    )
    
    # 2. Filter by embeddedness
    G_filtered = filter_by_embeddedness(G_processed, min_embeddedness=1)
    G_filtered = reindex_nodes_sequentially(G_filtered)
    results['filtered'] = experiment_with_preprocessing(
        G_filtered, "Filtered by embeddedness (min=1)", k=3
    )
    
    # 3. Balanced dataset
    G_balanced = create_balanced_dataset(G_processed)
    G_balanced = reindex_nodes_sequentially(G_balanced)
    results['balanced'] = experiment_with_preprocessing(
        G_balanced, "Balanced dataset", k=3
    )
    
    # Compare results
    print("\n" + "="*50)
    print("Results comparison:")
    for exp_name, metrics in results.items():
        print(f"\n{exp_name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  AUC: {metrics.get('auc', 'N/A'):.4f}")
        print(f"  FPR: {metrics['false_positive_rate']:.4f}")

if __name__ == "__main__":
    main()