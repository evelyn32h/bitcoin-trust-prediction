# Add parent directory to sys.path for src imports to work in both script and notebook contexts
import random
import sys
import os

from networkx import to_undirected
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import networkx as nx
from sklearn.model_selection import KFold
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


from src.utilities import sample_edges_with_positive_ratio
from src.data_loader import load_bitcoin_data
from src.preprocessing import filter_neutral_edges, map_to_unweighted_graph, ensure_connectivity, reindex_nodes, edge_bfs_holdout_split, sample_random_seed_edges
from src.feature_extraction import feature_matrix_from_graph
from src.models import train_edge_sign_classifier, predict_edge_signs, scale_training_features, scale_test_features
from src.evaluation import evaluate_sign_predictor, save_and_plot_results

# Set logging to reduce noise
logging.basicConfig(level=logging.WARNING)



def evaluate_fold(args):
    """
    Evaluate a single fold in the inductive evaluation pipeline.
    Trains a model on the training graph, predicts edge signs on a sampled subset of test edges,
    and returns the true labels, predicted labels, predicted probabilities, fold index, and fold runtime.

    Parameters:
        args: Tuple containing (fold, G_train, G_test, cycle_length, predictions_per_fold, pos_edges_ratio)
            - fold: Fold index (int)
            - G_train: Training graph (NetworkX Graph)
            - G_test: Test graph (NetworkX Graph)
            - cycle_length: Cycle length parameter for feature extraction (int)
            - predictions_per_fold: Number of test edges to predict (int)
            - pos_edges_ratio: Desired positive edge ratio in test sample (float)
    Returns:
        true_label: List of true edge labels for the test sample
        predicted_label: List of predicted edge labels for the test sample
        predicted_probabilities: List of predicted probabilities for the test sample
        fold: Fold index (int)
        fold_time: Runtime for this fold (float, seconds)
    """
    fold, G_train, G_test, cycle_length, predictions_per_fold, pos_edges_ratio = args
    
    fold_start = time.time()
    
    # Preprocessing
    # TODO: is ensure connectivity needed here?
    G_train = reindex_nodes(G_train)
    G_test = reindex_nodes(G_test)
    
    print(f"  {fold}: Train {G_train.number_of_nodes()}n/{G_train.number_of_edges()}e | Test {G_test.number_of_nodes()}n/{G_test.number_of_edges()}e")
    
    # Extract training features and train model
    X_train, y_train, _ = feature_matrix_from_graph(G_train, k=cycle_length)
    X_train_scaled, scaler = scale_training_features(X_train)
    model = train_edge_sign_classifier(X_train_scaled, y_train)
    
    # Sample subset of test edges to predict
    test_edges_sample = sample_edges_with_positive_ratio(G_test, sample_size=predictions_per_fold, pos_ratio=pos_edges_ratio)
    
    # Per-edge prediction
    true_label, predicted_label, predicted_probabilities = [], [], []
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
        true_label.append(y_test[0])
        predicted_label.append(pred[0])
        predicted_probabilities.append(prob[0])
        
        # Restore the edge
        G_test.add_edge(u, v, **edge_data)
        
        # Timing and progress
        edge_time = time.time() - edge_start
        edge_times.append(edge_time)
        if (idx + 1) % 50 == 0 or (idx + 1) == len(test_edges_sample):
            elapsed = time.time() - pred_start
            rate = (idx + 1) / elapsed
            remaining = (len(test_edges_sample) - idx - 1) / rate
            print(f"  Fold {fold}: Progress: {idx+1}/{len(test_edges_sample)} ({rate:.1f} edges/sec, ~{remaining/60:.1f} min remaining)")
            
    fold_time = time.time() - fold_start
    return true_label, predicted_label, predicted_probabilities, fold, fold_time



def create_training_split(G, n_folds):
    """
    Splits the graph G into n_folds training/test splits in parallel using edge_bfs_holdout_split.
    For each fold, a random seed edge is chosen, and a set of edges as close as possible to the seed edge (in edge-space) is held out for testing.
    Returns the list of seed edges and the corresponding list of (G_train, G_test) splits.
    
    Parameters:
        G: NetworkX Graph
        n_folds: Number of folds/splits to create
    Returns:
        seed_edges: List of seed edges used for each split
        training_split: List of (G_train, G_test) tuples for each fold
    """
    n_holdout_edges = int(G.number_of_edges() / n_folds) #TODO: make this a parameter 
    seed_edges = sample_random_seed_edges(G, n=n_folds, random_state=42)
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(edge_bfs_holdout_split, G, seed_edge, n_holdout_edges) for seed_edge in seed_edges]
        training_split = [future.result() for future in as_completed(futures)]
    
    print(f"Created training split from seed edges: {seed_edges}")
    return training_split



def inductive_evaluation(G, n_folds=10, cycle_length=3, predictions_per_fold=20, pos_edges_ratio=0.5):
    """
    Run inductive evaluation with multiple folds in parallel.
    For each fold, a training/test split is created, a model is trained, and predictions are made on a sampled subset of test edges.
    Returns lists of true labels, predicted labels, and predicted probabilities for all folds.

    Parameters:
        G: NetworkX Graph
        n_folds: Number of folds/splits to create (int)
        cycle_length: Cycle length parameter for feature extraction (int)
        predictions_per_fold: Number of test edges to predict per fold (int)
        pos_edges_ratio: Desired positive edge ratio in test sample (float)
    Returns:
        true_labels: List of true edge labels for all folds
        predicted_labels: List of predicted edge labels for all folds
        predicted_probabilities: List of predicted probabilities for all folds
    """
    print(f"Running {n_folds}-fold evaluation (k={cycle_length})...")
    
    # Create list of n (G_train, G_test) pairs
    training_split = create_training_split(G, n_folds)
        
    # Prepare arguments for each fold
    fold_args = [
        (fold, G_train, G_test, cycle_length, predictions_per_fold, pos_edges_ratio)
        for fold, (G_train, G_test) in enumerate(training_split)
    ]
    
    print(f"Running {n_folds} folds in parallel...")
    total_start = time.time()
    true_labels, predicted_labels, predicted_probabilities = [], [], []
    fold_times = [0] * n_folds
    
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(evaluate_fold, args) for args in fold_args]
        for future in as_completed(futures):
            y_true_fold, y_pred_fold, y_prob_fold, fold, fold_time = future.result()
            
            true_labels.extend(y_true_fold)
            predicted_labels.extend(y_pred_fold)
            predicted_probabilities.extend(y_prob_fold)
            
            fold_times[fold] = fold_time
            print(f"Fold {fold} completed in {fold_time/60:.1f} minutes")
    
    total_time = time.time() - total_start
    print(f"\nTotal evaluation time: {total_time/60:.1f} minutes")
    return true_labels, predicted_labels, predicted_probabilities



def main():
    # Load data
    data_path = os.path.join('..', 'data', 'soc-sign-bitcoinotc.csv')
    G, df = load_bitcoin_data(data_path)
    
    # Preprocess
    print("Preprocessing graph...")
    #G = filter_neutral_edges(G) #TODO  this doesnt do anything??
    G = map_to_unweighted_graph(G)
    G = ensure_connectivity(G)
    G = to_undirected(G)
    G = reindex_nodes(G)
    
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Run evaluation
    print("\n" + "="*50)
    print("Running Inductive Evaluation...")
    
    evaluation_config = {
        "n_folds": 4,
        "cycle_length": 4,
        "predictions_per_fold": 300,
        "pos_edges_ratio": 0.89
    }
    
    true_labels, predicted_labels, predicted_probabilities = inductive_evaluation(G, **evaluation_config)
    
    print("\n✓ evaluation completed!")
    
    # calculate and print metrics
    metrics = evaluate_sign_predictor(true_labels, predicted_labels, predicted_probabilities)
    print("\nInductive Evaluation Results")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.4f}")
    
    # Save and plot results 
    save_and_plot_results(true_labels, predicted_labels, predicted_probabilities, "inductive", evaluation_config["cycle_length"])     

if __name__ == "__main__":
    main()