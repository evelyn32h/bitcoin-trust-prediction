import os
import sys
import random
import logging
import networkx as nx
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.WARNING)

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import load_bitcoin_data
from src.preprocessing import filter_neutral_edges, map_to_unweighted_graph, ensure_connectivity, normalize_edge_weights, reindex_nodes
from src.feature_extraction import feature_matrix_from_graph
from src.models import train_edge_sign_classifier, print_model_info, scale_training_features, scale_test_features
from src.evaluation import remove_edges, cml_evaluation

def train_and_test_model(num_edges_to_suppress=1, threshold=0.5):
    """
    Run edge sign prediction with an option to set a threshold for positive predictions.
    
    Parameters:
    num_edges_to_suppress: Number of edges to suppress for testing
    threshold: Probability threshold for classifying edges as positive
    """
    # Load data
    data_path = os.path.join('..', 'data', 'soc-sign-bitcoinotc.csv')
    G, _ = load_bitcoin_data(data_path)  # Load the directed graph
    
    # Preprocess the graph
    G = filter_neutral_edges(G)
    G = map_to_unweighted_graph(G)
    G = ensure_connectivity(G)
    G = reindex_nodes(G) # NOTE: this needs to be run every time before feature extraction if changes has been made to the graph
    
    # Extract features and labels from the undirected graph
    X, y, edges = feature_matrix_from_graph(G, k=4)
    
    # Suppress edges for testing
    removed_edges = random.sample(edges, num_edges_to_suppress)
    X_training, y_training, training_edges = remove_edges(X, y, edges, removed_edges)
    
    # Use new scaling functions from models.py
    X_training_scaled, scaler = scale_training_features(X_training)
    X_scaled = scale_test_features(X, scaler)
    
    # Train the model
    model = train_edge_sign_classifier(X_training_scaled, y_training)
    print("Model training complete.")
    
    # Predict and evaluate the suppressed edges (use scaled features)
    cml_evaluation(model, X_scaled, y, edges, removed_edges, threshold=threshold)
    
    # Print model information
    print_model_info(model)
        
if __name__ == "__main__":
    train_and_test_model(num_edges_to_suppress=5, threshold=0.8)
