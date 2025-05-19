import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src directory to Python path
sys.path.append(os.path.join('..'))

# Import custom modules
from src.data_loader import load_bitcoin_data
from src.preprocessing import map_to_unweighted_graph, ensure_connectivity, filter_by_embeddedness, create_balanced_dataset
from src.evaluation import evaluate_sign_predictor, plot_roc_curve, plot_confusion_matrix, analyze_false_positives

def main():
    """
    Test evaluation functions with simulated data
    """
    print("Testing evaluation functions with simulated data...")
    
    # Create simulated test data
    np.random.seed(42)
    y_true = np.concatenate([np.ones(80), -np.ones(20)])  # 80% positive, 20% negative
    y_pred = np.concatenate([np.ones(70), -np.ones(10), np.ones(5), -np.ones(15)])  # Some errors
    y_prob = np.random.random(100) * 0.5 + 0.5  # Random probabilities between 0.5 and 1.0 for positive predictions
    y_prob[y_pred == -1] = np.random.random(25) * 0.5  # Random probabilities between 0 and 0.5 for negative predictions
    
    # Evaluate predictions
    print("\nEvaluating simulated predictions...")
    metrics = evaluate_sign_predictor(y_true, y_pred, y_prob)
    
    print("\nEvaluation metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Plot ROC curve
    print("\nPlotting ROC curve...")
    plot_roc_curve(y_true, y_prob)
    
    # Plot confusion matrix
    print("\nPlotting confusion matrix...")
    plot_confusion_matrix(y_true, y_pred)
    
    print("\nTesting preprocessing optimization functions with actual data...")
    
    # Load actual data
    data_path = os.path.join('..', 'data', 'soc-sign-bitcoinotc.csv')
    G, df = load_bitcoin_data(data_path)
    
    # Convert to signed graph
    G_signed = map_to_unweighted_graph(G)
    G_connected = ensure_connectivity(G_signed)
    
    # Test embeddedness filtering
    print("\nTesting embeddedness filtering...")
    G_filtered = filter_by_embeddedness(G_connected, min_embeddedness=1)
    
    # Test balanced dataset creation
    print("\nTesting balanced dataset creation...")
    G_balanced = create_balanced_dataset(G_connected)
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    main()