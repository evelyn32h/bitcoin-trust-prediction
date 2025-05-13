import sys
import os
import numpy as np
import random

sys.path.append(os.path.join('..'))

from strict_evaluation import strict_evaluation
from src.data_loader import load_bitcoin_data
from src.preprocessing import filter_neutral_edges, map_to_unweighted_graph, ensure_connectivity

def strict_evaluation_sample(G, n_folds=5, cycle_length=3, sample_size=5000):
    """
    Run strict evaluation on a sample of edges for faster results
    """
    # Sample edges
    all_edges = list(G.edges(data=True))
    if sample_size < len(all_edges):
        random.seed(42)
        sampled_edges = random.sample(all_edges, sample_size)
        
        # Create sampled graph
        G_sampled = type(G)()
        for u, v, data in sampled_edges:
            G_sampled.add_edge(u, v, **data)
        
        print(f"Sampled graph: {G_sampled.number_of_nodes()} nodes, {G_sampled.number_of_edges()} edges")
        return strict_evaluation(G_sampled, n_folds=n_folds, cycle_length=cycle_length)
    else:
        return strict_evaluation(G, n_folds=n_folds, cycle_length=cycle_length)

def main():
    # Load data
    data_path = os.path.join('..', 'data', 'soc-sign-bitcoinotc.csv')
    G, df = load_bitcoin_data(data_path)
    
    # Preprocess
    print("Preprocessing graph...")
    G = filter_neutral_edges(G)
    G_signed = map_to_unweighted_graph(G)
    G_connected = ensure_connectivity(G_signed)
    
    print(f"Original graph: {G_connected.number_of_nodes()} nodes, {G_connected.number_of_edges()} edges")
    
    # Run sampled evaluation
    print("\n" + "="*50)
    print("Running sampled strict evaluation for quick results...")
    
    # Test with 5000 edges and 5 folds
    metrics = strict_evaluation_sample(G_connected, n_folds=5, cycle_length=3, sample_size=5000)
    
    print("\nSampled Evaluation Results (5000 edges, 5 folds, k=3):")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.4f}")
    
    print("\nNote: These are approximate results based on sampling")
    print("For publication-ready results, run full evaluation")

if __name__ == "__main__":
    main()