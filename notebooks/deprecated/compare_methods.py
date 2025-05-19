import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join('..'))

from src.data_loader import load_bitcoin_data
from src.preprocessing import filter_neutral_edges, map_to_unweighted_graph, ensure_connectivity, reindex_nodes
from notebooks.deprecated.cross_validation_eval import run_cross_validation
from notebooks.deprecated.strict_evaluation import strict_evaluation

def compare_evaluation_methods(cycle_length=4):
    """
    Compare results from original method and strict method
    """
    # Load and preprocess data
    data_path = os.path.join('..', 'data', 'soc-sign-bitcoinotc.csv')
    G, df = load_bitcoin_data(data_path)
    
    print("Preprocessing graph...")
    G = filter_neutral_edges(G)
    G_signed = map_to_unweighted_graph(G)
    G_connected = ensure_connectivity(G_signed)
    G_processed = reindex_nodes(G_connected)
    
    # Run original method
    print("\n" + "="*50)
    print(f"Running original evaluation method (k={cycle_length})...")
    original_results = run_cross_validation(G_processed, n_splits=10, cycle_length=cycle_length)
    
    # Run strict method
    print("\n" + "="*50)
    print(f"Running strict evaluation method (k={cycle_length})...")
    strict_results = strict_evaluation(G_processed, n_folds=10, cycle_length=cycle_length)
    
    # Compare results
    print("\n" + "="*50)
    print("COMPARISON OF METHODS")
    print("="*50)
    
    print(f"\nOriginal method (k={cycle_length}):")
    for key, value in original_results.items():
        if key.startswith('avg_') and isinstance(value, (int, float)):
            print(f"{key}: {value:.4f}")
    
    print(f"\nStrict method (k={cycle_length}):")
    for key, value in strict_results.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.4f}")
    
    # Create comparison plot
    metrics = ['accuracy', 'false_positive_rate', 'auc']
    original_values = []
    strict_values = []
    
    for metric in metrics:
        if f'avg_{metric}' in original_results:
            original_values.append(original_results[f'avg_{metric}'])
        elif metric in original_results:
            original_values.append(original_results[metric])
        else:
            original_values.append(0)
            
        if metric in strict_results:
            strict_values.append(strict_results[metric])
        else:
            strict_values.append(0)
    
    # Plot
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, original_values, width, label='Original', alpha=0.8)
    bars2 = ax.bar(x + width/2, strict_values, width, label='Strict', alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title(f'Method Comparison (k={cycle_length})')
    ax.set_xticks(x)
    ax.set_xticklabels([m.title() for m in metrics])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    results_dir = os.path.join('..', 'results')
    plt.savefig(os.path.join(results_dir, f'method_comparison_k{cycle_length}.png'), dpi=300)
    plt.close()
    
    print(f"\n✓ Comparison plot saved!")

if __name__ == "__main__":
    compare_evaluation_methods(cycle_length=4)