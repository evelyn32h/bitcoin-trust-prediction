import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

# Add src directory to Python path
sys.path.append(os.path.join('..'))

# Import custom modules
from src.data_loader import load_bitcoin_data
from src.preprocessing import map_to_unweighted_graph, ensure_connectivity

def analyze_network(G):
    """
    Calculate and return basic statistics of the network
    
    Parameters:
    G: NetworkX graph
    
    Returns:
    stats: Dictionary containing statistical information
    """
    stats = {}
    
    # Basic information
    stats['num_nodes'] = G.number_of_nodes()
    stats['num_edges'] = G.number_of_edges()
    
    # Edge weight analysis
    weights = [data['weight'] for _, _, data in G.edges(data=True)]
    stats['positive_edges'] = sum(1 for w in weights if w > 0)
    stats['negative_edges'] = sum(1 for w in weights if w < 0)
    stats['positive_ratio'] = stats['positive_edges'] / stats['num_edges']
    
    # Degree analysis
    in_degrees = [d for n, d in G.in_degree()]
    out_degrees = [d for n, d in G.out_degree()]
    
    stats['avg_in_degree'] = np.mean(in_degrees)
    stats['max_in_degree'] = np.max(in_degrees)
    stats['avg_out_degree'] = np.mean(out_degrees)
    stats['max_out_degree'] = np.max(out_degrees)
    
    # Connectivity analysis
    stats['weakly_connected_components'] = nx.number_weakly_connected_components(G)
    largest_wcc = max(nx.weakly_connected_components(G), key=len)
    stats['largest_wcc_size'] = len(largest_wcc)
    stats['largest_wcc_ratio'] = stats['largest_wcc_size'] / stats['num_nodes']
    
    return stats

def visualize_weight_distribution(G, save_path=None):
    """
    Visualize distribution of edge weights
    
    Parameters:
    G: NetworkX graph
    save_path: Path to save the image (optional)
    """
    weights = [data['weight'] for _, _, data in G.edges(data=True)]
    
    plt.figure(figsize=(10, 6))
    plt.hist(weights, bins=20, alpha=0.7)
    plt.title('Edge Weight Distribution')
    plt.xlabel('Weight')
    plt.ylabel('Frequency')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    
    plt.show()

def calculate_embeddedness(G):
    """
    Calculate embeddedness (number of shared neighbors) for each edge in the graph
    
    Parameters:
    G: NetworkX graph
    
    Returns:
    edge_embeddedness: Mapping from edges to their embeddedness values
    """
    # Create undirected graph to calculate shared neighbors
    G_undirected = G.to_undirected()
    
    # Calculate embeddedness for each edge
    edge_embeddedness = {}
    
    for u, v in G.edges():
        # Get shared neighbors
        shared_neighbors = set(G_undirected.neighbors(u)) & set(G_undirected.neighbors(v))
        edge_embeddedness[(u, v)] = len(shared_neighbors)
    
    return edge_embeddedness

def visualize_embeddedness(G, save_path=None):
    """
    Visualize distribution of edge embeddedness
    
    Parameters:
    G: NetworkX graph
    save_path: Path to save the image (optional)
    """
    edge_embeddedness = calculate_embeddedness(G)
    embeddedness_values = list(edge_embeddedness.values())
    
    plt.figure(figsize=(10, 6))
    plt.hist(embeddedness_values, bins=20, alpha=0.7)
    plt.title('Edge Embeddedness Distribution')
    plt.xlabel('Embeddedness (Number of Common Neighbors)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    # Calculate cumulative distribution
    max_embed = max(embeddedness_values)
    cum_dist = []
    for i in range(max_embed + 1):
        cum_dist.append(sum(1 for x in embeddedness_values if x <= i) / len(embeddedness_values))
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(max_embed + 1), cum_dist, marker='o', linestyle='-')
    plt.title('Cumulative Distribution of Edge Embeddedness')
    plt.xlabel('Embeddedness Threshold')
    plt.ylabel('Proportion of Edges')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        base_name = os.path.splitext(save_path)[0]
        cum_path = f"{base_name}_cumulative.png"
        plt.savefig(cum_path)
        print(f"Figure saved to {cum_path}")
    
    plt.show()

if __name__ == "__main__":
    # Load data
    data_path = os.path.join('..', 'data', 'soc-sign-bitcoinotc.csv')
    G, df = load_bitcoin_data(data_path)
    
    # Analyze network
    print("\nNetwork Statistics:")
    stats = analyze_network(G)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Visualize weight distribution
    print("\nGenerating weight distribution visualization...")
    results_dir = os.path.join('..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    visualize_weight_distribution(G, os.path.join(results_dir, 'weight_distribution.png'))
    
    # Calculate and visualize embeddedness
    print("\nCalculating and visualizing embeddedness...")
    visualize_embeddedness(G, os.path.join(results_dir, 'embeddedness_distribution.png'))
    
    # Create signed graph and analyze
    print("\nCreating signed graph...")
    G_signed = map_to_unweighted_graph(G)
    print("\nSigned Graph Statistics:")
    signed_stats = analyze_network(G_signed)
    for key, value in signed_stats.items():
        print(f"{key}: {value}")