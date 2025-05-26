import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
from collections import Counter
import time
from datetime import datetime

# Add src directory to Python path
sys.path.append(os.path.join('..'))

# Import custom modules
from src.data_loader import load_bitcoin_data
from src.preprocessing import (map_to_unweighted_graph, ensure_connectivity, 
                             filter_neutral_edges, reindex_nodes)
from src.feature_extraction import feature_matrix_from_graph
from src.utilities import sample_edges_with_positive_ratio
from src.evaluation import (analyze_network, visualize_weight_distribution, 
                          visualize_degree_distribution, calculate_embeddedness,
                          visualize_embeddedness, analyze_temporal_patterns,
                          visualize_connectivity_analysis, visualize_preprocessing_pipeline)

# Configure matplotlib and seaborn for better plots
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
sns.set_palette("husl")

def analyze_network(G, graph_name="Graph"):
    """
    Calculate and return comprehensive statistics of the network
    
    Parameters:
    G: NetworkX graph
    graph_name: Name for the graph being analyzed
    
    Returns:
    stats: Dictionary containing statistical information
    """
    print(f"\n{'='*60}")
    print(f"NETWORK ANALYSIS: {graph_name}")
    print(f"{'='*60}")
    
    stats = {}
    
    # Basic information
    stats['num_nodes'] = G.number_of_nodes()
    stats['num_edges'] = G.number_of_edges()
    stats['graph_density'] = nx.density(G)
    
    print(f"Nodes: {stats['num_nodes']:,}")
    print(f"Edges: {stats['num_edges']:,}")
    print(f"Density: {stats['graph_density']:.6f}")
    
    # Edge weight analysis
    if G.edges():
        weights = [data.get('weight', 1) for _, _, data in G.edges(data=True)]
        stats['positive_edges'] = sum(1 for w in weights if w > 0)
        stats['negative_edges'] = sum(1 for w in weights if w < 0)
        stats['zero_edges'] = sum(1 for w in weights if w == 0)
        stats['positive_ratio'] = stats['positive_edges'] / stats['num_edges']
        stats['negative_ratio'] = stats['negative_edges'] / stats['num_edges']
        
        stats['min_weight'] = min(weights)
        stats['max_weight'] = max(weights)
        stats['mean_weight'] = np.mean(weights)
        stats['std_weight'] = np.std(weights)
        
        print(f"\nEdge Weight Distribution:")
        print(f"  Positive edges: {stats['positive_edges']:,} ({stats['positive_ratio']:.2%})")
        print(f"  Negative edges: {stats['negative_edges']:,} ({stats['negative_ratio']:.2%})")
        if stats['zero_edges'] > 0:
            print(f"  Zero weight edges: {stats['zero_edges']:,}")
        print(f"  Weight range: [{stats['min_weight']}, {stats['max_weight']}]")
        print(f"  Mean weight: {stats['mean_weight']:.3f} ± {stats['std_weight']:.3f}")
    
    # Degree analysis
    if isinstance(G, nx.DiGraph):
        in_degrees = [d for n, d in G.in_degree()]
        out_degrees = [d for n, d in G.out_degree()]
        
        stats['avg_in_degree'] = np.mean(in_degrees)
        stats['max_in_degree'] = np.max(in_degrees)
        stats['avg_out_degree'] = np.mean(out_degrees)
        stats['max_out_degree'] = np.max(out_degrees)
        
        print(f"\nDegree Statistics (Directed):")
        print(f"  Average in-degree: {stats['avg_in_degree']:.2f}")
        print(f"  Maximum in-degree: {stats['max_in_degree']}")
        print(f"  Average out-degree: {stats['avg_out_degree']:.2f}")
        print(f"  Maximum out-degree: {stats['max_out_degree']}")
    else:
        degrees = [d for n, d in G.degree()]
        stats['avg_degree'] = np.mean(degrees)
        stats['max_degree'] = np.max(degrees)
        
        print(f"\nDegree Statistics (Undirected):")
        print(f"  Average degree: {stats['avg_degree']:.2f}")
        print(f"  Maximum degree: {stats['max_degree']}")
    
    # Connectivity analysis
    if isinstance(G, nx.DiGraph):
        stats['weakly_connected_components'] = nx.number_weakly_connected_components(G)
        stats['strongly_connected_components'] = nx.number_strongly_connected_components(G)
        
        if stats['weakly_connected_components'] > 0:
            largest_wcc = max(nx.weakly_connected_components(G), key=len)
            stats['largest_wcc_size'] = len(largest_wcc)
            stats['largest_wcc_ratio'] = stats['largest_wcc_size'] / stats['num_nodes']
        
        if stats['strongly_connected_components'] > 0:
            largest_scc = max(nx.strongly_connected_components(G), key=len)
            stats['largest_scc_size'] = len(largest_scc)
            stats['largest_scc_ratio'] = stats['largest_scc_size'] / stats['num_nodes']
        
        print(f"\nConnectivity (Directed):")
        print(f"  Weakly connected components: {stats['weakly_connected_components']}")
        print(f"  Largest WCC size: {stats.get('largest_wcc_size', 0):,} ({stats.get('largest_wcc_ratio', 0):.2%})")
        print(f"  Strongly connected components: {stats['strongly_connected_components']}")
        print(f"  Largest SCC size: {stats.get('largest_scc_size', 0):,} ({stats.get('largest_scc_ratio', 0):.2%})")
    else:
        stats['connected_components'] = nx.number_connected_components(G)
        if stats['connected_components'] > 0:
            largest_cc = max(nx.connected_components(G), key=len)
            stats['largest_cc_size'] = len(largest_cc)
            stats['largest_cc_ratio'] = stats['largest_cc_size'] / stats['num_nodes']
        
        print(f"\nConnectivity (Undirected):")
        print(f"  Connected components: {stats['connected_components']}")
        print(f"  Largest CC size: {stats.get('largest_cc_size', 0):,} ({stats.get('largest_cc_ratio', 0):.2%})")
    
    return stats

def visualize_weight_distribution(G, save_path=None, graph_name="Graph"):
    """
    Visualize comprehensive distribution of edge weights
    
    Parameters:
    G: NetworkX graph
    save_path: Path to save the image (optional)
    graph_name: Name for the graph being visualized
    """
    weights = [data.get('weight', 1) for _, _, data in G.edges(data=True)]
    
    if not weights:
        print("No edges found in graph")
        return
    
    # Create subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Edge Weight Analysis - {graph_name}', fontsize=16, fontweight='bold')
    
    # 1. Histogram of all weights
    ax1 = axes[0, 0]
    n, bins, patches = ax1.hist(weights, bins=30, alpha=0.7, edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero line')
    ax1.set_title('Weight Distribution')
    ax1.set_xlabel('Weight')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Color bars based on sign
    for i, p in enumerate(patches):
        if bins[i] < 0:
            p.set_facecolor('lightcoral')
        elif bins[i] > 0:
            p.set_facecolor('lightblue')
        else:
            p.set_facecolor('gray')
    
    # 2. Separate positive and negative weights
    ax2 = axes[0, 1]
    pos_weights = [w for w in weights if w > 0]
    neg_weights = [w for w in weights if w < 0]
    
    if pos_weights:
        ax2.hist(pos_weights, bins=20, alpha=0.7, label=f'Positive ({len(pos_weights)})', 
                color='lightblue', edgecolor='black')
    if neg_weights:
        ax2.hist(neg_weights, bins=20, alpha=0.7, label=f'Negative ({len(neg_weights)})', 
                color='lightcoral', edgecolor='black')
    
    ax2.set_title('Positive vs Negative Weights')
    ax2.set_xlabel('Weight')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Cumulative distribution
    ax3 = axes[1, 0]
    sorted_weights = np.sort(weights)
    cumulative = np.arange(1, len(sorted_weights) + 1) / len(sorted_weights)
    ax3.plot(sorted_weights, cumulative, linewidth=2, color='darkblue')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero line')
    ax3.set_title('Cumulative Distribution')
    ax3.set_xlabel('Weight')
    ax3.set_ylabel('Cumulative Probability')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Box plot
    ax4 = axes[1, 1]
    box_data = []
    labels = []
    
    if pos_weights:
        box_data.append(pos_weights)
        labels.append(f'Positive\n(n={len(pos_weights)})')
    if neg_weights:
        box_data.append(neg_weights)
        labels.append(f'Negative\n(n={len(neg_weights)})')
    
    if box_data:
        bp = ax4.boxplot(box_data, labels=labels, patch_artist=True)
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
            patch.set_facecolor(color)
    
    ax4.set_title('Weight Distribution Summary')
    ax4.set_ylabel('Weight')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    # Print detailed statistics
    print(f"\nWeight Distribution Statistics for {graph_name}:")
    print(f"  Total edges: {len(weights):,}")
    print(f"  Positive edges: {len(pos_weights):,} ({len(pos_weights)/len(weights):.2%})")
    print(f"  Negative edges: {len(neg_weights):,} ({len(neg_weights)/len(weights):.2%})")
    print(f"  Weight range: [{min(weights):.3f}, {max(weights):.3f}]")
    print(f"  Mean: {np.mean(weights):.3f}, Median: {np.median(weights):.3f}")
    print(f"  Std dev: {np.std(weights):.3f}")
    
    if pos_weights:
        print(f"  Positive weights - Mean: {np.mean(pos_weights):.3f}, Std: {np.std(pos_weights):.3f}")
    if neg_weights:
        print(f"  Negative weights - Mean: {np.mean(neg_weights):.3f}, Std: {np.std(neg_weights):.3f}")

def visualize_degree_distribution(G, save_path=None, graph_name="Graph"):
    """
    Visualize degree distributions with comprehensive analysis
    
    Parameters:
    G: NetworkX graph
    save_path: Path to save the image (optional)
    graph_name: Name for the graph being visualized
    """
    if isinstance(G, nx.DiGraph):
        # Directed graph
        in_degrees = [d for n, d in G.in_degree()]
        out_degrees = [d for n, d in G.out_degree()]
        total_degrees = [G.in_degree(n) + G.out_degree(n) for n in G.nodes()]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Degree Distribution Analysis - {graph_name}', fontsize=16, fontweight='bold')
        
        # In-degree distribution
        ax1 = axes[0, 0]
        ax1.hist(in_degrees, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
        ax1.set_title('In-Degree Distribution')
        ax1.set_xlabel('In-Degree')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Out-degree distribution
        ax2 = axes[0, 1]
        ax2.hist(out_degrees, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.set_title('Out-Degree Distribution')
        ax2.set_xlabel('Out-Degree')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # Total degree distribution
        ax3 = axes[1, 0]
        ax3.hist(total_degrees, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.set_title('Total Degree Distribution')
        ax3.set_xlabel('Total Degree (In + Out)')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        # Degree correlation
        ax4 = axes[1, 1]
        ax4.scatter(in_degrees, out_degrees, alpha=0.6, s=20)
        ax4.set_title('In-Degree vs Out-Degree')
        ax4.set_xlabel('In-Degree')
        ax4.set_ylabel('Out-Degree')
        ax4.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        correlation = np.corrcoef(in_degrees, out_degrees)[0, 1]
        ax4.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=ax4.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
        
        print(f"\nDegree Distribution Statistics for {graph_name} (Directed):")
        print(f"  In-degree - Mean: {np.mean(in_degrees):.2f}, Std: {np.std(in_degrees):.2f}, Max: {max(in_degrees)}")
        print(f"  Out-degree - Mean: {np.mean(out_degrees):.2f}, Std: {np.std(out_degrees):.2f}, Max: {max(out_degrees)}")
        print(f"  Total degree - Mean: {np.mean(total_degrees):.2f}, Std: {np.std(total_degrees):.2f}, Max: {max(total_degrees)}")
        print(f"  In-Out degree correlation: {correlation:.3f}")
        
    else:
        # Undirected graph
        degrees = [d for n, d in G.degree()]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Degree Distribution Analysis - {graph_name}', fontsize=16, fontweight='bold')
        
        # Degree distribution
        ax1 = axes[0]
        ax1.hist(degrees, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
        ax1.set_title('Degree Distribution')
        ax1.set_xlabel('Degree')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Log-log plot for power law analysis
        ax2 = axes[1]
        degree_counts = Counter(degrees)
        degrees_sorted = sorted(degree_counts.keys())
        counts = [degree_counts[d] for d in degrees_sorted]
        
        ax2.loglog(degrees_sorted, counts, 'bo-', alpha=0.7)
        ax2.set_title('Degree Distribution (Log-Log)')
        ax2.set_xlabel('Degree (log)')
        ax2.set_ylabel('Frequency (log)')
        ax2.grid(True, alpha=0.3)
        
        print(f"\nDegree Distribution Statistics for {graph_name} (Undirected):")
        print(f"  Mean degree: {np.mean(degrees):.2f}")
        print(f"  Std degree: {np.std(degrees):.2f}")
        print(f"  Max degree: {max(degrees)}")
        print(f"  Min degree: {min(degrees)}")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
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

def visualize_embeddedness(G, save_path=None, graph_name="Graph"):
    """
    Visualize distribution of edge embeddedness with comprehensive analysis
    
    Parameters:
    G: NetworkX graph
    save_path: Path to save the image (optional)
    graph_name: Name for the graph being visualized
    """
    print(f"\nCalculating embeddedness for {graph_name}...")
    start_time = time.time()
    
    edge_embeddedness = calculate_embeddedness(G)
    embeddedness_values = list(edge_embeddedness.values())
    
    calc_time = time.time() - start_time
    print(f"Embeddedness calculation completed in {calc_time:.2f} seconds")
    
    if not embeddedness_values:
        print("No edges found for embeddedness analysis")
        return
    
    # Create comprehensive subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Edge Embeddedness Analysis - {graph_name}', fontsize=16, fontweight='bold')
    
    # 1. Histogram of embeddedness values
    ax1 = axes[0, 0]
    n, bins, patches = ax1.hist(embeddedness_values, bins=min(30, max(embeddedness_values)+1), 
                               alpha=0.7, edgecolor='black', color='skyblue')
    ax1.set_title('Embeddedness Distribution')
    ax1.set_xlabel('Embeddedness (Number of Common Neighbors)')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # 2. Cumulative distribution
    ax2 = axes[0, 1]
    max_embed = max(embeddedness_values)
    cum_dist = []
    x_values = list(range(max_embed + 1))
    
    for i in x_values:
        cum_dist.append(sum(1 for x in embeddedness_values if x <= i) / len(embeddedness_values))
    
    ax2.plot(x_values, cum_dist, marker='o', linestyle='-', linewidth=2, markersize=4)
    ax2.set_title('Cumulative Distribution of Embeddedness')
    ax2.set_xlabel('Embeddedness Threshold')
    ax2.set_ylabel('Proportion of Edges')
    ax2.grid(True, alpha=0.3)
    
    # Add key percentiles
    percentiles = [0.5, 0.8, 0.9, 0.95]
    for p in percentiles:
        threshold = np.percentile(embeddedness_values, p * 100)
        ax2.axhline(y=p, color='red', linestyle='--', alpha=0.5)
        ax2.text(max_embed * 0.7, p + 0.02, f'{p:.0%}: {threshold:.1f}', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    # 3. Box plot and violin plot
    ax3 = axes[1, 0]
    bp = ax3.boxplot(embeddedness_values, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    ax3.set_title('Embeddedness Box Plot')
    ax3.set_ylabel('Embeddedness')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticklabels(['All Edges'])
    
    # 4. Embeddedness vs edge weight (if weights available)
    ax4 = axes[1, 1]
    weights = [data.get('weight', 1) for _, _, data in G.edges(data=True)]
    
    if len(set(weights)) > 1:  # If there are different weights
        # Create scatter plot
        edge_list = list(G.edges())
        edge_weights = [G[u][v].get('weight', 1) for u, v in edge_list]
        edge_embeds = [edge_embeddedness.get((u, v), 0) for u, v in edge_list]
        
        scatter = ax4.scatter(edge_embeds, edge_weights, alpha=0.6, s=20, c=edge_weights, 
                            cmap='RdYlBu', edgecolors='black', linewidth=0.5)
        ax4.set_title('Embeddedness vs Edge Weight')
        ax4.set_xlabel('Embeddedness')
        ax4.set_ylabel('Edge Weight')
        plt.colorbar(scatter, ax=ax4, label='Weight')
        
        # Calculate correlation
        correlation = np.corrcoef(edge_embeds, edge_weights)[0, 1]
        ax4.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=ax4.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    else:
        # Show embeddedness frequency for uniform weights
        embed_counts = Counter(embeddedness_values)
        embeds = sorted(embed_counts.keys())
        counts = [embed_counts[e] for e in embeds]
        
        ax4.bar(embeds, counts, alpha=0.7, color='lightgreen', edgecolor='black')
        ax4.set_title('Embeddedness Frequency')
        ax4.set_xlabel('Embeddedness')
        ax4.set_ylabel('Number of Edges')
    
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    # Print detailed statistics
    print(f"\nEmbeddedness Statistics for {graph_name}:")
    print(f"  Total edges analyzed: {len(embeddedness_values):,}")
    print(f"  Embeddedness range: [{min(embeddedness_values)}, {max(embeddedness_values)}]")
    print(f"  Mean embeddedness: {np.mean(embeddedness_values):.3f}")
    print(f"  Median embeddedness: {np.median(embeddedness_values):.3f}")
    print(f"  Std deviation: {np.std(embeddedness_values):.3f}")
    
    # Show distribution of embeddedness levels
    embed_counts = Counter(embeddedness_values)
    print(f"\nEmbeddedness Level Distribution:")
    for embed in sorted(embed_counts.keys())[:10]:  # Show first 10 levels
        count = embed_counts[embed]
        percentage = count / len(embeddedness_values) * 100
        print(f"    {embed:2d} common neighbors: {count:5,} edges ({percentage:5.1f}%)")
    
    if len(embed_counts) > 10:
        remaining = sum(embed_counts[e] for e in sorted(embed_counts.keys())[10:])
        remaining_pct = remaining / len(embeddedness_values) * 100
        print(f"    >9 common neighbors: {remaining:5,} edges ({remaining_pct:5.1f}%)")
    
    # Calculate zero embeddedness
    zero_embed = embed_counts.get(0, 0)
    zero_pct = zero_embed / len(embeddedness_values) * 100
    print(f"\nEdges with zero embeddedness: {zero_embed:,} ({zero_pct:.1f}%)")
    
    return edge_embeddedness

def analyze_temporal_patterns(G, df, save_path=None):
    """
    Analyze temporal patterns in the Bitcoin trust network
    
    Parameters:
    G: NetworkX graph
    df: Original dataframe with timestamp information
    save_path: Path to save the image (optional)
    """
    print(f"\nAnalyzing temporal patterns...")
    
    # Convert timestamps to datetime
    df_temp = df.copy()
    df_temp['datetime'] = pd.to_datetime(df_temp['time'], unit='s')
    df_temp['year'] = df_temp['datetime'].dt.year
    df_temp['month'] = df_temp['datetime'].dt.to_period('M')
    df_temp['day'] = df_temp['datetime'].dt.date
    
    # Create subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Temporal Analysis of Bitcoin Trust Network', fontsize=16, fontweight='bold')
    
    # 1. Edges over time (daily)
    ax1 = axes[0, 0]
    daily_counts = df_temp.groupby('day').size()
    ax1.plot(daily_counts.index, daily_counts.values, alpha=0.7, linewidth=1)
    ax1.set_title('Daily Edge Creation')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Number of Edges')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 2. Monthly aggregation with sign analysis
    ax2 = axes[0, 1]
    df_temp['is_positive'] = df_temp['rating'] > 0
    monthly_pos = df_temp[df_temp['is_positive']].groupby('month').size()
    monthly_neg = df_temp[~df_temp['is_positive']].groupby('month').size()
    
    # Ensure both series have same index
    all_months = sorted(set(monthly_pos.index) | set(monthly_neg.index))
    monthly_pos = monthly_pos.reindex(all_months, fill_value=0)
    monthly_neg = monthly_neg.reindex(all_months, fill_value=0)
    
    ax2.bar(range(len(all_months)), monthly_pos.values, alpha=0.7, 
           label='Positive', color='lightblue')
    ax2.bar(range(len(all_months)), monthly_neg.values, alpha=0.7, 
           bottom=monthly_pos.values, label='Negative', color='lightcoral')
    ax2.set_title('Monthly Edge Creation by Sign')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Number of Edges')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Set x-axis labels (show every 6th month)
    tick_positions = range(0, len(all_months), max(1, len(all_months)//6))
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels([str(all_months[i]) for i in tick_positions], rotation=45)
    
    # 3. Weight distribution over time
    ax3 = axes[1, 0]
    yearly_weight_means = df_temp.groupby('year')['rating'].mean()
    yearly_weight_stds = df_temp.groupby('year')['rating'].std()
    
    ax3.errorbar(yearly_weight_means.index, yearly_weight_means.values, 
                yerr=yearly_weight_stds.values, marker='o', linestyle='-', 
                capsize=5, capthick=2)
    ax3.set_title('Average Edge Weight by Year')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Average Weight')
    ax3.grid(True, alpha=0.3)
    
    # 4. Cumulative network growth
    ax4 = axes[1, 1]
    df_temp_sorted = df_temp.sort_values('datetime')
    cumulative_edges = range(1, len(df_temp_sorted) + 1)
    
    # Sample data for visualization (every 100th point for large datasets)
    sample_indices = range(0, len(df_temp_sorted), max(1, len(df_temp_sorted)//1000))
    sample_dates = df_temp_sorted.iloc[sample_indices]['datetime']
    sample_cumulative = [cumulative_edges[i] for i in sample_indices]
    
    ax4.plot(sample_dates, sample_cumulative, linewidth=2, color='darkgreen')
    ax4.set_title('Cumulative Network Growth')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Cumulative Number of Edges')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    # Print temporal statistics
    print(f"\nTemporal Statistics:")
    print(f"  Date range: {df_temp['datetime'].min()} to {df_temp['datetime'].max()}")
    print(f"  Total time span: {(df_temp['datetime'].max() - df_temp['datetime'].min()).days} days")
    print(f"  Average edges per day: {len(df_temp) / (df_temp['datetime'].max() - df_temp['datetime'].min()).days:.2f}")
    
    # Show most active periods
    daily_counts = df_temp.groupby('day').size()
    top_days = daily_counts.nlargest(5)
    print(f"\nMost active days:")
    for date, count in top_days.items():
        print(f"    {date}: {count} edges")

def visualize_connectivity_analysis(graphs_dict, save_path=None):
    """
    Compare connectivity properties across different preprocessing steps
    
    Parameters:
    graphs_dict: Dictionary with {name: graph} pairs
    save_path: Path to save the image (optional)
    """
    print(f"\nAnalyzing connectivity across preprocessing steps...")
    
    # Calculate connectivity metrics for each graph
    metrics = {}
    for name, G in graphs_dict.items():
        metrics[name] = {}
        
        if isinstance(G, nx.DiGraph):
            metrics[name]['weakly_connected'] = nx.number_weakly_connected_components(G)
            metrics[name]['strongly_connected'] = nx.number_strongly_connected_components(G)
            
            if metrics[name]['weakly_connected'] > 0:
                largest_wcc = max(nx.weakly_connected_components(G), key=len)
                metrics[name]['largest_wcc_size'] = len(largest_wcc)
                metrics[name]['largest_wcc_ratio'] = len(largest_wcc) / G.number_of_nodes()
            else:
                metrics[name]['largest_wcc_size'] = 0
                metrics[name]['largest_wcc_ratio'] = 0
                
            if metrics[name]['strongly_connected'] > 0:
                largest_scc = max(nx.strongly_connected_components(G), key=len)
                metrics[name]['largest_scc_size'] = len(largest_scc)
                metrics[name]['largest_scc_ratio'] = len(largest_scc) / G.number_of_nodes()
            else:
                metrics[name]['largest_scc_size'] = 0
                metrics[name]['largest_scc_ratio'] = 0
        else:
            metrics[name]['connected_components'] = nx.number_connected_components(G)
            if metrics[name]['connected_components'] > 0:
                largest_cc = max(nx.connected_components(G), key=len)
                metrics[name]['largest_cc_size'] = len(largest_cc)
                metrics[name]['largest_cc_ratio'] = len(largest_cc) / G.number_of_nodes()
            else:
                metrics[name]['largest_cc_size'] = 0
                metrics[name]['largest_cc_ratio'] = 0
        
        metrics[name]['nodes'] = G.number_of_nodes()
        metrics[name]['edges'] = G.number_of_edges()
        metrics[name]['density'] = nx.density(G)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Connectivity Analysis Across Preprocessing Steps', fontsize=16, fontweight='bold')
    
    graph_names = list(graphs_dict.keys())
    
    # 1. Nodes and edges comparison
    ax1 = axes[0, 0]
    nodes = [metrics[name]['nodes'] for name in graph_names]
    edges = [metrics[name]['edges'] for name in graph_names]
    
    x = np.arange(len(graph_names))
    width = 0.35
    
    ax1.bar(x - width/2, nodes, width, label='Nodes', alpha=0.8, color='lightblue')
    ax1.bar(x + width/2, edges, width, label='Edges', alpha=0.8, color='lightcoral')
    ax1.set_title('Nodes and Edges Comparison')
    ax1.set_xlabel('Preprocessing Step')
    ax1.set_ylabel('Count')
    ax1.set_xticks(x)
    ax1.set_xticklabels(graph_names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Density comparison
    ax2 = axes[0, 1]
    densities = [metrics[name]['density'] for name in graph_names]
    bars = ax2.bar(graph_names, densities, alpha=0.8, color='lightgreen')
    ax2.set_title('Graph Density Comparison')
    ax2.set_xlabel('Preprocessing Step')
    ax2.set_ylabel('Density')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, density in zip(bars, densities):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{density:.6f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Connected components
    ax3 = axes[1, 0]
    # Check if graphs are directed or undirected
    sample_graph = list(graphs_dict.values())[0]
    if isinstance(sample_graph, nx.DiGraph):
        wcc_counts = [metrics[name]['weakly_connected'] for name in graph_names]
        scc_counts = [metrics[name]['strongly_connected'] for name in graph_names]
        
        x = np.arange(len(graph_names))
        ax3.bar(x - width/2, wcc_counts, width, label='Weakly Connected', alpha=0.8, color='orange')
        ax3.bar(x + width/2, scc_counts, width, label='Strongly Connected', alpha=0.8, color='purple')
        ax3.set_title('Connected Components (Directed)')
    else:
        cc_counts = [metrics[name]['connected_components'] for name in graph_names]
        ax3.bar(graph_names, cc_counts, alpha=0.8, color='skyblue')
        ax3.set_title('Connected Components (Undirected)')
    
    ax3.set_xlabel('Preprocessing Step')
    ax3.set_ylabel('Number of Components')
    ax3.set_xticks(x)
    ax3.set_xticklabels(graph_names, rotation=45)
    if isinstance(sample_graph, nx.DiGraph):
        ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Largest component size ratio
    ax4 = axes[1, 1]
    if isinstance(sample_graph, nx.DiGraph):
        largest_wcc_ratios = [metrics[name]['largest_wcc_ratio'] for name in graph_names]
        largest_scc_ratios = [metrics[name]['largest_scc_ratio'] for name in graph_names]
        
        x = np.arange(len(graph_names))
        ax4.bar(x - width/2, largest_wcc_ratios, width, label='Largest WCC', alpha=0.8, color='orange')
        ax4.bar(x + width/2, largest_scc_ratios, width, label='Largest SCC', alpha=0.8, color='purple')
        ax4.set_title('Largest Component Size Ratio')
        ax4.legend()
    else:
        largest_cc_ratios = [metrics[name]['largest_cc_ratio'] for name in graph_names]
        ax4.bar(graph_names, largest_cc_ratios, alpha=0.8, color='skyblue')
        ax4.set_title('Largest Component Size Ratio')
    
    ax4.set_xlabel('Preprocessing Step')
    ax4.set_ylabel('Ratio of Nodes in Largest Component')
    ax4.set_xticks(x)
    ax4.set_xticklabels(graph_names, rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    # Print connectivity summary
    print(f"\nConnectivity Summary:")
    for name in graph_names:
        print(f"\n{name}:")
        print(f"  Nodes: {metrics[name]['nodes']:,}")
        print(f"  Edges: {metrics[name]['edges']:,}")
        print(f"  Density: {metrics[name]['density']:.6f}")
        if isinstance(sample_graph, nx.DiGraph):
            print(f"  Weakly connected components: {metrics[name]['weakly_connected']}")
            print(f"  Strongly connected components: {metrics[name]['strongly_connected']}")
            print(f"  Largest WCC ratio: {metrics[name]['largest_wcc_ratio']:.3f}")
            print(f"  Largest SCC ratio: {metrics[name]['largest_scc_ratio']:.3f}")
        else:
            print(f"  Connected components: {metrics[name]['connected_components']}")
            print(f"  Largest CC ratio: {metrics[name]['largest_cc_ratio']:.3f}")

def visualize_preprocessing_pipeline(save_path=None):
    """
    Visualize the complete preprocessing pipeline with before/after comparisons
    
    Parameters:
    save_path: Path to save the main summary image (optional)
    """
    print(f"\n{'='*80}")
    print("COMPREHENSIVE PREPROCESSING PIPELINE ANALYSIS")
    print(f"{'='*80}")
    
    # Load original data
    data_path = os.path.join('..', 'data', 'soc-sign-bitcoinotc.csv')
    G_original, df = load_bitcoin_data(data_path)
    
    # Store all preprocessing steps
    graphs = {
        '1. Original': G_original
    }
    
    # Step 2: Filter neutral edges
    print("\nStep 2: Filtering neutral edges...")
    G_filtered = filter_neutral_edges(G_original)
    graphs['2. Filtered'] = G_filtered
    
    # Step 3: Map to signed graph
    print("Step 3: Converting to signed graph...")
    G_signed = map_to_unweighted_graph(G_filtered)
    graphs['3. Signed'] = G_signed
    
    # Step 4: Ensure connectivity
    print("Step 4: Ensuring connectivity...")
    G_connected = ensure_connectivity(G_signed)
    graphs['4. Connected'] = G_connected
    
    # Step 5: Reindex nodes
    print("Step 5: Reindexing nodes...")
    G_final = reindex_nodes(G_connected)
    graphs['5. Final'] = G_final
    
    # Create results directory
    results_dir = os.path.join('..', 'results', 'preprocessing_analysis')
    os.makedirs(results_dir, exist_ok=True)
    
    # Analyze each step
    stats_summary = {}
    for step_name, graph in graphs.items():
        print(f"\n{'-'*60}")
        stats = analyze_network(graph, step_name)
        stats_summary[step_name] = stats
        
        # Create individual visualizations for key steps
        if step_name in ['1. Original', '3. Signed', '5. Final']:
            step_dir = os.path.join(results_dir, step_name.replace('. ', '_').lower())
            os.makedirs(step_dir, exist_ok=True)
            
            # Weight distribution
            visualize_weight_distribution(graph, 
                                        os.path.join(step_dir, 'weight_distribution.png'),
                                        step_name)
            
            # Degree distribution
            visualize_degree_distribution(graph, 
                                        os.path.join(step_dir, 'degree_distribution.png'),
                                        step_name)
            
            # Embeddedness (for final graph only to save time)
            if step_name == '5. Final':
                visualize_embeddedness(graph, 
                                     os.path.join(step_dir, 'embeddedness_distribution.png'),
                                     step_name)
    
    # Connectivity analysis across all steps
    visualize_connectivity_analysis(graphs, 
                                  os.path.join(results_dir, 'connectivity_comparison.png'))
    
    # Temporal analysis on original data
    analyze_temporal_patterns(G_original, df, 
                            os.path.join(results_dir, 'temporal_analysis.png'))
    
    # Create summary comparison table
    create_preprocessing_summary_table(stats_summary, 
                                     os.path.join(results_dir, 'preprocessing_summary.png'))
    
    print(f"\n{'='*80}")
    print("PREPROCESSING PIPELINE ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"All results saved to: {results_dir}")
    
    return graphs, stats_summary

def create_preprocessing_summary_table(stats_summary, save_path=None):
    """
    Create a summary table comparing all preprocessing steps
    
    Parameters:
    stats_summary: Dictionary of statistics for each preprocessing step
    save_path: Path to save the summary table image
    """
    # Prepare data for table
    steps = list(stats_summary.keys())
    
    # Select key metrics to display
    metrics = [
        ('Nodes', 'num_nodes'),
        ('Edges', 'num_edges'),
        ('Density', 'graph_density'),
        ('Positive Ratio', 'positive_ratio'),
        ('Avg In-Degree', 'avg_in_degree'),
        ('Avg Out-Degree', 'avg_out_degree'),
        ('WCC Count', 'weakly_connected_components'),
        ('Largest WCC %', 'largest_wcc_ratio')
    ]
    
    # Create table data
    table_data = []
    for metric_name, metric_key in metrics:
        row = [metric_name]
        for step in steps:
            value = stats_summary[step].get(metric_key, 'N/A')
            if isinstance(value, float):
                if metric_key in ['graph_density', 'positive_ratio', 'largest_wcc_ratio']:
                    row.append(f"{value:.3f}")
                else:
                    row.append(f"{value:.2f}")
            elif isinstance(value, int):
                row.append(f"{value:,}")
            else:
                row.append(str(value))
        table_data.append(row)
    
    # Create figure for table
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    headers = ['Metric'] + steps
    table = ax.table(cellText=table_data, colLabels=headers, 
                    cellLoc='center', loc='center',
                    colWidths=[0.15] + [0.17] * len(steps))
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Color header row
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color metric column
    for i in range(1, len(table_data) + 1):
        table[(i, 0)].set_facecolor('#f1f1f2')
        table[(i, 0)].set_text_props(weight='bold')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(1, len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f9f9f9')
    
    plt.title('Preprocessing Pipeline Summary', fontsize=16, fontweight='bold', pad=20)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Summary table saved to {save_path}")
    
    plt.show()

plt.show()

if __name__ == "__main__":
    """
    Main execution: Run the complete preprocessing pipeline analysis
    """
    print("Starting comprehensive Bitcoin Trust Network analysis...")
    print("This will analyze the complete preprocessing pipeline and generate visualizations")
    
    try:
        # Run the complete preprocessing pipeline analysis
        graphs, stats_summary = visualize_preprocessing_pipeline()
        
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE!")
        print(f"{'='*80}")
        print("\nKey findings:")
        print(f"• Original network: {stats_summary['1. Original']['num_nodes']:,} nodes, {stats_summary['1. Original']['num_edges']:,} edges")
        print(f"• Final network: {stats_summary['5. Final']['num_nodes']:,} nodes, {stats_summary['5. Final']['num_edges']:,} edges")
        print(f"• Positive edge ratio: {stats_summary['5. Final'].get('positive_ratio', 0):.1%}")
        print(f"• Network density: {stats_summary['5. Final']['graph_density']:.6f}")
        
        # Show largest component info
        if 'largest_wcc_ratio' in stats_summary['5. Final']:
            print(f"• Largest component: {stats_summary['5. Final']['largest_wcc_ratio']:.1%} of nodes")
        
        print("\nAll analysis results and visualizations have been saved to '../results/preprocessing_analysis/'")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        print("Please check that the data file exists and the required dependencies are installed.")
        print("\nRequired dependencies:")
        print("- networkx")
        print("- matplotlib")
        print("- seaborn") 
        print("- pandas")
        print("- numpy")
        
        # Still try to run a basic analysis if possible
        try:
            print("\nAttempting basic analysis...")
            data_path = os.path.join('..', 'data', 'soc-sign-bitcoinotc.csv')
            if os.path.exists(data_path):
                G_original, df = load_bitcoin_data(data_path)
                basic_stats = analyze_network(G_original, "Original Network")
                print("Basic analysis completed successfully.")
            else:
                print(f"Data file not found at: {data_path}")
        except Exception as basic_error:
            print(f"Basic analysis also failed: {str(basic_error)}")