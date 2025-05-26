import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict, Counter

# Add project root to sys.path for src imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_bitcoin_data

def analyze_bidirectional_edges(G):
    """
    Analyze bidirectional edges in the directed graph.
    
    Parameters:
    G: NetworkX directed graph
    
    Returns:
    analysis: Dictionary containing bidirectional edge analysis
    """
    print("=== Bidirectional Edge Analysis ===")
    
    # Find all bidirectional edge pairs
    bidirectional_pairs = []
    edge_dict = {}
    
    # Create a dictionary of all edges with their weights
    for u, v, data in G.edges(data=True):
        edge_dict[(u, v)] = data['weight']
    
    # Check for bidirectional edges
    for (u, v), weight_uv in edge_dict.items():
        if (v, u) in edge_dict:  # Found a bidirectional pair
            weight_vu = edge_dict[(v, u)]
            if u < v:  # Avoid counting the same pair twice
                bidirectional_pairs.append({
                    'node_1': u,
                    'node_2': v,
                    'weight_12': weight_uv,
                    'weight_21': weight_vu,
                    'same_sign': (weight_uv > 0) == (weight_vu > 0),
                    'weight_diff': abs(weight_uv - weight_vu),
                    'weight_sum': weight_uv + weight_vu,
                    'weight_avg': (weight_uv + weight_vu) / 2
                })
    
    # Basic statistics
    total_edges = G.number_of_edges()
    total_bidirectional_edges = len(bidirectional_pairs) * 2  # Each pair counts as 2 edges
    bidirectional_ratio = total_bidirectional_edges / total_edges if total_edges > 0 else 0
    
    # Sign consistency analysis
    same_sign_count = sum(1 for pair in bidirectional_pairs if pair['same_sign'])
    sign_consistency_rate = same_sign_count / len(bidirectional_pairs) if bidirectional_pairs else 0
    
    # Weight difference analysis
    weight_diffs = [pair['weight_diff'] for pair in bidirectional_pairs]
    avg_weight_diff = np.mean(weight_diffs) if weight_diffs else 0
    max_weight_diff = max(weight_diffs) if weight_diffs else 0
    
    # Weight correlation analysis
    weights_12 = [pair['weight_12'] for pair in bidirectional_pairs]
    weights_21 = [pair['weight_21'] for pair in bidirectional_pairs]
    weight_correlation = np.corrcoef(weights_12, weights_21)[0, 1] if len(weights_12) > 1 else 0
    
    # Sign pattern analysis
    sign_patterns = Counter()
    for pair in bidirectional_pairs:
        w1, w2 = pair['weight_12'], pair['weight_21']
        if w1 > 0 and w2 > 0:
            sign_patterns['pos_pos'] += 1
        elif w1 > 0 and w2 < 0:
            sign_patterns['pos_neg'] += 1
        elif w1 < 0 and w2 > 0:
            sign_patterns['neg_pos'] += 1
        else:  # w1 < 0 and w2 < 0
            sign_patterns['neg_neg'] += 1
    
    analysis = {
        'total_edges': total_edges,
        'total_bidirectional_pairs': len(bidirectional_pairs),
        'total_bidirectional_edges': total_bidirectional_edges,
        'bidirectional_ratio': bidirectional_ratio,
        'sign_consistency_rate': sign_consistency_rate,
        'same_sign_pairs': same_sign_count,
        'different_sign_pairs': len(bidirectional_pairs) - same_sign_count,
        'avg_weight_difference': avg_weight_diff,
        'max_weight_difference': max_weight_diff,
        'weight_correlation': weight_correlation,
        'sign_patterns': dict(sign_patterns),
        'bidirectional_pairs': bidirectional_pairs
    }
    
    return analysis

def print_analysis_results(analysis):
    """Print the analysis results in a readable format."""
    print(f"Total edges in graph: {analysis['total_edges']:,}")
    print(f"Bidirectional pairs found: {analysis['total_bidirectional_pairs']:,}")
    print(f"Total bidirectional edges: {analysis['total_bidirectional_edges']:,}")
    print(f"Bidirectional edge ratio: {analysis['bidirectional_ratio']:.2%}")
    print()
    
    print("=== Sign Consistency Analysis ===")
    print(f"Pairs with same sign: {analysis['same_sign_pairs']:,}")
    print(f"Pairs with different signs: {analysis['different_sign_pairs']:,}")
    print(f"Sign consistency rate: {analysis['sign_consistency_rate']:.2%}")
    print()
    
    print("=== Weight Analysis ===")
    print(f"Average weight difference: {analysis['avg_weight_difference']:.3f}")
    print(f"Maximum weight difference: {analysis['max_weight_difference']:.3f}")
    print(f"Weight correlation: {analysis['weight_correlation']:.3f}")
    print()
    
    print("=== Sign Pattern Distribution ===")
    total_pairs = analysis['total_bidirectional_pairs']
    for pattern, count in analysis['sign_patterns'].items():
        percentage = (count / total_pairs * 100) if total_pairs > 0 else 0
        print(f"{pattern}: {count:,} pairs ({percentage:.1f}%)")

def create_visualizations(analysis, save_dir):
    """Create and save visualizations of the bidirectional edge analysis."""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Sign pattern distribution pie chart
    if analysis['sign_patterns']:
        plt.figure(figsize=(10, 8))
        labels = []
        sizes = []
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']
        
        pattern_labels = {
            'pos_pos': 'Positive ↔ Positive',
            'pos_neg': 'Positive ↔ Negative', 
            'neg_pos': 'Negative ↔ Positive',
            'neg_neg': 'Negative ↔ Negative'
        }
        
        for pattern, count in analysis['sign_patterns'].items():
            if count > 0:
                labels.append(f"{pattern_labels[pattern]}\n({count:,} pairs)")
                sizes.append(count)
        
        plt.pie(sizes, labels=labels, colors=colors[:len(sizes)], autopct='%1.1f%%', startangle=90)
        plt.title('Bidirectional Edge Sign Patterns', fontsize=14, fontweight='bold')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'sign_pattern_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Weight difference histogram
    if analysis['bidirectional_pairs']:
        weight_diffs = [pair['weight_diff'] for pair in analysis['bidirectional_pairs']]
        
        plt.figure(figsize=(10, 6))
        plt.hist(weight_diffs, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Weight Difference |w1 - w2|')
        plt.ylabel('Number of Bidirectional Pairs')
        plt.title('Distribution of Weight Differences in Bidirectional Edges')
        plt.axvline(np.mean(weight_diffs), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(weight_diffs):.2f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'weight_difference_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Weight correlation scatter plot
    if len(analysis['bidirectional_pairs']) > 1:
        weights_12 = [pair['weight_12'] for pair in analysis['bidirectional_pairs']]
        weights_21 = [pair['weight_21'] for pair in analysis['bidirectional_pairs']]
        
        plt.figure(figsize=(10, 8))
        plt.scatter(weights_12, weights_21, alpha=0.6, s=50)
        plt.xlabel('Weight (Node 1 → Node 2)')
        plt.ylabel('Weight (Node 2 → Node 1)')
        plt.title(f'Weight Correlation in Bidirectional Edges\n(Correlation: {analysis["weight_correlation"]:.3f})')
        
        # Add diagonal lines for reference
        min_weight = min(min(weights_12), min(weights_21))
        max_weight = max(max(weights_12), max(weights_21))
        plt.plot([min_weight, max_weight], [min_weight, max_weight], 'r--', alpha=0.8, label='Perfect correlation')
        plt.plot([min_weight, max_weight], [-min_weight, -max_weight], 'g--', alpha=0.8, label='Perfect anti-correlation')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'weight_correlation_scatter.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Summary statistics bar chart
    plt.figure(figsize=(12, 6))
    
    categories = ['Total Edges', 'Bidirectional Edges', 'Same Sign Pairs', 'Different Sign Pairs']
    values = [
        analysis['total_edges'],
        analysis['total_bidirectional_edges'], 
        analysis['same_sign_pairs'],
        analysis['different_sign_pairs']
    ]
    colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']
    
    bars = plt.bar(categories, values, color=colors)
    plt.ylabel('Count')
    plt.title('Bidirectional Edge Analysis Summary')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                f'{value:,}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'summary_statistics.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_detailed_results(analysis, save_dir):
    """Save detailed analysis results to CSV files."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save summary statistics
    summary_data = {
        'metric': [
            'total_edges', 'total_bidirectional_pairs', 'total_bidirectional_edges',
            'bidirectional_ratio', 'sign_consistency_rate', 'avg_weight_difference',
            'max_weight_difference', 'weight_correlation'
        ],
        'value': [
            analysis['total_edges'], analysis['total_bidirectional_pairs'], 
            analysis['total_bidirectional_edges'], analysis['bidirectional_ratio'],
            analysis['sign_consistency_rate'], analysis['avg_weight_difference'],
            analysis['max_weight_difference'], analysis['weight_correlation']
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(save_dir, 'bidirectional_summary.csv'), index=False)
    
    # Save bidirectional pairs details
    if analysis['bidirectional_pairs']:
        pairs_df = pd.DataFrame(analysis['bidirectional_pairs'])
        pairs_df.to_csv(os.path.join(save_dir, 'bidirectional_pairs_details.csv'), index=False)
    
    # Save sign pattern statistics
    pattern_data = []
    total_pairs = analysis['total_bidirectional_pairs']
    for pattern, count in analysis['sign_patterns'].items():
        percentage = (count / total_pairs * 100) if total_pairs > 0 else 0
        pattern_data.append({
            'sign_pattern': pattern,
            'count': count,
            'percentage': percentage
        })
    
    pattern_df = pd.DataFrame(pattern_data)
    pattern_df.to_csv(os.path.join(save_dir, 'sign_patterns.csv'), index=False)

def main():
    """Main function to run the bidirectional edge analysis."""
    # Load the Bitcoin OTC dataset
    data_path = os.path.join(PROJECT_ROOT, 'data', 'soc-sign-bitcoinotc.csv')
    print(f"Loading data from {data_path}...")
    
    G, df = load_bitcoin_data(data_path)
    print(f"Loaded graph with {G.number_of_nodes():,} nodes and {G.number_of_edges():,} edges")
    print()
    
    # Run bidirectional edge analysis
    analysis = analyze_bidirectional_edges(G)
    
    # Print results
    print_analysis_results(analysis)
    
    # Create output directory
    results_dir = os.path.join(PROJECT_ROOT, 'results', 'bidirectional_analysis')
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(analysis, results_dir)
    
    # Save detailed results
    print("Saving detailed results...")
    save_detailed_results(analysis, results_dir)
    
    print(f"\n✓ Analysis complete! Results saved to {results_dir}")
    print("\nKey findings:")
    print(f"- {analysis['bidirectional_ratio']:.1%} of edges are bidirectional")
    print(f"- {analysis['sign_consistency_rate']:.1%} of bidirectional pairs have consistent signs")
    print(f"- Weight correlation in bidirectional pairs: {analysis['weight_correlation']:.3f}")
    
    # Recommendations based on findings
    print("\nRecommendations for handling bidirectional edges:")
    if analysis['bidirectional_ratio'] > 0.1:  # More than 10%
        print("- Bidirectional edges are significant and should be handled explicitly")
        if analysis['sign_consistency_rate'] > 0.8:  # More than 80% consistent
            print("- High sign consistency suggests averaging or sum methods might work well")
        else:
            print("- Low sign consistency suggests careful handling needed (e.g., keep stronger signal)")
    else:
        print("- Bidirectional edges are rare, simple handling methods should suffice")

if __name__ == "__main__":
    main()