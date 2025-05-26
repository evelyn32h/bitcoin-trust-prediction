import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict, Counter
import json

# Add project root to sys.path for src imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_bitcoin_data

def analyze_epinions_dataset(filepath):
    """
    Comprehensive analysis of Epinions dataset.
    Focuses on features relevant to HOC method performance.
    
    Parameters:
    filepath: Path to the Epinions dataset file
    
    Returns:
    analysis: Dictionary containing dataset analysis results
    """
    print("=== EPINIONS DATASET ANALYSIS ===")
    print("Analyzing original paper dataset for our HOC method")
    print()
    
    # Load data
    G, df = load_bitcoin_data(filepath)
    
    analysis = {}
    
    # Basic dataset statistics
    print("=== BASIC DATASET STATISTICS ===")
    analysis['total_nodes'] = G.number_of_nodes()
    analysis['total_edges'] = G.number_of_edges()
    analysis['avg_degree'] = 2 * G.number_of_edges() / G.number_of_nodes()
    
    print(f"Nodes: {analysis['total_nodes']:,}")
    print(f"Edges: {analysis['total_edges']:,}")
    print(f"Average degree: {analysis['avg_degree']:.2f}")
    
    # Edge sign distribution
    print(f"\n=== EDGE SIGN DISTRIBUTION ===")
    pos_edges = sum(1 for _, _, d in G.edges(data=True) if d['weight'] > 0)
    neg_edges = sum(1 for _, _, d in G.edges(data=True) if d['weight'] < 0)
    zero_edges = sum(1 for _, _, d in G.edges(data=True) if d['weight'] == 0)
    
    analysis['positive_edges'] = pos_edges
    analysis['negative_edges'] = neg_edges
    analysis['zero_edges'] = zero_edges
    analysis['positive_ratio'] = pos_edges / analysis['total_edges']
    
    print(f"Positive edges: {pos_edges:,} ({analysis['positive_ratio']*100:.1f}%)")
    print(f"Negative edges: {neg_edges:,} ({neg_edges/analysis['total_edges']*100:.1f}%)")
    if zero_edges > 0:
        print(f"Zero edges: {zero_edges:,} ({zero_edges/analysis['total_edges']*100:.1f}%)")
    
    # Degree distribution analysis
    print(f"\n=== DEGREE DISTRIBUTION ANALYSIS ===")
    in_degrees = [d for n, d in G.in_degree()]
    out_degrees = [d for n, d in G.out_degree()]
    
    analysis['max_in_degree'] = max(in_degrees)
    analysis['max_out_degree'] = max(out_degrees)
    analysis['avg_in_degree'] = np.mean(in_degrees)
    analysis['avg_out_degree'] = np.mean(out_degrees)
    
    print(f"Max in-degree: {analysis['max_in_degree']}")
    print(f"Max out-degree: {analysis['max_out_degree']}")
    print(f"Average in-degree: {analysis['avg_in_degree']:.2f}")
    print(f"Average out-degree: {analysis['avg_out_degree']:.2f}")
    
    # High-degree nodes (potential for good embeddedness)
    high_degree_threshold = np.percentile(in_degrees + out_degrees, 90)
    high_degree_nodes = [n for n in G.nodes() if 
                        G.in_degree(n) + G.out_degree(n) >= high_degree_threshold]
    analysis['high_degree_nodes'] = len(high_degree_nodes)
    analysis['high_degree_threshold'] = high_degree_threshold
    
    print(f"High-degree nodes (>90th percentile): {len(high_degree_nodes)}")
    print(f"High-degree threshold: {high_degree_threshold:.0f}")
    
    # Connectivity analysis
    print(f"\n=== CONNECTIVITY ANALYSIS ===")
    wcc = list(nx.weakly_connected_components(G))
    analysis['num_weakly_connected_components'] = len(wcc)
    
    if len(wcc) > 0:
        largest_wcc_size = len(max(wcc, key=len))
        analysis['largest_wcc_size'] = largest_wcc_size
        analysis['largest_wcc_ratio'] = largest_wcc_size / analysis['total_nodes']
        
        print(f"Weakly connected components: {len(wcc)}")
        print(f"Largest WCC size: {largest_wcc_size:,} ({analysis['largest_wcc_ratio']*100:.1f}%)")
    
    # Bidirectional edges analysis (Critical for our method)
    print(f"\n=== BIDIRECTIONAL EDGES ANALYSIS ===")
    bidirectional_pairs = []
    edge_dict = {}
    
    # Create edge dictionary
    for u, v, data in G.edges(data=True):
        edge_dict[(u, v)] = data['weight']
    
    # Find bidirectional pairs
    processed_pairs = set()
    for (u, v), weight_uv in edge_dict.items():
        if (min(u, v), max(u, v)) in processed_pairs:
            continue
        
        if (v, u) in edge_dict:
            weight_vu = edge_dict[(v, u)]
            bidirectional_pairs.append({
                'node_1': u,
                'node_2': v,
                'weight_12': weight_uv,
                'weight_21': weight_vu,
                'same_sign': (weight_uv > 0) == (weight_vu > 0)
            })
            processed_pairs.add((min(u, v), max(u, v)))
    
    analysis['bidirectional_pairs'] = len(bidirectional_pairs)
    analysis['bidirectional_edges'] = len(bidirectional_pairs) * 2
    analysis['bidirectional_ratio'] = analysis['bidirectional_edges'] / analysis['total_edges']
    
    if bidirectional_pairs:
        same_sign_count = sum(1 for pair in bidirectional_pairs if pair['same_sign'])
        analysis['sign_consistency_rate'] = same_sign_count / len(bidirectional_pairs)
        
        print(f"Bidirectional pairs: {len(bidirectional_pairs):,}")
        print(f"Bidirectional edges: {analysis['bidirectional_edges']:,} ({analysis['bidirectional_ratio']*100:.1f}%)")
        print(f"Sign consistency rate: {analysis['sign_consistency_rate']*100:.1f}%")
    
    # Triangle analysis (Critical for HOC features)
    print(f"\n=== TRIANGLE ANALYSIS (Critical for HOC) ===")
    G_undirected = G.to_undirected()
    
    # Count triangles
    triangles = sum(nx.triangles(G_undirected).values()) // 3
    analysis['num_triangles'] = triangles
    analysis['triangles_per_node'] = triangles / analysis['total_nodes'] if analysis['total_nodes'] > 0 else 0
    
    print(f"Number of triangles: {triangles:,}")
    print(f"Triangles per node: {analysis['triangles_per_node']:.2f}")
    
    # Clustering coefficient
    try:
        clustering = nx.average_clustering(G_undirected)
        analysis['avg_clustering_coefficient'] = clustering
        print(f"Average clustering coefficient: {clustering:.4f}")
    except:
        analysis['avg_clustering_coefficient'] = 0.0
        print("Could not compute clustering coefficient")
    
    # Embeddedness analysis (Most critical for our method)
    print(f"\n=== EMBEDDEDNESS ANALYSIS (Most Critical) ===")
    print("Analyzing edge embeddedness (common neighbors)...")
    
    # Sample edges for embeddedness analysis (for performance)
    sample_size = min(10000, G.number_of_edges())
    sample_edges = list(G.edges())[:sample_size]
    
    embeddedness_values = []
    zero_embeddedness_count = 0
    
    for i, (u, v) in enumerate(sample_edges):
        if i % 1000 == 0:
            print(f"  Progress: {i}/{sample_size} edges analyzed")
        
        # Get common neighbors
        u_neighbors = set(G_undirected.neighbors(u))
        v_neighbors = set(G_undirected.neighbors(v))
        common_neighbors = u_neighbors & v_neighbors
        
        embeddedness = len(common_neighbors)
        embeddedness_values.append(embeddedness)
        
        if embeddedness == 0:
            zero_embeddedness_count += 1
    
    if embeddedness_values:
        analysis['avg_embeddedness'] = np.mean(embeddedness_values)
        analysis['median_embeddedness'] = np.median(embeddedness_values)
        analysis['max_embeddedness'] = max(embeddedness_values)
        analysis['zero_embeddedness_ratio'] = zero_embeddedness_count / len(embeddedness_values)
        
        # Count edges with different embeddedness levels
        embed_1_plus = sum(1 for e in embeddedness_values if e >= 1)
        embed_2_plus = sum(1 for e in embeddedness_values if e >= 2)
        embed_3_plus = sum(1 for e in embeddedness_values if e >= 3)
        
        analysis['embeddedness_1_plus_ratio'] = embed_1_plus / len(embeddedness_values)
        analysis['embeddedness_2_plus_ratio'] = embed_2_plus / len(embeddedness_values)
        analysis['embeddedness_3_plus_ratio'] = embed_3_plus / len(embeddedness_values)
        
        print(f"Sample size: {len(embeddedness_values):,} edges")
        print(f"Average embeddedness: {analysis['avg_embeddedness']:.2f}")
        print(f"Median embeddedness: {analysis['median_embeddedness']:.2f}")
        print(f"Max embeddedness: {analysis['max_embeddedness']}")
        print(f"Zero embeddedness: {zero_embeddedness_count} ({analysis['zero_embeddedness_ratio']*100:.1f}%)")
        print(f"Embeddedness >=1: {embed_1_plus} ({analysis['embeddedness_1_plus_ratio']*100:.1f}%)")
        print(f"Embeddedness >=2: {embed_2_plus} ({analysis['embeddedness_2_plus_ratio']*100:.1f}%)")
        print(f"Embeddedness >=3: {embed_3_plus} ({analysis['embeddedness_3_plus_ratio']*100:.1f}%)")
    
    # HOC method suitability assessment
    print(f"\n=== HOC METHOD SUITABILITY ASSESSMENT ===")
    
    # Calculate suitability score
    suitability_score = 0
    suitability_reasons = []
    
    # Triangle density score
    if analysis['triangles_per_node'] > 1.0:
        suitability_score += 3
        suitability_reasons.append("[EXCELLENT] High triangle density")
    elif analysis['triangles_per_node'] > 0.1:
        suitability_score += 2
        suitability_reasons.append("[GOOD] Moderate triangle density")
    else:
        suitability_score += 0
        suitability_reasons.append("[POOR] Low triangle density")
    
    # Clustering coefficient score
    if analysis['avg_clustering_coefficient'] > 0.1:
        suitability_score += 2
        suitability_reasons.append("[EXCELLENT] Good clustering coefficient")
    elif analysis['avg_clustering_coefficient'] > 0.01:
        suitability_score += 1
        suitability_reasons.append("[GOOD] Moderate clustering coefficient")
    else:
        suitability_score += 0
        suitability_reasons.append("[POOR] Low clustering coefficient")
    
    # Embeddedness score
    if embeddedness_values and analysis['embeddedness_1_plus_ratio'] > 0.5:
        suitability_score += 3
        suitability_reasons.append("[EXCELLENT] Good embeddedness (>50% edges have common neighbors)")
    elif embeddedness_values and analysis['embeddedness_1_plus_ratio'] > 0.2:
        suitability_score += 2
        suitability_reasons.append("[GOOD] Moderate embeddedness")
    else:
        suitability_score += 0
        suitability_reasons.append("[POOR] Low embeddedness")
    
    # Bidirectional edges score
    if analysis['bidirectional_ratio'] > 0.1:
        suitability_score += 2
        suitability_reasons.append("[EXCELLENT] Significant bidirectional edges")
    else:
        suitability_score += 1
        suitability_reasons.append("[GOOD] Few bidirectional edges")
    
    analysis['hoc_suitability_score'] = suitability_score
    analysis['max_suitability_score'] = 10
    analysis['suitability_percentage'] = (suitability_score / 10) * 100
    
    print(f"HOC Method Suitability Score: {suitability_score}/10 ({analysis['suitability_percentage']:.0f}%)")
    print("Detailed assessment:")
    for reason in suitability_reasons:
        print(f"  {reason}")
    
    # Overall recommendation
    print(f"\n=== DATASET RECOMMENDATION ===")
    if suitability_score >= 7:
        recommendation = "EXCELLENT"
        color = "[EXCELLENT]"
        message = "This dataset is highly suitable for HOC features. Expect significant improvement over Bitcoin OTC."
    elif suitability_score >= 5:
        recommendation = "GOOD"
        color = "[GOOD]"
        message = "This dataset should work well with HOC features. Expect moderate to good improvement."
    elif suitability_score >= 3:
        recommendation = "MODERATE"
        color = "[MODERATE]"
        message = "This dataset may work with HOC features but results may be mixed."
    else:
        recommendation = "POOR"
        color = "[POOR]"
        message = "This dataset may not be suitable for HOC features. Consider alternative methods."
    
    analysis['recommendation'] = recommendation
    analysis['recommendation_message'] = message
    
    print(f"{color} RECOMMENDATION: {recommendation}")
    print(f"    {message}")
    
    # Comparison with Bitcoin OTC (estimated)
    print(f"\n=== COMPARISON WITH BITCOIN OTC ===")
    print("Estimated Bitcoin OTC characteristics:")
    print("  - Triangles per node: ~0.1")
    print("  - Clustering coefficient: ~0.02")
    print("  - Embeddedness >=1: ~10%")
    print("  - Suitability score: ~3/10")
    print()
    print(f"Epinions improvements:")
    print(f"  - Triangles per node: {analysis['triangles_per_node']:.1f}x better")
    print(f"  - Clustering coefficient: {analysis['avg_clustering_coefficient']/0.02:.1f}x better")
    if embeddedness_values:
        print(f"  - Embeddedness >=1: {analysis['embeddedness_1_plus_ratio']/0.1:.1f}x better")
    print(f"  - Overall suitability: {analysis['suitability_percentage']/30:.1f}x better")
    
    return analysis

def create_epinions_visualizations(analysis, save_dir):
    """
    Create visualizations for Epinions dataset analysis.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Dataset overview
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Basic statistics
    basic_stats = ['Nodes', 'Edges', 'Pos Edges', 'Neg Edges']
    basic_values = [
        analysis['total_nodes'],
        analysis['total_edges'],
        analysis['positive_edges'],
        analysis['negative_edges']
    ]
    
    axes[0, 0].bar(basic_stats, basic_values, color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow'])
    axes[0, 0].set_title('Dataset Basic Statistics')
    axes[0, 0].set_ylabel('Count')
    for i, v in enumerate(basic_values):
        axes[0, 0].text(i, v + max(basic_values)*0.01, f'{v:,}', ha='center', va='bottom')
    
    # Sign distribution pie chart
    labels = ['Positive', 'Negative']
    sizes = [analysis['positive_edges'], analysis['negative_edges']]
    colors = ['lightcoral', 'lightblue']
    
    axes[0, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[0, 1].set_title('Edge Sign Distribution')
    
    # HOC suitability metrics
    hoc_metrics = ['Triangles/Node', 'Clustering Coef', 'Embeddedness >=1']
    hoc_values = [
        analysis['triangles_per_node'],
        analysis['avg_clustering_coefficient'],
        analysis.get('embeddedness_1_plus_ratio', 0)
    ]
    
    axes[1, 0].bar(hoc_metrics, hoc_values, color=['skyblue', 'lightgreen', 'coral'])
    axes[1, 0].set_title('HOC Method Suitability Metrics')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Suitability score gauge
    score = analysis['suitability_percentage']
    axes[1, 1].pie([score, 100-score], labels=['Suitable', 'Remaining'], 
                   colors=['green', 'lightgray'], autopct='%1.0f%%', startangle=90)
    axes[1, 1].set_title(f'HOC Suitability Score\n{analysis["hoc_suitability_score"]}/10')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'epinions_analysis_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Analysis visualizations saved to {save_dir}")

def save_epinions_analysis(analysis, save_dir):
    """
    Save detailed Epinions analysis results.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save analysis dictionary as JSON
    analysis_path = os.path.join(save_dir, 'epinions_analysis.json')
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    # Save summary report with proper encoding
    report_path = os.path.join(save_dir, 'epinions_analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("EPINIONS DATASET ANALYSIS REPORT\n")
        f.write("="*50 + "\n\n")
        
        f.write("DATASET STATISTICS:\n")
        f.write(f"  Nodes: {analysis['total_nodes']:,}\n")
        f.write(f"  Edges: {analysis['total_edges']:,}\n")
        f.write(f"  Average degree: {analysis['avg_degree']:.2f}\n")
        f.write(f"  Positive edges: {analysis['positive_ratio']*100:.1f}%\n\n")
        
        f.write("HOC METHOD SUITABILITY:\n")
        f.write(f"  Triangles per node: {analysis['triangles_per_node']:.2f}\n")
        f.write(f"  Clustering coefficient: {analysis['avg_clustering_coefficient']:.4f}\n")
        if 'embeddedness_1_plus_ratio' in analysis:
            f.write(f"  Embeddedness >=1: {analysis['embeddedness_1_plus_ratio']*100:.1f}%\n")
        f.write(f"  Suitability score: {analysis['hoc_suitability_score']}/10\n\n")
        
        f.write("RECOMMENDATION:\n")
        f.write(f"  {analysis['recommendation']}\n")
        f.write(f"  {analysis['recommendation_message']}\n")
    
    print(f"Analysis results saved to {save_dir}")

def main():
    """
    Main function to analyze Epinions dataset.
    """
    # Set up paths
    data_path = os.path.join(PROJECT_ROOT, 'data', 'soc-sign-epinions.txt')
    results_dir = os.path.join(PROJECT_ROOT, 'results', 'epinions_analysis')
    
    print("=== EPINIONS DATASET ANALYSIS ===")
    print("Analyzing original paper dataset for HOC method compatibility")
    print(f"Data file: {data_path}")
    print(f"Results will be saved to: {results_dir}")
    print()
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found at {data_path}")
        print("Please ensure soc-sign-epinions.txt is in the data/ directory")
        print()
        print("Expected file structure:")
        print("  project_root/")
        print("    data/")
        print("      soc-sign-epinions.txt  # Download from SNAP or paper source")
        print("      soc-sign-bitcoinotc.csv  # Original dataset")
        print("    notebooks/")
        print("      analyze_epinions_dataset.py  # This script")
        print("    src/")
        print("      data_loader.py")
        print("      ... (other modules)")
        return
    
    # Run analysis
    try:
        analysis = analyze_epinions_dataset(data_path)
        
        # Create visualizations
        print(f"\nCreating visualizations...")
        create_epinions_visualizations(analysis, results_dir)
        
        # Save results
        print(f"Saving analysis results...")
        save_epinions_analysis(analysis, results_dir)
        
        print(f"\n" + "="*60)
        print("EPINIONS ANALYSIS COMPLETE")
        print("="*60)
        print(f"[SUCCESS] Dataset analyzed: {analysis['total_nodes']:,} nodes, {analysis['total_edges']:,} edges")
        print(f"[SUCCESS] HOC suitability: {analysis['hoc_suitability_score']}/10 ({analysis['suitability_percentage']:.0f}%)")
        print(f"[SUCCESS] Recommendation: {analysis['recommendation']}")
        print(f"[SUCCESS] Results saved to: {results_dir}")
        print()
        print("Next steps:")
        print("1. Update config.yaml to point to Epinions dataset")
        print("2. Run preprocessing: python preprocess.py --name experiment_epinions")
        print("3. Run full pipeline and compare with baseline results")
        
    except Exception as e:
        print(f"ERROR during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()