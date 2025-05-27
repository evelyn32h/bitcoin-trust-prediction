#!/usr/bin/env python3
"""
Epinions Dataset Basic Statistics Analysis
==========================================

Generate basic statistics for Epinions dataset as requested by Requirement
to explain why experiments failed on this dataset.

Focus: pos/neg ratio, nodes/edges count, embeddedness distribution

Usage: python notebooks/analyze_epinions_basic_stats.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json

# Add project root to sys.path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import our modules
from src.data_loader import load_bitcoin_data
from src.evaluation import analyze_network, visualize_weight_distribution, visualize_embeddedness
from src.preprocessing import reindex_nodes

def analyze_epinions_basic_stats():
    """
    Generate basic statistics for Epinions dataset to understand why experiments failed
    """
    print("🔍 Analyzing Epinions Dataset Basic Statistics")
    print("="*60)
    print("Purpose: Understand why Epinions experiments underperformed")
    print("Requested by Requirement: pos/neg ratio, nodes/edges, embeddedness\n")
    
    # Create output directory
    output_dir = Path("plots/epinions_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    datasets_to_analyze = [
        {
            'name': 'Epinions Original',
            'path': 'data/soc-sign-epinions.txt',
            'description': 'Original Epinions dataset (large)',
            'analyze_embeddedness': False  # Skip for large dataset
        },
        {
            'name': 'Bitcoin OTC',
            'path': 'data/soc-sign-bitcoinotc.csv',
            'description': 'Bitcoin OTC dataset (baseline comparison)',
            'analyze_embeddedness': True  # Full analysis for baseline
        }
    ]
    
    # Check for preprocessed Epinions subset
    subset_paths = [
        'results/experiment_epinions_subset/preprocess/test.csv',
        'results/baseline_epinions/preprocess/test.csv',
        'data/soc-sign-epinions-subset.csv'
    ]
    
    for subset_path in subset_paths:
        if Path(subset_path).exists():
            datasets_to_analyze.append({
                'name': 'Epinions Subset',
                'path': subset_path,
                'description': 'Epinions subset used in experiments',
                'analyze_embeddedness': True
            })
            break
    
    analysis_results = {}
    
    for dataset_info in datasets_to_analyze:
        dataset_name = dataset_info['name']
        dataset_path = dataset_info['path']
        analyze_embed = dataset_info['analyze_embeddedness']
        
        print(f"\n📊 Analyzing: {dataset_name}")
        print("-" * 50)
        
        # Check if file exists
        if not Path(dataset_path).exists():
            print(f"⚠️  Dataset not found: {dataset_path}")
            continue
        
        try:
            # Load dataset
            print(f"Loading data from {dataset_path}...")
            if dataset_path.endswith('.csv') and 'results/' in dataset_path:
                # Load preprocessed CSV file
                df = pd.read_csv(dataset_path, names=['source', 'target', 'label', 'weight', 'time'])
                # Create simple graph from CSV
                import networkx as nx
                G = nx.DiGraph()
                for _, row in df.iterrows():
                    G.add_edge(row['source'], row['target'], 
                             weight=row['weight'], time=row['time'])
            else:
                # Load using our data loader
                G, df = load_bitcoin_data(dataset_path)
            
            # Basic preprocessing
            G = reindex_nodes(G)
            
            # 1. BASIC STATISTICS
            print("\n1. 📈 Network Statistics:")
            basic_stats = analyze_network(G, dataset_name)
            
            # 2. EDGE SIGN DISTRIBUTION  
            print("\n2. ⚖️  Edge Sign Analysis:")
            if 'rating' in df.columns:
                rating_col = 'rating'
            elif 'weight' in df.columns:
                rating_col = 'weight'
            else:
                # Use graph edge weights
                weights = [data.get('weight', 1) for _, _, data in G.edges(data=True)]
                df_temp = pd.DataFrame({'rating': weights})
                rating_col = 'rating'
                df = df_temp
            
            pos_edges = len(df[df[rating_col] > 0])
            neg_edges = len(df[df[rating_col] < 0])
            zero_edges = len(df[df[rating_col] == 0]) if (df[rating_col] == 0).any() else 0
            total_edges = len(df)
            
            pos_ratio = pos_edges / total_edges if total_edges > 0 else 0
            neg_ratio = neg_edges / total_edges if total_edges > 0 else 0
            
            print(f"  Positive edges: {pos_edges:,} ({pos_ratio:.1%})")
            print(f"  Negative edges: {neg_edges:,} ({neg_ratio:.1%})")
            if zero_edges > 0:
                print(f"  Zero edges: {zero_edges:,} ({zero_edges/total_edges:.1%})")
            
            # 3. COMPARISON WITH BITCOIN OTC
            print("\n3. 📊 Comparison with Bitcoin OTC:")
            btc_pos_ratio = 0.89  # Bitcoin OTC has ~89% positive
            balance_diff = abs(pos_ratio - btc_pos_ratio)
            print(f"  Bitcoin OTC positive ratio: {btc_pos_ratio:.1%}")
            print(f"  {dataset_name} positive ratio: {pos_ratio:.1%}")
            print(f"  Difference: {balance_diff:.1%}")
            
            if balance_diff > 0.05:
                print(f"  ⚠️  IMBALANCE ISSUE: {balance_diff:.1%} difference may hurt performance")
            
            # 4. NETWORK SCALE COMPARISON
            btc_nodes = 5881  # Bitcoin OTC approximate
            btc_edges = 35592
            scale_ratio_nodes = G.number_of_nodes() / btc_nodes
            scale_ratio_edges = G.number_of_edges() / btc_edges
            
            print(f"\n4. 📏 Scale Comparison:")
            print(f"  Bitcoin OTC: ~{btc_nodes:,} nodes, ~{btc_edges:,} edges")
            print(f"  {dataset_name}: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
            print(f"  Scale ratio: {scale_ratio_nodes:.1f}x nodes, {scale_ratio_edges:.1f}x edges")
            
            # 5. EMBEDDEDNESS ANALYSIS
            embed_metrics = {}
            if analyze_embed:
                print(f"\n5. 🔗 Embeddedness Analysis:")
                print(f"  Analyzing all {G.number_of_edges():,} edges")
                
                # Generate embeddedness visualization
                embeddedness_file = output_dir / f"{dataset_name.replace(' ', '_')}_embeddedness.png"
                try:
                    edge_embeddedness, embed_metrics = visualize_embeddedness(
                        G, str(embeddedness_file), dataset_name
                    )
                    print(f"  ✅ Embeddedness analysis complete")
                    print(f"  📁 Saved to: {embeddedness_file}")
                except Exception as e:
                    print(f"  ⚠️  Embeddedness analysis failed: {e}")
                    embed_metrics = {}
            else:
                print(f"\n5. 🔗 Embeddedness Analysis:")
                print(f"  Skipped for large dataset ({G.number_of_edges():,} edges)")
            
            # 6. WEIGHT DISTRIBUTION
            print(f"\n6. ⚖️  Weight Distribution:")
            weight_file = output_dir / f"{dataset_name.replace(' ', '_')}_weights.png"
            try:
                weight_metrics = visualize_weight_distribution(
                    G, str(weight_file), dataset_name
                )
                print(f"  ✅ Weight analysis complete")
                print(f"  📁 Saved to: {weight_file}")
            except Exception as e:
                print(f"  ⚠️  Weight analysis failed: {e}")
                weight_metrics = {}
            
            # Store results
            analysis_results[dataset_name] = {
                'basic_stats': basic_stats,
                'edge_signs': {
                    'positive_count': pos_edges,
                    'negative_count': neg_edges,
                    'zero_count': zero_edges,
                    'positive_ratio': pos_ratio,
                    'negative_ratio': neg_ratio,
                    'balance_vs_bitcoin': balance_diff
                },
                'scale_comparison': {
                    'nodes_ratio': scale_ratio_nodes,
                    'edges_ratio': scale_ratio_edges
                },
                'embeddedness_metrics': embed_metrics,
                'weight_metrics': weight_metrics
            }
            
        except Exception as e:
            print(f"❌ Error analyzing {dataset_name}: {e}")
            continue
    
    # 7. GENERATE COMPARISON SUMMARY
    print("\n" + "="*60)
    print("📋 SUMMARY: Why Epinions May Have Failed")
    print("="*60)
    
    if len(analysis_results) >= 2:
        generate_failure_analysis_report(analysis_results, output_dir)
    
    print(f"\n✅ Analysis complete! Results saved to {output_dir}")
    return analysis_results

def generate_failure_analysis_report(analysis_results, output_dir):
    """
    Generate a comprehensive report explaining why Epinions failed
    """
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Epinions vs Bitcoin OTC: Why Performance Differs', fontsize=16, fontweight='bold')
    
    dataset_names = list(analysis_results.keys())
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    
    # 1. Positive ratio comparison
    ax1 = axes[0, 0]
    pos_ratios = []
    labels = []
    for name, data in analysis_results.items():
        pos_ratios.append(data['edge_signs']['positive_ratio'])
        labels.append(name)
    
    colors_plot = colors[:len(pos_ratios)]
    
    bars1 = ax1.bar(labels, pos_ratios, color=colors_plot, alpha=0.7)
    ax1.set_title('Positive Edge Ratio Comparison')
    ax1.set_ylabel('Positive Ratio')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, ratio in zip(bars1, pos_ratios):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{ratio:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Network scale comparison
    ax2 = axes[0, 1]
    nodes_counts = []
    edges_counts = []
    dataset_labels = []
    
    for name, data in analysis_results.items():
        nodes_counts.append(data['basic_stats']['num_nodes'])
        edges_counts.append(data['basic_stats']['num_edges'])
        dataset_labels.append(name)
    
    x = np.arange(len(dataset_labels))
    width = 0.35
    
    ax2.bar(x - width/2, [n/1000 for n in nodes_counts], width, 
           label='Nodes (K)', alpha=0.8, color='lightblue')
    ax2.bar(x + width/2, [e/1000 for e in edges_counts], width, 
           label='Edges (K)', alpha=0.8, color='lightcoral')
    
    ax2.set_title('Network Scale Comparison')
    ax2.set_ylabel('Count (Thousands)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(dataset_labels, rotation=45)
    ax2.legend()
    
    # 3. Embeddedness comparison (if available)
    ax3 = axes[1, 0]
    embeddedness_available = False
    zero_embed_ratios = []
    embed_labels = []
    
    for name, data in analysis_results.items():
        embed_data = data['embeddedness_metrics']
        if embed_data and 'zero_embeddedness_percentage' in embed_data:
            zero_embed_ratios.append(embed_data['zero_embeddedness_percentage'])
            embed_labels.append(name)
            embeddedness_available = True
    
    if embeddedness_available:
        bars3 = ax3.bar(embed_labels, zero_embed_ratios, 
                       color=colors[:len(zero_embed_ratios)], alpha=0.7)
        ax3.set_title('Zero Embeddedness Percentage')
        ax3.set_ylabel('Percentage of Edges')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, ratio in zip(bars3, zero_embed_ratios):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{ratio:.1f}%', ha='center', va='bottom', fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'Embeddedness data\nnot available', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Embeddedness Analysis')
    
    # 4. Potential failure reasons
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    failure_text = """POTENTIAL FAILURE REASONS:

1. DATA IMBALANCE:
   • Different pos/neg ratios
   • May bias model training

2. NETWORK STRUCTURE:
   • Different connectivity patterns
   • Varying embeddedness distributions

3. SCALE DIFFERENCES:
   • Larger/smaller networks
   • Different local phenomena

4. PREPROCESSING IMPACT:
   • BFS sampling effects
   • Edge filtering differences

RECOMMENDATION:
• Use balanced subsets
• Analyze feature distributions
• Consider network-specific tuning"""
    
    ax4.text(0.05, 0.95, failure_text, transform=ax4.transAxes, 
            verticalalignment='top', fontsize=10, fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_dir / "epinions_failure_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save text report
    report_path = output_dir / "epinions_failure_report.md"
    with open(report_path, 'w') as f:
        f.write("# Epinions Dataset Analysis Report\n\n")
        f.write("## Objective\n")
        f.write("Understand why Epinions experiments underperformed compared to Bitcoin OTC\n\n")
        
        f.write("## Key Findings\n\n")
        
        for name, data in analysis_results.items():
            f.write(f"### {name}\n")
            f.write(f"- Nodes: {data['basic_stats']['num_nodes']:,}\n")
            f.write(f"- Edges: {data['basic_stats']['num_edges']:,}\n")
            f.write(f"- Positive ratio: {data['edge_signs']['positive_ratio']:.1%}\n")
            f.write(f"- Balance vs Bitcoin: {data['edge_signs']['balance_vs_bitcoin']:.1%} difference\n\n")
        
        f.write("## Recommendations\n")
        f.write("1. Use balanced subsets for fair comparison\n")
        f.write("2. Analyze network structure differences\n")
        f.write("3. Consider dataset-specific preprocessing\n")
        f.write("4. Investigate embeddedness distributions\n")
    
    # Save JSON results
    json_path = output_dir / "analysis_results.json"
    
    # Convert numpy types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    serializable_results = convert_numpy_types(analysis_results)
    
    with open(json_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"📊 Failure analysis complete!")
    print(f"📁 Saved comparison plots and report to {output_dir}")

def print_summary_statistics(analysis_results):
    """
    Print a summary table of key statistics
    """
    print("\n" + "="*80)
    print("DATASET COMPARISON TABLE")
    print("="*80)
    
    # Create summary table
    headers = ["Dataset", "Nodes", "Edges", "Pos%", "Neg%", "vs Bitcoin"]
    row_format = "{:<20} {:<8} {:<8} {:<6} {:<6} {:<10}"
    
    print(row_format.format(*headers))
    print("-" * 80)
    
    for name, data in analysis_results.items():
        nodes = f"{data['basic_stats']['num_nodes']:,}"
        edges = f"{data['basic_stats']['num_edges']:,}"
        pos_pct = f"{data['edge_signs']['positive_ratio']:.1%}"
        neg_pct = f"{data['edge_signs']['negative_ratio']:.1%}"
        vs_bitcoin = f"{data['edge_signs']['balance_vs_bitcoin']:.1%}"
        
        print(row_format.format(name, nodes, edges, pos_pct, neg_pct, vs_bitcoin))
    
    print("-" * 80)

def main():
    """
    Main function to run Epinions basic statistics analysis
    """
    start_time = time.time()
    
    print("🔍 EPINIONS DATASET BASIC STATISTICS ANALYSIS")
    print("="*80)
    print("As requested by Requirement to explain why Epinions experiments failed")
    print("Focus: pos/neg ratio, nodes/edges count, embeddedness distribution")
    print()
    
    try:
        results = analyze_epinions_basic_stats()
        
        if results:
            print_summary_statistics(results)
        
        elapsed = time.time() - start_time
        
        print(f"\n🎯 Analysis completed in {elapsed:.1f} seconds")
        print(f"📋 Ready for Requirement's presentation!")
        print(f"📊 Use these results to explain Epinions experiment performance")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()