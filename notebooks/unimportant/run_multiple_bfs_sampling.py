#!/usr/bin/env python3
"""
Multiple BFS Sampling for Better Edge Balance
==============================================

Requested by Requirement to find Epinions subsets closer to original 85% positive ratio.
Current BFS result: 93% positive / 7% negative
Target: Find subsets closer to 85% positive / 15% negative

Usage: python notebooks/run_multiple_bfs_sampling.py
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import time
import json
import random
import networkx as nx
from collections import defaultdict

# Add project root to sys.path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import our modules
from src.data_loader import load_bitcoin_data

def run_single_bfs_attempt(run_id, base_config, data_path):
    """
    Run a single BFS sampling attempt with specific configuration
    
    Parameters:
    run_id: Identifier for this run
    base_config: Base configuration dictionary
    data_path: Path to the dataset
    
    Returns:
    dict: Results from this BFS attempt
    """
    print(f"🎯 Run {run_id + 1}")
    print("-" * 40)
    
    # Create variation in sampling strategy
    strategies = [
        {'method': 'random_moderate_degree', 'percentile': 60},
        {'method': 'random_moderate_degree', 'percentile': 70}, 
        {'method': 'random_moderate_degree', 'percentile': 80},
        {'method': 'random_high_degree', 'percentile': 85},
        {'method': 'random_high_degree', 'percentile': 90},
        {'method': 'random_low_degree', 'percentile': 30}
    ]
    
    strategy = strategies[run_id % len(strategies)]
    
    # Set random seeds for reproducibility
    random.seed(42 + run_id * 17)
    np.random.seed(42 + run_id * 17)
    
    # Update configuration
    config = base_config.copy()
    config['bfs_seed_selection'] = strategy['method']
    config['bfs_degree_percentile'] = strategy['percentile']
    
    print(f"   Strategy: {strategy['method']}, percentile: {strategy['percentile']}")
    print(f"   Random seed: {42 + run_id * 17}")
    
    try:
        start_time = time.time()
        
        # Load data with BFS sampling
        G, df = load_bitcoin_data(data_path, 
                                enable_subset_sampling=True, 
                                subset_config=config)
        
        elapsed_time = time.time() - start_time
        
        # Calculate edge statistics
        total_edges = len(df)
        
        # Analyze edge signs/ratings
        if 'rating' in df.columns:
            positive_edges = len(df[df['rating'] > 0])
            negative_edges = len(df[df['rating'] < 0])
            zero_edges = len(df[df['rating'] == 0])
        elif 'weight' in df.columns:
            positive_edges = len(df[df['weight'] > 0])
            negative_edges = len(df[df['weight'] < 0]) 
            zero_edges = len(df[df['weight'] == 0])
        else:
            # Fallback: assume binary classification
            positive_edges = len(df) // 2  # Rough estimate
            negative_edges = len(df) - positive_edges
            zero_edges = 0
        
        pos_ratio = positive_edges / total_edges if total_edges > 0 else 0
        neg_ratio = negative_edges / total_edges if total_edges > 0 else 0
        zero_ratio = zero_edges / total_edges if total_edges > 0 else 0
        
        # Calculate network statistics
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        
        # Calculate degree statistics
        try:
            degrees = dict(G.degree())
            degree_values = list(degrees.values())
            avg_degree = np.mean(degree_values) if degree_values else 0
            max_degree = max(degree_values) if degree_values else 0
            min_degree = min(degree_values) if degree_values else 0
            median_degree = np.median(degree_values) if degree_values else 0
        except Exception as e:
            print(f"   Warning: Could not calculate degree statistics: {e}")
            avg_degree = max_degree = min_degree = median_degree = 0
        
        # Try to calculate clustering coefficient (sample if too large)
        try:
            if n_nodes > 2000:
                # Sample nodes for clustering calculation on large graphs
                sample_nodes = random.sample(list(G.nodes()), min(1000, n_nodes))
                subgraph = G.subgraph(sample_nodes)
                avg_clustering = nx.average_clustering(subgraph)
            else:
                avg_clustering = nx.average_clustering(G)
        except Exception as e:
            print(f"   Warning: Could not calculate clustering: {e}")
            avg_clustering = 0
        
        # Calculate connectivity statistics
        try:
            if nx.is_connected(G.to_undirected()):
                largest_cc_size = n_nodes
            else:
                largest_cc = max(nx.connected_components(G.to_undirected()), key=len)
                largest_cc_size = len(largest_cc)
        except:
            largest_cc_size = n_nodes
        
        result = {
            'run_id': run_id + 1,
            'config': config,
            'strategy': strategy,
            'timing': {
                'elapsed_seconds': elapsed_time
            },
            'edge_statistics': {
                'total_edges': total_edges,
                'positive_edges': positive_edges,
                'negative_edges': negative_edges,
                'zero_edges': zero_edges,
                'positive_ratio': pos_ratio,
                'negative_ratio': neg_ratio,
                'zero_ratio': zero_ratio,
                'distance_from_85pct': abs(pos_ratio - 0.85),
                'distance_from_current': abs(pos_ratio - 0.93)  # Current BFS result
            },
            'network_statistics': {
                'nodes': n_nodes,
                'edges': n_edges,
                'avg_degree': avg_degree,
                'max_degree': max_degree,
                'min_degree': min_degree,
                'median_degree': median_degree,
                'avg_clustering': avg_clustering,
                'largest_cc_size': largest_cc_size,
                'connectivity_ratio': largest_cc_size / n_nodes if n_nodes > 0 else 0
            },
            'success': True
        }
        
        # Report results
        print(f"✅ Sampling completed in {elapsed_time:.1f}s")
        print(f"   📊 Edges: {total_edges:,} ({positive_edges:,} pos, {negative_edges:,} neg)")
        if zero_edges > 0:
            print(f"   📊 Zero edges: {zero_edges:,}")
        print(f"   📈 Ratio: {pos_ratio:.1%} positive, {neg_ratio:.1%} negative")
        print(f"   🎯 Target distance: {abs(pos_ratio - 0.85):.1%} from 85%")
        print(f"   📊 Current distance: {abs(pos_ratio - 0.93):.1%} from current 93%")
        print(f"   🌐 Network: {n_nodes:,} nodes, avg degree {avg_degree:.1f}")
        print(f"   🔗 Clustering: {avg_clustering:.3f}")
        
        # Save promising subsets
        if abs(pos_ratio - 0.85) < 0.08:  # Within 8% of target
            save_promising_subset(df, run_id, pos_ratio, neg_ratio, config)
        
        return result
        
    except Exception as e:
        print(f"❌ Run {run_id + 1} failed: {e}")
        result = {
            'run_id': run_id + 1,
            'config': config,
            'strategy': strategy,
            'error': str(e),
            'success': False
        }
        return result

def save_promising_subset(df, run_id, pos_ratio, neg_ratio, config):
    """
    Save a promising subset for later use
    """
    output_dir = Path("data/bfs_subsets")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create descriptive filename
    pos_pct = int(pos_ratio * 100)
    neg_pct = int(neg_ratio * 100)
    strategy = config.get('bfs_seed_selection', 'unknown')
    percentile = config.get('bfs_degree_percentile', 0)
    
    filename = f"epinions_bfs_run{run_id + 1}_pos{pos_pct}pct_neg{neg_pct}pct_{strategy}_p{percentile}.txt"
    filepath = output_dir / filename
    
    # Save in same format as original (tab-separated, no header)
    df_save = df.copy()
    
    # Ensure correct format for Epinions data
    if 'rating' in df_save.columns:
        # Keep rating column as is
        save_columns = ['source', 'target', 'rating']
    elif 'weight' in df_save.columns:
        # Use weight as rating
        df_save['rating'] = df_save['weight']
        save_columns = ['source', 'target', 'rating']
    else:
        print(f"   Warning: Unknown data format for {filename}")
        return
    
    # Save without header, tab-separated
    df_save[save_columns].to_csv(filepath, sep='\t', index=False, header=False)
    print(f"   💾 Saved promising subset to {filepath}")

def run_multiple_bfs_sampling(n_runs=6, target_edge_count=35000):
    """
    Run BFS sampling multiple times with different strategies
    
    Parameters:
    n_runs: Number of different BFS runs to attempt
    target_edge_count: Target number of edges per subset
    
    Returns:
    List of results with statistics for each run
    """
    print(f"🔄 Running {n_runs} BFS sampling attempts for better edge balance")
    print(f"Target: {target_edge_count:,} edges, seeking ~85% positive ratio")
    print(f"Current BFS result: 93% positive / 7% negative")
    print(f"Goal: Find subsets closer to original Epinions distribution\n")
    
    # Check if data file exists
    data_file = 'data/soc-sign-epinions.txt'
    if not Path(data_file).exists():
        print(f"❌ Data file not found: {data_file}")
        print("   Please ensure the Epinions dataset is available in data/ directory")
        print("   Download from: https://snap.stanford.edu/data/soc-sign-epinions.html")
        return []
    
    # Base configuration for BFS sampling
    base_config = {
        'enable_subset_sampling': True,
        'subset_sampling_method': 'bfs_sampling',
        'target_edge_count': target_edge_count,
        'subset_preserve_structure': True,
        'bfs_max_iterations': 10000,  # Allow more iterations
        'bfs_tolerance': 0.1  # 10% tolerance on target size
    }
    
    results = []
    
    for run_id in range(n_runs):
        result = run_single_bfs_attempt(run_id, base_config, data_file)
        results.append(result)
        print()  # Empty line between runs
    
    return results

def analyze_multiple_runs(results):
    """
    Analyze results from multiple BFS runs and identify the best options
    """
    print("📈 Analysis of Multiple BFS Sampling Results")
    print("=" * 60)
    
    successful_runs = [r for r in results if r.get('success', False)]
    failed_runs = [r for r in results if not r.get('success', False)]
    
    print(f"✅ Successful runs: {len(successful_runs)}/{len(results)}")
    if failed_runs:
        print(f"❌ Failed runs: {len(failed_runs)}")
        for run in failed_runs:
            print(f"   Run {run['run_id']}: {run.get('error', 'Unknown error')}")
    print()
    
    if not successful_runs:
        print("❌ No successful runs to analyze")
        return None
    
    # Create comparison DataFrame
    comparison_data = []
    for result in successful_runs:
        stats = result['edge_statistics']
        network = result['network_statistics']
        strategy = result['strategy']
        
        comparison_data.append({
            'Run': result['run_id'],
            'Strategy': strategy['method'],
            'Percentile': strategy['percentile'],
            'Total_Edges': stats['total_edges'],
            'Pos_Ratio': stats['positive_ratio'],
            'Neg_Ratio': stats['negative_ratio'],
            'Distance_from_85%': stats['distance_from_85pct'],
            'Distance_from_93%': stats['distance_from_current'],
            'Nodes': network['nodes'],
            'Avg_Degree': network['avg_degree'],
            'Clustering': network['avg_clustering'],
            'Connectivity': network['connectivity_ratio']
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Sort by distance from target 85% positive
    df_comparison = df_comparison.sort_values('Distance_from_85%')
    
    print("🏆 Ranking by Closeness to 85% Positive Target:")
    print("-" * 55)
    for _, row in df_comparison.iterrows():
        improvement_vs_current = row['Distance_from_93%'] - row['Distance_from_85%']
        status = "🔥" if improvement_vs_current > 0.02 else "✅" if improvement_vs_current > 0 else "📊"
        print(f"{status} Run {row['Run']:2d}: {row['Pos_Ratio']:.1%} pos "
              f"({row['Distance_from_85%']:.1%} from target), "
              f"{row['Strategy'][:15]}... p{row['Percentile']}")
    
    # Find best options
    best_run = df_comparison.iloc[0]
    print(f"\n🎯 Best Run: #{best_run['Run']}")
    print(f"   Positive ratio: {best_run['Pos_Ratio']:.1%} (target: 85.0%)")
    print(f"   Distance from target: {best_run['Distance_from_85%']:.1%}")
    print(f"   Total edges: {best_run['Total_Edges']:,}")
    print(f"   Network size: {best_run['Nodes']:,} nodes")
    print(f"   Strategy: {best_run['Strategy']}, percentile: {best_run['Percentile']}")
    
    # Check if any runs are significantly better than current 93% positive
    current_distance = abs(0.93 - 0.85)  # Current BFS result distance from target
    improved_runs = df_comparison[df_comparison['Distance_from_85%'] < current_distance]
    
    print(f"\n🔄 Comparison with Current BFS Result:")
    print(f"   Current: 93% positive (distance: {current_distance:.1%} from 85% target)")
    print(f"   Improved options: {len(improved_runs)}/{len(successful_runs)} runs")
    
    if len(improved_runs) > 0:
        print("✨ Found better balanced subsets!")
        for _, row in improved_runs.iterrows():
            improvement = current_distance - row['Distance_from_85%']
            print(f"   Run {row['Run']}: {improvement:.1%} better balance "
                  f"({row['Pos_Ratio']:.1%} positive)")
        
        # Recommend the best option
        best_improved = improved_runs.iloc[0]
        print(f"\n💡 RECOMMENDATION:")
        print(f"   Use Run #{best_improved['Run']} results for better balance")
        print(f"   Improvement: {current_distance - best_improved['Distance_from_85%']:.1%} closer to target")
        print(f"   New ratio: {best_improved['Pos_Ratio']:.1%} positive vs current 93%")
        
    else:
        print("⚠️  No runs achieved significantly better balance than current 93%")
        print("   Current BFS sampling appears near-optimal for this network structure")
    
    return df_comparison

def save_analysis_results(results, comparison_df, save_dir="results/bfs_multiple_runs"):
    """
    Save detailed analysis results to files
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Save comparison table
    if comparison_df is not None:
        comparison_df.round(4).to_csv(Path(save_dir) / "bfs_runs_comparison.csv", index=False)
        print(f"📊 Comparison table saved to {save_dir}/bfs_runs_comparison.csv")
    
    # Save detailed results
    with open(Path(save_dir) / "bfs_runs_detailed.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"📄 Detailed results saved to {save_dir}/bfs_runs_detailed.json")
    
    # Generate summary report
    generate_bfs_summary_report(results, comparison_df, save_dir)

def generate_bfs_summary_report(results, comparison_df, save_dir):
    """
    Generate a markdown summary report
    """
    report_path = Path(save_dir) / "bfs_sampling_report.md"
    
    successful_runs = [r for r in results if r.get('success', False)]
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Multiple BFS Sampling Analysis Report\n\n")
        f.write("Generated in response to Requirement's request for better edge balance\n\n")
        
        f.write("## Objective\n")
        f.write("Find Epinions subsets closer to original 85% positive / 15% negative distribution.\n")
        f.write("Current BFS result: 93% positive / 7% negative\n\n")
        
        f.write("## Results Summary\n")
        f.write(f"- Total attempts: {len(results)}\n")
        f.write(f"- Successful runs: {len(successful_runs)}\n")
        f.write(f"- Failed runs: {len(results) - len(successful_runs)}\n\n")
        
        if comparison_df is not None and len(comparison_df) > 0:
            best_run = comparison_df.iloc[0]
            f.write("## Best Result\n")
            f.write(f"- **Run #{best_run['Run']}**\n")
            f.write(f"- Positive ratio: {best_run['Pos_Ratio']:.1%}\n")
            f.write(f"- Distance from 85% target: {best_run['Distance_from_85%']:.1%}\n")
            f.write(f"- Strategy: {best_run['Strategy']}, percentile {best_run['Percentile']}\n")
            f.write(f"- Network: {best_run['Nodes']:,} nodes, {best_run['Total_Edges']:,} edges\n\n")
            
            # Check for improvements
            current_distance = abs(0.93 - 0.85)
            improved_runs = comparison_df[comparison_df['Distance_from_85%'] < current_distance]
            
            if len(improved_runs) > 0:
                f.write("## Recommendation\n")
                f.write("✅ **Found better balanced subsets!**\n\n")
                best_improved = improved_runs.iloc[0]
                improvement = current_distance - best_improved['Distance_from_85%']
                f.write(f"Use Run #{best_improved['Run']} for {improvement:.1%} better balance:\n")
                f.write(f"- New ratio: {best_improved['Pos_Ratio']:.1%} positive\n")
                f.write(f"- Saved as: `data/bfs_subsets/epinions_bfs_run{best_improved['Run']}_*`\n\n")
                
                f.write("### Next Steps\n")
                f.write("1. Update config.yaml to use the improved subset\n")
                f.write("2. Re-run experiments with better balanced data\n")
                f.write("3. Compare results in final report\n\n")
            else:
                f.write("## Conclusion\n")
                f.write("Current 93% positive ratio appears near-optimal for BFS sampling on this network.\n")
                f.write("Network structure may inherently favor positive edges in connected components.\n\n")
        
        f.write("## Files Generated\n")
        f.write("- `bfs_runs_comparison.csv` - Comparison table\n")
        f.write("- `bfs_runs_detailed.json` - Detailed results\n")
        f.write("- `data/bfs_subsets/` - Promising subset files\n")
    
    print(f"📝 Summary report saved to {report_path}")

def main():
    """
    Main function to run multiple BFS sampling attempts
    """
    print("🔍 Multiple BFS Sampling for Better Edge Balance")
    print("=" * 60)
    print("Requested by Requirement: Find Epinions subsets closer to 85% positive ratio")
    print("Current situation: BFS sampling yields 93% positive / 7% negative")
    print("Goal: Improve balance for fairer comparison with Bitcoin OTC dataset\n")
    
    # Configuration
    n_runs = 6  # Try 6 different sampling strategies
    target_edges = 35000  # Target similar to Bitcoin OTC size
    
    print(f"Configuration:")
    print(f"  Number of runs: {n_runs}")
    print(f"  Target edges per run: {target_edges:,}")
    print(f"  Strategies: Varying degree percentiles and seed selection methods")
    print(f"  Seeds: Different random seeds for each run\n")
    
    # Run multiple sampling attempts
    print("Starting multiple BFS sampling runs...")
    results = run_multiple_bfs_sampling(n_runs, target_edges)
    
    if not results:
        print("❌ No results to analyze. Check data file availability.")
        return
    
    # Analyze results
    print("\n" + "="*60)
    comparison_df = analyze_multiple_runs(results)
    
    # Save results
    print(f"\n📁 Saving analysis results...")
    save_analysis_results(results, comparison_df)
    
    # Final recommendations
    print(f"\n💡 Final Recommendations for Requirement:")
    print("-" * 40)
    
    successful_runs = [r for r in results if r.get('success', False)]
    if successful_runs and comparison_df is not None:
        current_distance = abs(0.93 - 0.85)
        best_result = comparison_df.iloc[0]
        
        if best_result['Distance_from_85%'] < current_distance:
            improvement = current_distance - best_result['Distance_from_85%']
            print(f"✅ SUCCESS: Found {improvement:.1%} better balance!")
            print(f"   Best option: Run #{best_result['Run']} with {best_result['Pos_Ratio']:.1%} positive")
            print(f"   Saved subset ready for use in data/bfs_subsets/")
            print(f"   Recommend re-running experiments with improved balance")
        else:
            print(f"📊 ANALYSIS COMPLETE: Current 93% positive appears optimal")
            print(f"   Network structure may inherently favor positive edges")
            print(f"   Consider this finding for report discussion")
    
    print(f"\n🎯 Ready for report preparation!")
    print(f"📊 All analysis data and plots available for inclusion")

if __name__ == "__main__":
    main()