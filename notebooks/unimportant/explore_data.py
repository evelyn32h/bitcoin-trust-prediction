#!/usr/bin/env python3
"""
Bitcoin Trust Network Data Exploration Pipeline

This script provides a command-line interface for exploring and analyzing the Bitcoin trust network data.
It separates analysis functions from data saving concerns for better modularity.

Usage:
    python explore_data.py [--stages STAGES] [--output-dir OUTPUT_DIR] [--save-plots] [--save-metrics]

Arguments:
    --stages: Comma-separated list of analysis stages to run (default: all)
              Options: network, weights, degrees, embeddedness, temporal, connectivity, preprocessing
    --output-dir: Directory to save results (default: ../results/exploration)
    --save-plots: Save generated plots to files
    --save-metrics: Save calculated metrics to JSON files
    
Examples:
    python explore_data.py --stages network,weights --save-plots
    python explore_data.py --stages all --output-dir ../results/my_analysis --save-metrics
    python explore_data.py --stages preprocessing --save-plots --save-metrics
"""

import argparse
import sys
import os
from pathlib import Path

import yaml

# Add src directory to Python path
# Add project root to sys.path for src imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Load config from YAML
with open(os.path.join(PROJECT_ROOT, 'config.yaml'), 'r') as f:
    config = yaml.safe_load(f)
    
# Import analysis functions (separated from saving)
from src.evaluation import (
    # Network analysis functions
    analyze_network,
    calculate_embeddedness,
    
    # Visualization functions (now return metrics only)
    visualize_weight_distribution,
    visualize_degree_distribution, 
    visualize_embeddedness,
    analyze_temporal_patterns,
    visualize_connectivity_analysis,
    visualize_preprocessing_pipeline,
)

# Import data handling functions
from src.data_loader import load_bitcoin_data, save_metrics_to_json
from src.preprocessing import (
    map_to_unweighted_graph, 
    ensure_connectivity, 
    filter_neutral_edges, 
    reindex_nodes
)

def setup_output_directory(output_dir):
    """Create output directory structure"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for different types of outputs
    plots_dir = output_path / "plots"
    metrics_dir = output_path / "metrics"
    
    plots_dir.mkdir(exist_ok=True)
    metrics_dir.mkdir(exist_ok=True)
    
    return output_path, plots_dir, metrics_dir

def run_network_analysis(G, output_path, save_plots, save_metrics):
    """Run basic network analysis"""
    print("\n" + "="*60)
    print("STAGE 1: BASIC NETWORK ANALYSIS")
    print("="*60)
    
    # Analyze network structure
    network_stats = analyze_network(G, "Bitcoin Trust Network")
    
    if save_metrics:
        metrics_file = output_path / "metrics" / "network_analysis.json"
        save_metrics_to_json(network_stats, str(metrics_file))
        print(f"Network metrics saved to {metrics_file}")
    
    return network_stats

def run_weight_analysis(G, output_path, save_plots, save_metrics):
    """Run weight distribution analysis"""
    print("\n" + "="*60)
    print("STAGE 2: WEIGHT DISTRIBUTION ANALYSIS")
    print("="*60)
    
    plot_path = None
    if save_plots:
        plot_path = str(output_path / "plots" / "weight_distribution.png")
    
    weight_metrics = visualize_weight_distribution(G, save_path=plot_path, graph_name="Bitcoin Trust Network")
    
    if save_metrics:
        metrics_file = output_path / "metrics" / "weight_metrics.json"
        save_metrics_to_json(weight_metrics, str(metrics_file))
        print(f"Weight metrics saved to {metrics_file}")
    
    return weight_metrics

def run_degree_analysis(G, output_path, save_plots, save_metrics):
    """Run degree distribution analysis"""
    print("\n" + "="*60)
    print("STAGE 3: DEGREE DISTRIBUTION ANALYSIS")
    print("="*60)
    
    plot_path = None
    if save_plots:
        plot_path = str(output_path / "plots" / "degree_distribution.png")
    
    degree_metrics = visualize_degree_distribution(G, save_path=plot_path, graph_name="Bitcoin Trust Network")
    
    if save_metrics:
        metrics_file = output_path / "metrics" / "degree_metrics.json"
        save_metrics_to_json(degree_metrics, str(metrics_file))
        print(f"Degree metrics saved to {metrics_file}")
    
    return degree_metrics

def run_embeddedness_analysis(G, output_path, save_plots, save_metrics):
    """Run embeddedness analysis"""
    print("\n" + "="*60)
    print("STAGE 4: EMBEDDEDNESS ANALYSIS")
    print("="*60)
    
    plot_path = None
    if save_plots:
        plot_path = str(output_path / "plots" / "embeddedness_distribution.png")
    
    edge_embeddedness, embeddedness_metrics = visualize_embeddedness(
        G, save_path=plot_path, graph_name="Bitcoin Trust Network"
    )
    
    if save_metrics:
        metrics_file = output_path / "metrics" / "embeddedness_metrics.json"
        save_metrics_to_json(embeddedness_metrics, str(metrics_file))
        print(f"Embeddedness metrics saved to {metrics_file}")
    
    return embeddedness_metrics

def run_temporal_analysis(G, df, output_path, save_plots, save_metrics):
    """Run temporal pattern analysis"""
    print("\n" + "="*60)
    print("STAGE 5: TEMPORAL ANALYSIS")
    print("="*60)
    
    plot_path = None
    if save_plots:
        plot_path = str(output_path / "plots" / "temporal_patterns.png")
    
    temporal_metrics = analyze_temporal_patterns(G, df, save_path=plot_path)
    
    if save_metrics:
        metrics_file = output_path / "metrics" / "temporal_metrics.json"
        save_metrics_to_json(temporal_metrics, str(metrics_file))
        print(f"Temporal metrics saved to {metrics_file}")
    
    return temporal_metrics

def run_connectivity_analysis(output_path, save_plots, save_metrics):
    """Run connectivity analysis across preprocessing steps"""
    print("\n" + "="*60)
    print("STAGE 6: CONNECTIVITY ANALYSIS")
    print("="*60)
    
    # Load and preprocess data to show connectivity changes
    data_path = Path('..') / 'data' / 'soc-sign-bitcoinotc.csv'
    if not data_path.exists():
        data_path = Path('data') / 'soc-sign-bitcoinotc.csv'
    
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return {}
    
    # Load original data and create preprocessing steps
    G_original, _ = load_bitcoin_data(str(data_path))
    
    graphs_dict = {
        'Original': G_original,
        'Filtered': filter_neutral_edges(G_original),
    }
    
    # Add signed and connected versions
    G_signed = map_to_unweighted_graph(graphs_dict['Filtered'])
    G_connected = ensure_connectivity(G_signed)
    G_final = reindex_nodes(G_connected)
    
    graphs_dict.update({
        'Signed': G_signed,
        'Connected': G_connected,
        'Final': G_final
    })
    
    plot_path = None
    if save_plots:
        plot_path = str(output_path / "plots" / "connectivity_analysis.png")
    
    connectivity_metrics = visualize_connectivity_analysis(graphs_dict, save_path=plot_path)
    
    if save_metrics:
        metrics_file = output_path / "metrics" / "connectivity_metrics.json"
        save_metrics_to_json(connectivity_metrics, str(metrics_file))
        print(f"Connectivity metrics saved to {metrics_file}")
    
    return connectivity_metrics

def run_preprocessing_analysis(output_path, save_plots, save_metrics):
    """Run complete preprocessing pipeline analysis"""
    print("\n" + "="*60)
    print("STAGE 7: PREPROCESSING PIPELINE ANALYSIS")
    print("="*60)
    
    # Note: visualize_preprocessing_pipeline handles its own file saving
    # We'll save additional summary metrics if requested
    graphs, pipeline_stats = visualize_preprocessing_pipeline()
    
    if save_metrics:
        metrics_file = output_path / "metrics" / "preprocessing_pipeline_metrics.json"
        save_metrics_to_json(pipeline_stats, str(metrics_file))
        print(f"Preprocessing pipeline metrics saved to {metrics_file}")
    
    return pipeline_stats

def main():
    parser = argparse.ArgumentParser(
        description="Bitcoin Trust Network Data Exploration Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('\n\n', 1)[1]  # Show the usage examples
    )
    
    parser.add_argument(
        '--stages', 
        type=str, 
        default='all',
        help='Comma-separated list of analysis stages to run. Options: network, weights, degrees, embeddedness, temporal, connectivity, preprocessing, all (default: all)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=f'./results/{config["default_experiment_name"]}/exploration',
        help='Directory to save results (default: results/exploration)'
    )
    
    parser.add_argument(
        '--save-plots',
        action='store_true',
        default=True,
        help='Save generated plots to files'
    )
    
    parser.add_argument(
        '--save-metrics',
        action='store_true',
        default=True,
        help='Save calculated metrics to JSON files'
    )
    
    args = parser.parse_args()
    
    # Parse stages
    if args.stages.lower() == 'all':
        stages = ['network', 'weights', 'degrees', 'embeddedness', 'temporal', 'connectivity', 'preprocessing']
    else:
        stages = [stage.strip().lower() for stage in args.stages.split(',')]
    
    # Validate stages
    valid_stages = ['network', 'weights', 'degrees', 'embeddedness', 'temporal', 'connectivity', 'preprocessing']
    invalid_stages = [stage for stage in stages if stage not in valid_stages]
    if invalid_stages:
        print(f"Error: Invalid stages: {invalid_stages}")
        print(f"Valid stages: {valid_stages}")
        return 1
    
    # Setup output directory
    output_path, plots_dir, metrics_dir = setup_output_directory(args.output_dir)
    print(f"Output directory: {output_path}")
    if args.save_plots:
        print(f"Plots will be saved to: {plots_dir}")
    if args.save_metrics:
        print(f"Metrics will be saved to: {metrics_dir}")
    
    # Load data for stages that need it
    G, df = None, None
    if any(stage in stages for stage in ['network', 'weights', 'degrees', 'embeddedness', 'temporal']):
        data_path = Path('..') / 'data' / 'soc-sign-bitcoinotc.csv'
        if not data_path.exists():
            data_path = Path('data') / 'soc-sign-bitcoinotc.csv'
        
        if not data_path.exists():
            print(f"Error: Data file not found at {data_path}")
            return 1
        
        print(f"Loading data from {data_path}")
        G, df = load_bitcoin_data(str(data_path))
    
    # Store all results
    all_results = {}
    
    # Run selected stages
    if 'network' in stages:
        all_results['network'] = run_network_analysis(G, output_path, args.save_plots, args.save_metrics)
    
    if 'weights' in stages:
        all_results['weights'] = run_weight_analysis(G, output_path, args.save_plots, args.save_metrics)
    
    if 'degrees' in stages:
        all_results['degrees'] = run_degree_analysis(G, output_path, args.save_plots, args.save_metrics)
    
    if 'embeddedness' in stages:
        all_results['embeddedness'] = run_embeddedness_analysis(G, output_path, args.save_plots, args.save_metrics)
    
    if 'temporal' in stages:
        all_results['temporal'] = run_temporal_analysis(G, df, output_path, args.save_plots, args.save_metrics)
    
    if 'connectivity' in stages:
        all_results['connectivity'] = run_connectivity_analysis(output_path, args.save_plots, args.save_metrics)
    
    if 'preprocessing' in stages:
        all_results['preprocessing'] = run_preprocessing_analysis(output_path, args.save_plots, args.save_metrics)
    
    # Save summary of all results
    if args.save_metrics and all_results:
        summary_file = output_path / "metrics" / "exploration_summary.json"
        save_metrics_to_json(all_results, str(summary_file))
        print(f"\nComplete exploration summary saved to {summary_file}")
    
    print("\n" + "="*60)
    print("DATA EXPLORATION COMPLETE")
    print("="*60)
    print(f"Stages completed: {', '.join(stages)}")
    print(f"Results saved to: {output_path}")
    
    return 0

if __name__ == "__main__":
    exit(main())