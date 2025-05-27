#!/usr/bin/env python3
"""
Complete Results Analysis - Enhanced Version
============================================

Generate all 6 required performance comparison plots with proper handling of different metrics formats.

Creates all required charts:
1. optimal_split_comparison.png - 74:12:14 vs standard split comparison
2. weighted_vs_unweighted_performance.png - Feature type comparison  
3. performance_summary_table.png - Complete results ranking
4. cycle_length_comparison.png - HOC3/4/5 comparison
5. embeddedness_level_comparison.png - Embeddedness 0/1/2 comparison
6. positive_ratio_impact.png - Positive ratio 10%/20%/30% impact

Usage: python notebooks/analyze_all_results.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to sys.path for src imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def safe_extract_metrics(metrics_data, metric_name, default=0.0):
    """
    Safely extract metrics from either nested or flat JSON structure
    
    Parameters:
    metrics_data: Dictionary containing metrics
    metric_name: Name of the metric to extract
    default: Default value if metric not found
    
    Returns:
    float: The metric value
    """
    if not metrics_data:
        return default
    
    # Method 1: Check if metrics are nested under 'actual'
    if 'actual' in metrics_data:
        actual_metrics = metrics_data['actual']
        if isinstance(actual_metrics, dict) and metric_name in actual_metrics:
            value = actual_metrics[metric_name]
            return float(value) if value is not None else default
    
    # Method 2: Check if metrics are at root level
    if metric_name in metrics_data:
        value = metrics_data[metric_name]
        return float(value) if value is not None else default
    
    # Method 3: Check common alternative names
    alternative_names = {
        'accuracy': ['acc', 'test_accuracy', 'best_accuracy'],
        'f1_score': ['f1', 'f1_macro', 'best_f1'],
        'roc_auc': ['auc', 'roc_auc_score', 'auc_score'],
        'precision': ['prec', 'test_precision'],
        'recall': ['rec', 'test_recall']
    }
    
    if metric_name in alternative_names:
        for alt_name in alternative_names[metric_name]:
            # Check nested
            if 'actual' in metrics_data and alt_name in metrics_data['actual']:
                value = metrics_data['actual'][alt_name]
                return float(value) if value is not None else default
            # Check root level
            if alt_name in metrics_data:
                value = metrics_data[alt_name]
                return float(value) if value is not None else default
    
    return default

def load_experiment_metrics(experiment_name):
    """
    Load metrics from validation and test results for a given experiment.
    Enhanced to handle different JSON structures and detect optimal split configuration.
    
    Parameters:
    experiment_name: Name of the experiment
    
    Returns:
    metrics: Dictionary containing validation and test metrics with split info
    """
    base_path = os.path.join(PROJECT_ROOT, 'results', experiment_name)
    
    metrics = {
        'experiment_name': experiment_name,
        'validation_metrics': None,
        'test_metrics': None,
        'config_used': None,
        'dataset_type': None,
        'dataset_info': {},
        'optimal_split': False,
        'split_type': 'standard'
    }
    
    # Load validation metrics
    val_metrics_path = os.path.join(base_path, 'validation', 'metrics.json')
    if os.path.exists(val_metrics_path):
        try:
            with open(val_metrics_path, 'r') as f:
                metrics['validation_metrics'] = json.load(f)
        except Exception as e:
            print(f"⚠️  Error loading validation metrics for {experiment_name}: {e}")
    
    # Load test metrics
    test_metrics_path = os.path.join(base_path, 'testing', 'metrics.json')
    if os.path.exists(test_metrics_path):
        try:
            with open(test_metrics_path, 'r') as f:
                metrics['test_metrics'] = json.load(f)
        except Exception as e:
            print(f"⚠️  Error loading test metrics for {experiment_name}: {e}")
    
    # Load config used
    config_path = os.path.join(base_path, 'preprocess', 'config_used.yaml')
    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                metrics['config_used'] = config_data
                
                # Check for optimal split - multiple detection methods
                optimal_split = False
                test_edges = config_data.get('num_test_edges', 0)
                val_edges = config_data.get('num_validation_edges', 0)
                
                # Method 1: Check for optimal_split flag
                if config_data.get('optimal_split', False):
                    optimal_split = True
                    print(f"✅ Detected optimal split flag in {experiment_name}")
                
                # Method 2: Check for specific edge counts (3080 test, 2640 validation)
                elif test_edges == 3080 and val_edges == 2640:
                    optimal_split = True
                    print(f"✅ Detected optimal split by edge counts in {experiment_name} (test={test_edges}, val={val_edges})")
                
                # Method 3: Check experiment name
                elif 'optimal' in experiment_name.lower():
                    optimal_split = True
                    print(f"✅ Detected optimal split by name in {experiment_name}")
                
                if optimal_split:
                    metrics['optimal_split'] = True
                    metrics['split_type'] = 'optimal'
        except Exception as e:
            print(f"⚠️  Error loading config for {experiment_name}: {e}")
    
    # Determine dataset type
    if metrics['config_used']:
        data_path = metrics['config_used'].get('data_path', '')
        enable_subset = metrics['config_used'].get('enable_subset_sampling', False)
        target_edges = metrics['config_used'].get('target_edge_count', 0)
        
        if 'epinions' in data_path.lower():
            if enable_subset:
                base_name = f'Epinions Subset (~{target_edges//1000}K edges)'
                metrics['dataset_type'] = base_name + (' (Optimal Split)' if metrics['optimal_split'] else '')
                metrics['dataset_info'] = {
                    'type': 'epinions_subset',
                    'scale': 'subset',
                    'description': 'Epinions subset (comparable to Bitcoin OTC)',
                    'target_edges': target_edges,
                    'optimal_split': metrics['optimal_split']
                }
            else:
                base_name = 'Epinions Full (841K edges)'
                metrics['dataset_type'] = base_name + (' (Optimal Split)' if metrics['optimal_split'] else '')
                metrics['dataset_info'] = {
                    'type': 'epinions_full',
                    'scale': 'full',
                    'description': 'Complete Epinions dataset',
                    'optimal_split': metrics['optimal_split']
                }
        elif 'bitcoinotc' in data_path.lower():
            base_name = 'Bitcoin OTC (35K edges)'
            metrics['dataset_type'] = base_name + (' (Optimal Split)' if metrics['optimal_split'] else '')
            metrics['dataset_info'] = {
                'type': 'bitcoin_otc',
                'scale': 'baseline',
                'description': 'Bitcoin OTC trading network (baseline)',
                'target_edges': 35000,
                'optimal_split': metrics['optimal_split']
            }
        else:
            metrics['dataset_type'] = 'Unknown'
            metrics['dataset_info'] = {'type': 'unknown', 'optimal_split': False}
    
    return metrics

def find_all_experiments():
    """
    Find all available experiments in the results directory.
    Enhanced to detect and categorize optimal split experiments.
    """
    print("🔍 Searching for all available experiments...")
    
    results_paths = [
        os.path.join(PROJECT_ROOT, 'results'),
        '../results/',
        'results/',
        '../../results/'
    ]
    
    found_experiments = {}
    optimal_experiments = []
    
    for results_path in results_paths:
        if os.path.exists(results_path):
            print(f"✅ Found results directory: {results_path}")
            
            for exp_dir in Path(results_path).iterdir():
                if exp_dir.is_dir():
                    # Check for validation and testing results
                    val_metrics = exp_dir / "validation" / "metrics.json"
                    test_metrics = exp_dir / "testing" / "metrics.json"
                    config_file = exp_dir / "preprocess" / "config_used.yaml"
                    
                    # Require either validation or testing results
                    if val_metrics.exists() or test_metrics.exists():
                        exp_name = exp_dir.name
                        found_experiments[exp_name] = str(exp_dir)
                        
                        # Quick check for optimal split experiments
                        is_optimal = False
                        
                        # Check by name
                        if 'optimal' in exp_name.lower():
                            is_optimal = True
                        
                        # Check config if available
                        if config_file.exists():
                            try:
                                import yaml
                                with open(config_file, 'r') as f:
                                    config = yaml.safe_load(f)
                                if config.get('optimal_split', False):
                                    is_optimal = True
                                elif config.get('num_test_edges') == 3080 and config.get('num_validation_edges') == 2640:
                                    is_optimal = True
                            except:
                                pass
                        
                        if is_optimal:
                            optimal_experiments.append(exp_name)
                            print(f"   🎯 Found optimal experiment: {exp_name}")
                        else:
                            print(f"   📊 Found standard experiment: {exp_name}")
            
            break  # Use first valid results directory found
    
    if not found_experiments:
        print("❌ No experiment results found!")
        return {}, []
    
    print(f"🎯 Total experiments found: {len(found_experiments)}")
    print(f"🚀 Optimal split experiments: {len(optimal_experiments)}")
    
    return found_experiments, optimal_experiments

def extract_experiment_data(all_metrics):
    """
    Extract structured data from all experiments for analysis
    Enhanced with robust metric extraction and additional feature detection
    """
    print("\n📊 Extracting experiment data for analysis...")
    
    results = []
    
    for exp_metrics in all_metrics:
        exp_name = exp_metrics['experiment_name']
        
        # Extract test metrics with safe extraction
        test_data = exp_metrics.get('test_metrics')
        if test_data:
            accuracy = safe_extract_metrics(test_data, 'accuracy')
            f1_score = safe_extract_metrics(test_data, 'f1_score')
            roc_auc = safe_extract_metrics(test_data, 'roc_auc')
            precision = safe_extract_metrics(test_data, 'precision')
            recall = safe_extract_metrics(test_data, 'recall')
            
            print(f"   📊 {exp_name}: accuracy={accuracy:.3f}, f1={f1_score:.3f}, roc_auc={roc_auc:.3f}")
        else:
            # Try validation metrics as fallback
            val_data = exp_metrics.get('validation_metrics')
            if val_data:
                accuracy = safe_extract_metrics(val_data, 'accuracy')
                f1_score = safe_extract_metrics(val_data, 'f1_score')
                roc_auc = safe_extract_metrics(val_data, 'roc_auc')
                precision = safe_extract_metrics(val_data, 'precision')
                recall = safe_extract_metrics(val_data, 'recall')
                
                print(f"   📊 {exp_name}: (validation) accuracy={accuracy:.3f}, f1={f1_score:.3f}, roc_auc={roc_auc:.3f}")
            else:
                print(f"   ⚠️  {exp_name}: No metrics available")
                accuracy = f1_score = roc_auc = precision = recall = 0.0
        
        # Extract configuration details for additional analysis
        config = exp_metrics.get('config_used', {})
        
        # Determine feature type
        use_weighted = config.get('use_weighted_features', False)
        weight_method = config.get('weight_method', 'sign')
        
        if use_weighted:
            if weight_method == 'raw':
                feature_type = "Weighted (Raw)"
            elif weight_method == 'binned':
                feature_type = "Weighted (Binned)"
            else:
                feature_type = "Weighted (Sign)"
        else:
            if weight_method == 'sign' or not weight_method:
                feature_type = "Binary (Sign)"
            else:
                feature_type = f"Unknown ({weight_method})"
        
        # Extract cycle length (for HOC analysis)
        cycle_length = config.get('cycle_length', 4)
        
        # Extract embeddedness settings
        min_train_embeddedness = config.get('min_train_embeddedness', 0)
        min_test_embeddedness = config.get('min_test_embeddedness', 0)
        # Use the higher of the two for classification
        embeddedness_level = max(min_train_embeddedness, min_test_embeddedness)
        
        # Extract positive ratio settings
        pos_train_ratio = config.get('pos_train_edges_ratio', 0.8)
        pos_test_ratio = config.get('pos_test_edges_ratio', 0.8)
        # Use training ratio as primary indicator
        positive_ratio = pos_train_ratio
        
        # Determine dataset and split type
        dataset_type = exp_metrics.get('dataset_type', 'Unknown')
        optimal_split = exp_metrics.get('optimal_split', False)
        split_indicator = "🎯" if optimal_split else ""
        
        results.append({
            'experiment_name': exp_name,
            'display_name': f"{split_indicator} {exp_name}" if split_indicator else exp_name,
            'accuracy': accuracy,
            'f1_score': f1_score,
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'feature_type': feature_type,
            'dataset_type': dataset_type,
            'optimal_split': optimal_split,
            'split_type': 'optimal' if optimal_split else 'standard',
            # NEW: Additional analysis dimensions
            'cycle_length': cycle_length,
            'embeddedness_level': embeddedness_level,
            'positive_ratio': positive_ratio,
            'use_weighted': use_weighted,
            'weight_method': weight_method
        })
    
    return pd.DataFrame(results)

def create_optimal_split_comparison(df, output_dir):
    """
    Create comparison plot showing optimal split vs standard split performance
    """
    print("\n🎯 Creating optimal split comparison...")
    
    # Separate optimal and standard experiments
    optimal_df = df[df['optimal_split'] == True].copy()
    standard_df = df[df['optimal_split'] == False].copy()
    
    print(f"📊 Optimal Split Performance Analysis:")
    print(f"Standard Split Experiments: {len(standard_df)}")
    print(f"Optimal Split Experiments: {len(optimal_df)}")
    
    if len(optimal_df) == 0:
        print("⚠️  No optimal split experiments found - skipping comparison")
        return None, None
    
    if len(standard_df) == 0:
        print("⚠️  No standard split experiments found - skipping comparison")
        return None, None
    
    # Calculate average performance for each group
    metrics = ['accuracy', 'f1_score', 'roc_auc', 'precision']
    standard_avg = standard_df[metrics].mean()
    optimal_avg = optimal_df[metrics].mean()
    
    # Show the comparison
    for metric in metrics:
        std_val = standard_avg[metric]
        opt_val = optimal_avg[metric]
        improvement = ((opt_val - std_val) / std_val * 100) if std_val > 0 else 0
        print(f"{metric.title()}: {std_val:.3f} → {opt_val:.3f} ({improvement:+.1f}%)")
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Optimal Split (74:12:14) vs Standard Split Performance Comparison', fontsize=16, fontweight='bold')
    
    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        
        # Get values for plotting
        std_values = standard_df[metric].values
        opt_values = optimal_df[metric].values
        
        # Create violin plot
        try:
            parts = ax.violinplot([std_values, opt_values], positions=[1, 2], widths=0.6)
            
            # Color the violin plots
            parts['bodies'][0].set_facecolor('lightblue')
            parts['bodies'][1].set_facecolor('orange')
        except:
            # Fallback to box plot if violin plot fails
            ax.boxplot([std_values, opt_values], positions=[1, 2])
        
        # Add individual points
        for j, val in enumerate(std_values):
            ax.scatter(1, val, c='darkblue', alpha=0.6, s=30)
        for j, val in enumerate(opt_values):
            ax.scatter(2, val, c='darkorange', alpha=0.6, s=30)
        
        # Add mean lines
        ax.axhline(y=standard_avg[metric], xmin=0.15, xmax=0.45, color='darkblue', linewidth=3, 
                   label=f'Standard Avg: {standard_avg[metric]:.3f}')
        ax.axhline(y=optimal_avg[metric], xmin=0.55, xmax=0.85, color='darkorange', linewidth=3, 
                   label=f'Optimal Avg: {optimal_avg[metric]:.3f}')
        
        ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Standard Split', 'Optimal Split (74:12:14)'])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Add improvement text
        improvement = ((optimal_avg[metric] - standard_avg[metric]) / standard_avg[metric] * 100) if standard_avg[metric] > 0 else 0
        ax.text(1.5, ax.get_ylim()[1] * 0.9, f'Improvement: {improvement:+.1f}%', 
                ha='center', fontweight='bold', 
                color='green' if improvement > 0 else 'red')
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, 'optimal_split_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Optimal split comparison plot saved to {output_path}")
    
    return optimal_avg, standard_avg

def create_weighted_vs_unweighted_comparison(df, output_dir):
    """
    Create comparison plot for weighted vs unweighted features
    """
    print("\n📊 Creating weighted vs unweighted performance comparison...")
    
    # Filter relevant feature types
    weighted_df = df[df['feature_type'].str.contains('Weighted', na=False)]
    binary_df = df[df['feature_type'].str.contains('Binary', na=False)]
    
    if len(weighted_df) == 0 or len(binary_df) == 0:
        print("⚠️  Insufficient data for weighted vs unweighted comparison")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Weighted vs Unweighted Features Performance', fontsize=16, fontweight='bold')
    
    metrics = ['accuracy', 'f1_score', 'roc_auc', 'precision']
    
    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        
        # Create box plot
        data_to_plot = [binary_df[metric].values, weighted_df[metric].values]
        box_plot = ax.boxplot(data_to_plot, labels=['Binary (Unweighted)', 'Weighted'], patch_artist=True)
        
        # Color the boxes
        box_plot['boxes'][0].set_facecolor('lightblue')
        box_plot['boxes'][1].set_facecolor('lightcoral')
        
        # Add individual points
        for j, val in enumerate(binary_df[metric].values):
            ax.scatter(1, val, c='darkblue', alpha=0.6, s=30)
        for j, val in enumerate(weighted_df[metric].values):
            ax.scatter(2, val, c='darkred', alpha=0.6, s=30)
        
        ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        binary_mean = binary_df[metric].mean()
        weighted_mean = weighted_df[metric].mean()
        
        ax.text(0.5, ax.get_ylim()[1] * 0.9, f'Binary Avg: {binary_mean:.3f}', 
                transform=ax.transAxes, ha='left')
        ax.text(0.5, ax.get_ylim()[1] * 0.85, f'Weighted Avg: {weighted_mean:.3f}', 
                transform=ax.transAxes, ha='left')
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, 'weighted_vs_unweighted_performance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Weighted vs unweighted performance plot saved to {output_path}")

def create_cycle_length_comparison(df, output_dir):
    """
    Create comparison plot for different cycle lengths (HOC3, HOC4, HOC5)
    """
    print("\n🔄 Creating cycle length comparison...")
    
    # Group by cycle length
    cycle_groups = df.groupby('cycle_length')
    
    if len(cycle_groups) <= 1:
        print("⚠️  Insufficient cycle length variations - skipping comparison")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Impact of Cycle Length (HOC3 vs HOC4 vs HOC5) on Performance', fontsize=16, fontweight='bold')
    
    metrics = ['accuracy', 'f1_score', 'roc_auc', 'precision']
    
    # Prepare data for plotting
    cycle_lengths = sorted([length for length, _ in cycle_groups])
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
    
    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        
        # Prepare data for box plot
        data_to_plot = []
        labels = []
        
        for cycle_length in cycle_lengths:
            group_data = cycle_groups.get_group(cycle_length)
            if len(group_data) > 0:
                data_to_plot.append(group_data[metric].values)
                labels.append(f'HOC{cycle_length}')
        
        if len(data_to_plot) > 1:
            # Create box plot
            box_plot = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            # Color the boxes
            for j, box in enumerate(box_plot['boxes']):
                box.set_facecolor(colors[j % len(colors)])
            
            # Add individual points
            for j, data in enumerate(data_to_plot):
                for val in data:
                    ax.scatter(j+1, val, c='darkred', alpha=0.6, s=30)
            
            # Add mean values as text
            for j, (cycle_length, data) in enumerate(zip(cycle_lengths, data_to_plot)):
                if len(data) > 0:
                    mean_val = np.mean(data)
                    ax.text(j+1, ax.get_ylim()[1] * 0.95, f'μ={mean_val:.3f}', 
                           ha='center', va='top', fontweight='bold')
        
        ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
        ax.set_ylabel('Score')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, 'cycle_length_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Cycle length comparison plot saved to {output_path}")

def create_embeddedness_level_comparison(df, output_dir):
    """
    Create comparison plot for different embeddedness levels (0 vs 1 vs 2)
    """
    print("\n🔗 Creating embeddedness level comparison...")
    
    # Group by embeddedness level
    embed_groups = df.groupby('embeddedness_level')
    
    if len(embed_groups) <= 1:
        print("⚠️  Insufficient embeddedness level variations - skipping comparison")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Impact of Embeddedness Filtering (Level 0 vs 1 vs 2) on Performance', fontsize=16, fontweight='bold')
    
    # Focus on accuracy and false positive rate as mentioned in requirements
    metrics = ['accuracy', 'precision']  # Using precision as proxy for false positive rate impact
    
    # Prepare data for plotting
    embed_levels = sorted([level for level, _ in embed_groups])
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Prepare data for box plot
        data_to_plot = []
        labels = []
        
        for embed_level in embed_levels:
            group_data = embed_groups.get_group(embed_level)
            if len(group_data) > 0:
                data_to_plot.append(group_data[metric].values)
                labels.append(f'Embeddedness ≥{embed_level}')
        
        if len(data_to_plot) > 1:
            # Create box plot
            box_plot = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            # Color the boxes
            for j, box in enumerate(box_plot['boxes']):
                box.set_facecolor(colors[j % len(colors)])
            
            # Add individual points
            for j, data in enumerate(data_to_plot):
                for val in data:
                    ax.scatter(j+1, val, c='darkred', alpha=0.6, s=30)
            
            # Add mean values and sample counts
            for j, (embed_level, data) in enumerate(zip(embed_levels, data_to_plot)):
                if len(data) > 0:
                    mean_val = np.mean(data)
                    count = len(data)
                    ax.text(j+1, ax.get_ylim()[1] * 0.95, f'μ={mean_val:.3f}\nn={count}', 
                           ha='center', va='top', fontweight='bold')
        
        ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
        ax.set_ylabel('Score')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, 'embeddedness_level_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Embeddedness level comparison plot saved to {output_path}")

def create_positive_ratio_impact(df, output_dir):
    """
    Create comparison plot for different positive ratios (10% vs 20% vs 30%)
    """
    print("\n📊 Creating positive ratio impact comparison...")
    
    # Convert positive ratio to percentage categories
    df_copy = df.copy()
    df_copy['pos_ratio_category'] = df_copy['positive_ratio'].apply(lambda x: 
        '10%' if 0.05 <= x <= 0.15 else
        '20%' if 0.15 < x <= 0.25 else
        '30%' if 0.25 < x <= 0.35 else
        '80%' if 0.75 <= x <= 0.85 else
        f'{x:.0%}'
    )
    
    # Group by positive ratio category
    ratio_groups = df_copy.groupby('pos_ratio_category')
    
    if len(ratio_groups) <= 1:
        print("⚠️  Insufficient positive ratio variations - skipping comparison")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Impact of Positive Edge Ratio on Performance', fontsize=16, fontweight='bold')
    
    metrics = ['accuracy', 'f1_score', 'roc_auc', 'precision']
    
    # Prepare data for plotting
    ratio_categories = sorted([cat for cat, _ in ratio_groups])
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
    
    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        
        # Prepare data for box plot
        data_to_plot = []
        labels = []
        
        for ratio_cat in ratio_categories:
            group_data = ratio_groups.get_group(ratio_cat)
            if len(group_data) > 0:
                data_to_plot.append(group_data[metric].values)
                labels.append(f'{ratio_cat} Positive')
        
        if len(data_to_plot) > 1:
            # Create box plot
            box_plot = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            # Color the boxes
            for j, box in enumerate(box_plot['boxes']):
                box.set_facecolor(colors[j % len(colors)])
            
            # Add individual points
            for j, data in enumerate(data_to_plot):
                for val in data:
                    ax.scatter(j+1, val, c='darkred', alpha=0.6, s=30)
            
            # Add mean values and sample counts
            for j, (ratio_cat, data) in enumerate(zip(ratio_categories, data_to_plot)):
                if len(data) > 0:
                    mean_val = np.mean(data)
                    count = len(data)
                    ax.text(j+1, ax.get_ylim()[1] * 0.95, f'μ={mean_val:.3f}\nn={count}', 
                           ha='center', va='top', fontweight='bold')
        
        ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
        ax.set_ylabel('Score')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, 'positive_ratio_impact.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Positive ratio impact plot saved to {output_path}")

def create_performance_summary_table(df, output_dir):
    """
    Create a comprehensive performance summary table
    """
    print("\n📋 Creating performance summary table...")
    
    # Sort by accuracy descending
    df_sorted = df.sort_values('accuracy', ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    headers = ['Rank', 'Experiment', 'Accuracy', 'F1 Score', 'ROC AUC', 'Precision', 'Feature Type', 'Split Type', 'HOC', 'Embed']
    
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        table_data.append([
            i + 1,
            row['display_name'][:25] + '...' if len(row['display_name']) > 25 else row['display_name'],
            f"{row['accuracy']:.3f}",
            f"{row['f1_score']:.3f}",
            f"{row['roc_auc']:.3f}",
            f"{row['precision']:.3f}",
            row['feature_type'],
            row['split_type'].title(),
            f"HOC{row['cycle_length']}",
            f"≥{row['embeddedness_level']}"
        ])
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    
    # Color rows based on split type
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        row_idx = i + 1  # +1 because of header
        if row['optimal_split']:
            # Highlight optimal split experiments
            for j in range(len(headers)):
                table[(row_idx, j)].set_facecolor('#FFE6CC')  # Light orange
        else:
            # Standard experiments
            for j in range(len(headers)):
                table[(row_idx, j)].set_facecolor('#F0F0F0')  # Light gray
    
    # Color header
    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#4CAF50')  # Green
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    plt.title('Complete Performance Summary Table\n(🎯 indicates Optimal Split experiments)', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Save plot
    output_path = os.path.join(output_dir, 'performance_summary_table.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Performance summary table saved to {output_path}")

def generate_analysis_report(df, optimal_avg=None, standard_avg=None, output_dir=None):
    """
    Generate a comprehensive markdown report
    """
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, 'plots', 'complete_results_analysis')
    
    report_path = os.path.join(output_dir, 'complete_results_analysis_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Complete Results Analysis Report\n\n")
        f.write("## Executive Summary\n\n")
        f.write(f"- **Total Experiments Analyzed**: {len(df)}\n")
        f.write(f"- **Best Overall Accuracy**: {df['accuracy'].max():.3f}\n")
        f.write(f"- **Best Overall F1 Score**: {df['f1_score'].max():.3f}\n")
        f.write(f"- **Best Overall ROC AUC**: {df['roc_auc'].max():.3f}\n\n")
        
        # Optimal split analysis
        optimal_df = df[df['optimal_split'] == True]
        standard_df = df[df['optimal_split'] == False]
        
        if len(optimal_df) > 0:
            f.write("## Optimal Split Analysis (74:12:14)\n\n")
            f.write(f"- **Optimal Split Experiments**: {len(optimal_df)}\n")
            f.write(f"- **Standard Split Experiments**: {len(standard_df)}\n")
            f.write(f"- **Best Optimal Split Accuracy**: {optimal_df['accuracy'].max():.3f}\n")
            f.write(f"- **Average Optimal Split Accuracy**: {optimal_df['accuracy'].mean():.3f}\n")
            
            if len(standard_df) > 0 and optimal_avg is not None and standard_avg is not None:
                f.write("\n### Performance Improvements with Optimal Split:\n\n")
                for metric in ['accuracy', 'f1_score', 'roc_auc', 'precision']:
                    std_val = standard_avg[metric]
                    opt_val = optimal_avg[metric]
                    improvement = ((opt_val - std_val) / std_val * 100) if std_val > 0 else 0
                    f.write(f"- **{metric.title()}**: {std_val:.3f} → {opt_val:.3f} ({improvement:+.1f}%)\n")
        
        # Feature type analysis
        f.write("\n## Feature Type Analysis\n\n")
        feature_summary = df.groupby('feature_type').agg({
            'accuracy': ['count', 'max', 'mean'],
            'f1_score': ['max', 'mean'],
            'roc_auc': ['max', 'mean']
        }).round(3)
        
        for feature_type in feature_summary.index:
            count = feature_summary.loc[feature_type, ('accuracy', 'count')]
            max_acc = feature_summary.loc[feature_type, ('accuracy', 'max')]
            avg_acc = feature_summary.loc[feature_type, ('accuracy', 'mean')]
            f.write(f"- **{feature_type}**: {count} experiments, max accuracy={max_acc}, avg accuracy={avg_acc}\n")
        
        # Cycle length analysis
        f.write("\n## Cycle Length Analysis (HOC)\n\n")
        cycle_summary = df.groupby('cycle_length').agg({
            'accuracy': ['count', 'max', 'mean'],
            'f1_score': ['max', 'mean']
        }).round(3)
        
        for cycle_length in cycle_summary.index:
            count = cycle_summary.loc[cycle_length, ('accuracy', 'count')]
            max_acc = cycle_summary.loc[cycle_length, ('accuracy', 'max')]
            avg_acc = cycle_summary.loc[cycle_length, ('accuracy', 'mean')]
            f.write(f"- **HOC{cycle_length}**: {count} experiments, max accuracy={max_acc}, avg accuracy={avg_acc}\n")
        
        # Embeddedness analysis
        f.write("\n## Embeddedness Level Analysis\n\n")
        embed_summary = df.groupby('embeddedness_level').agg({
            'accuracy': ['count', 'max', 'mean'],
            'precision': ['max', 'mean']
        }).round(3)
        
        for embed_level in embed_summary.index:
            count = embed_summary.loc[embed_level, ('accuracy', 'count')]
            max_acc = embed_summary.loc[embed_level, ('accuracy', 'max')]
            avg_acc = embed_summary.loc[embed_level, ('accuracy', 'mean')]
            f.write(f"- **Embeddedness ≥{embed_level}**: {count} experiments, max accuracy={max_acc}, avg accuracy={avg_acc}\n")
        
        f.write("\n## Top Performing Experiments\n\n")
        top_experiments = df.nlargest(5, 'accuracy')
        for i, (_, row) in enumerate(top_experiments.iterrows()):
            split_indicator = "🎯 " if row['optimal_split'] else ""
            f.write(f"{i+1}. {split_indicator}**{row['experiment_name']}**: {row['accuracy']:.3f} accuracy, {row['f1_score']:.3f} F1, HOC{row['cycle_length']}, Embed≥{row['embeddedness_level']}\n")
        
        f.write(f"\n## Generated Plots\n\n")
        f.write("- `optimal_split_comparison.png` - Comparison of optimal (74:12:14) vs standard split performance\n")
        f.write("- `weighted_vs_unweighted_performance.png` - Feature type comparison\n")
        f.write("- `cycle_length_comparison.png` - HOC3 vs HOC4 vs HOC5 performance\n")
        f.write("- `embeddedness_level_comparison.png` - Embeddedness filtering impact\n")
        f.write("- `positive_ratio_impact.png` - Effect of positive edge ratios\n")
        f.write("- `performance_summary_table.png` - Complete results table\n")
        
        f.write(f"\n---\n*Report generated automatically by the results analysis pipeline*\n")
    
    print(f"📄 Comprehensive results report saved to {report_path}")

def main():
    """
    Main function to run complete results analysis with all 6 charts
    """
    print("📈 COMPLETE RESULTS ANALYSIS - All 6 Required Charts")
    print("="*80)
    print("Creating all required performance plots:")
    print("1. optimal_split_comparison.png")
    print("2. weighted_vs_unweighted_performance.png") 
    print("3. performance_summary_table.png")
    print("4. cycle_length_comparison.png")
    print("5. embeddedness_level_comparison.png")
    print("6. positive_ratio_impact.png")
    print()
    
    # Create output directory with absolute path
    output_dir = os.path.abspath(os.path.join(PROJECT_ROOT, 'plots', 'complete_results_analysis'))
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📁 Output directory: {output_dir}")
    
    # Find all experiments
    found_experiments, optimal_experiments = find_all_experiments()
    
    if not found_experiments:
        print("❌ No experiments found. Please run some experiments first.")
        return
    
    # Load all experiment metrics
    print(f"\n📊 Loading all experiment data...")
    all_metrics = []
    
    for exp_name in found_experiments.keys():
        print(f"   Loading metrics for {exp_name}...")
        metrics = load_experiment_metrics(exp_name)
        all_metrics.append(metrics)
        
        # Show loading status
        split_status = " (OPTIMAL)" if metrics['optimal_split'] else ""
        print(f"   ✅ Loaded: {exp_name}{split_status}")
    
    print(f"📋 Successfully loaded {len(all_metrics)} experiments")
    print(f"🚀 Including {len(optimal_experiments)} optimal split experiments")
    
    # Extract structured data
    df = extract_experiment_data(all_metrics)
    
    if df.empty:
        print("❌ No valid experiment data found")
        return
    
    print(f"\n📊 Analyzing {len(df)} experiments...")
    
    # Create all 6 required charts
    print(f"\n🎯 Generating all required performance charts...")
    
    # 1. Optimal split comparison (REQUIRED)
    optimal_avg, standard_avg = create_optimal_split_comparison(df, output_dir)
    
    # 2. Weighted vs unweighted comparison (REQUIRED)
    create_weighted_vs_unweighted_comparison(df, output_dir)
    
    # 3. Performance summary table (REQUIRED)
    create_performance_summary_table(df, output_dir)
    
    # 4. Cycle length comparison (STRONGLY RECOMMENDED)
    create_cycle_length_comparison(df, output_dir)
    
    # 5. Embeddedness level comparison (STRONGLY RECOMMENDED)
    create_embeddedness_level_comparison(df, output_dir)
    
    # 6. Positive ratio impact (STRONGLY RECOMMENDED)
    create_positive_ratio_impact(df, output_dir)
    
    # Generate comprehensive report
    generate_analysis_report(df, optimal_avg, standard_avg, output_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("COMPLETE RESULTS ANALYSIS SUMMARY")
    print("="*80)
    print(f"📊 Analyzed {len(df)} experiments")
    print(f"📈 Best accuracy: {df['accuracy'].max():.3f}")
    print(f"📈 Best F1 score: {df['f1_score'].max():.3f}")
    print(f"📈 Best ROC AUC: {df['roc_auc'].max():.3f}")
    
    # Optimal split summary
    optimal_df = df[df['optimal_split'] == True]
    if len(optimal_df) > 0:
        print(f"\n🎯 Optimal Split Analysis:")
        print(f"   Experiments: {len(optimal_df)}")
        print(f"   Best accuracy: {optimal_df['accuracy'].max():.3f}")
        print(f"   Avg accuracy: {optimal_df['accuracy'].mean():.3f}")
        
        standard_df = df[df['optimal_split'] == False]
        if len(standard_df) > 0 and optimal_avg is not None and standard_avg is not None:
            improvement = ((optimal_avg['accuracy'] - standard_avg['accuracy']) / standard_avg['accuracy'] * 100) if standard_avg['accuracy'] > 0 else 0
            print(f"   Improvement over standard: {improvement:+.1f}%")
    
    # Feature type summary
    print(f"\n🔍 Analysis Dimensions Summary:")
    print(f"   Feature types: {len(df['feature_type'].unique())}")
    print(f"   Cycle lengths: {sorted(df['cycle_length'].unique())}")
    print(f"   Embeddedness levels: {sorted(df['embeddedness_level'].unique())}")
    print(f"   Positive ratios: {sorted(df['positive_ratio'].unique())}")
    
    print(f"\n🎯 ALL 6 REQUIRED CHARTS GENERATED!")
    print(f"📁 All plots saved to: {output_dir}")
    
    print(f"\n📋 FINAL CHART LIST:")
    expected_files = [
        'optimal_split_comparison.png',
        'weighted_vs_unweighted_performance.png',
        'performance_summary_table.png', 
        'cycle_length_comparison.png',
        'embeddedness_level_comparison.png',
        'positive_ratio_impact.png'
    ]
    
    for filename in expected_files:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"✅ {filename}: {size:,} bytes")
        else:
            print(f"❌ {filename}: NOT FOUND")
    
    print(f"\n🎉 READY FOR PRESENTATION WITH ALL REQUIRED ANALYSIS!")

if __name__ == "__main__":
    main()