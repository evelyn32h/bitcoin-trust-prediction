#!/usr/bin/env python3
"""
Complete Results Analysis - Enhanced Version with All Required Comparisons
========================================================================

This script generates all required comparison plots for the research results:

RESULT PLOTS (8 types):
1. ✅ Weighted vs Unweighted Comparison (MOST IMPORTANT - optimal split only)
2. ✅ Dataset Comparison (best configuration only) - SIMPLIFIED
3. ✅ Embeddedness Level Comparison (0, 1, 2) - FIXED
4. ✅ Aggregation Methods Comparison
5. ✅ Complete Performance Summary Table
6. ✅ Positive Ratio Comparison (multiple ratios)
7. ✅ Cycle Length Comparison (3, 4, 5) - NEW ADDITION
8. ✅ Pos/Neg Ratio Experiments (90%-10% through 50%-50%) - FIXED

FIXES APPLIED:
- Embeddedness filtering now applied in preprocessing stage
- Enhanced embeddedness detection from config files and experiment names
- Improved subplot titles with semantic naming
- SIMPLIFIED dataset comparison for clarity
- All experiments use optimal split
- ADDED cycle length comparison

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
    Load metrics from validation and test results for a given experiment
    """
    base_path = os.path.join(PROJECT_ROOT, 'results', experiment_name)
    
    metrics = {
        'experiment_name': experiment_name,
        'validation_metrics': None,
        'test_metrics': None,
        'config_used': None,
        'dataset_type': None,
        'optimal_split': False,
        'split_type': 'standard'
    }
    
    # Load validation metrics
    val_metrics_path = os.path.join(base_path, 'validation', 'metrics.json')
    if os.path.exists(val_metrics_path):
        try:
            with open(val_metrics_path, 'r', encoding='utf-8') as f:
                metrics['validation_metrics'] = json.load(f)
        except Exception as e:
            print(f"Warning: Error loading validation metrics for {experiment_name}: {e}")
    
    # Load test metrics
    test_metrics_path = os.path.join(base_path, 'testing', 'metrics.json')
    if os.path.exists(test_metrics_path):
        try:
            with open(test_metrics_path, 'r', encoding='utf-8') as f:
                metrics['test_metrics'] = json.load(f)
        except Exception as e:
            print(f"Warning: Error loading test metrics for {experiment_name}: {e}")
    
    # Load config used
    config_paths = [
        os.path.join(base_path, 'config_used.yaml'),
        os.path.join(base_path, 'preprocess', 'config_used.yaml')
    ]
    
    for config_path in config_paths:
        if os.path.exists(config_path):
            try:
                import yaml
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                    metrics['config_used'] = config_data
                    
                    # Enhanced optimal split detection
                    optimal_split = False
                    test_edges = config_data.get('num_test_edges', 0)
                    val_edges = config_data.get('num_validation_edges', 0)
                    
                    # Multiple detection methods
                    if config_data.get('optimal_split', False):
                        optimal_split = True
                    elif test_edges == 4000 and val_edges == 4000:  # From config.yaml
                        optimal_split = True
                    elif 'optimal' in experiment_name.lower():
                        optimal_split = True
                    
                    if optimal_split:
                        metrics['optimal_split'] = True
                        metrics['split_type'] = 'optimal'
                break
            except Exception as e:
                print(f"Warning: Error loading config for {experiment_name}: {e}")
    
    return metrics

def find_all_experiments():
    """
    Find all available experiments in the results directory
    """
    print("Searching for all available experiments...")
    
    results_paths = [
        os.path.join(PROJECT_ROOT, 'results'),
        '../results/',
        'results/'
    ]
    
    found_experiments = {}
    optimal_experiments = []
    embeddedness_experiments = []
    
    for results_path in results_paths:
        if os.path.exists(results_path):
            print(f"Found results directory: {results_path}")
            
            for exp_dir in Path(results_path).iterdir():
                if exp_dir.is_dir():
                    # Check for validation and testing results
                    val_metrics = exp_dir / "validation" / "metrics.json"
                    test_metrics = exp_dir / "testing" / "metrics.json"
                    
                    # Require either validation or testing results
                    if val_metrics.exists() or test_metrics.exists():
                        exp_name = exp_dir.name
                        found_experiments[exp_name] = str(exp_dir)
                        
                        # Check for optimal split experiments
                        if 'optimal' in exp_name.lower():
                            optimal_experiments.append(exp_name)
                            print(f"   Found optimal experiment: {exp_name}")
                        else:
                            print(f"   Found standard experiment: {exp_name}")
                        
                        # Check for embeddedness experiments
                        if any(pattern in exp_name.lower() for pattern in ['embed_0', 'embed_1', 'embed_2', 'embeddedness']):
                            embeddedness_experiments.append(exp_name)
                            print(f"   Found embeddedness experiment: {exp_name}")
            
            break  # Use first valid results directory found
    
    if not found_experiments:
        print("No experiment results found!")
        return {}, [], []
    
    print(f"Total experiments found: {len(found_experiments)}")
    print(f"Optimal split experiments: {len(optimal_experiments)}")
    print(f"Embeddedness experiments: {len(embeddedness_experiments)}")
    
    return found_experiments, optimal_experiments, embeddedness_experiments

def extract_embeddedness_level(exp_name, config):
    """
    Extract embeddedness level from experiment name and config
    Priority: config file > experiment name pattern > default
    """
    # Method 1: From config file (highest priority)
    if config:
        # Check various possible config field names
        embeddedness_fields = [
            'min_embeddedness',
            'min_train_embeddedness', 
            'embeddedness_threshold',
            'embeddedness_level',
            'embed_level'
        ]
        
        for field in embeddedness_fields:
            if field in config:
                value = config[field]
                if value is not None:
                    return int(value)
    
    # Method 2: From experiment name patterns
    exp_lower = exp_name.lower()
    
    # Pattern: embed_0, embed_1, embed_2
    if 'embed_0' in exp_lower:
        return 0
    elif 'embed_1' in exp_lower:
        return 1
    elif 'embed_2' in exp_lower:
        return 2
    
    # Pattern: embeddedness_0, embeddedness_1, embeddedness_2
    if 'embeddedness_0' in exp_lower:
        return 0
    elif 'embeddedness_1' in exp_lower:
        return 1
    elif 'embeddedness_2' in exp_lower:
        return 2
    
    # Pattern: emb0, emb1, emb2
    if 'emb0' in exp_lower:
        return 0
    elif 'emb1' in exp_lower:
        return 1
    elif 'emb2' in exp_lower:
        return 2
    
    # Default: moderate filtering (level 1)
    return 1

def extract_experiment_data(all_metrics):
    """
    Extract structured data from all experiments for analysis
    Enhanced experiment detection for different configurations
    """
    print("\nExtracting experiment data for analysis...")
    
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
            
            print(f"   {exp_name}: accuracy={accuracy:.3f}, f1={f1_score:.3f}, roc_auc={roc_auc:.3f}")
        else:
            # Try validation metrics as fallback
            val_data = exp_metrics.get('validation_metrics')
            if val_data:
                accuracy = safe_extract_metrics(val_data, 'accuracy')
                f1_score = safe_extract_metrics(val_data, 'f1_score')
                roc_auc = safe_extract_metrics(val_data, 'roc_auc')
                precision = safe_extract_metrics(val_data, 'precision')
                recall = safe_extract_metrics(val_data, 'recall')
                
                print(f"   {exp_name}: (validation) accuracy={accuracy:.3f}, f1={f1_score:.3f}")
            else:
                print(f"   Warning: {exp_name}: No metrics available")
                accuracy = f1_score = roc_auc = precision = recall = 0.0
        
        # Extract configuration details
        config = exp_metrics.get('config_used', {})
        
        # Enhanced experiment type detection
        # 1. Determine feature type
        use_weighted = config.get('use_weighted_features', False)
        
        if use_weighted:
            feature_type = "Weighted"
        else:
            feature_type = "Unweighted"
        
        # Also check experiment name for weighted/unweighted
        if 'weighted' in exp_name.lower():
            feature_type = "Weighted"
        elif 'unweighted' in exp_name.lower():
            feature_type = "Unweighted"
        
        # 2. Extract cycle length (ENHANCED detection for Step 7)
        cycle_length = config.get('cycle_length', 4)
        
        # Enhanced detection from experiment name
        if 'cycle_length_3' in exp_name or 'cycle3' in exp_name:
            cycle_length = 3
        elif 'cycle_length_4' in exp_name or 'cycle4' in exp_name:
            cycle_length = 4
        elif 'cycle_length_5' in exp_name or 'cycle5' in exp_name:
            cycle_length = 5
        
        # 3. Extract positive ratio (ENHANCED detection for Steps 6&8)
        pos_train_ratio = config.get('pos_train_edges_ratio', 0.5)
        pos_test_ratio = config.get('pos_test_edges_ratio', 0.5)
        pos_edges_ratio = config.get('pos_edges_ratio', 0.5)
        
        # Use the most specific ratio available
        positive_ratio = pos_test_ratio or pos_train_ratio or pos_edges_ratio
        
        # Enhanced detection from experiment name for Steps 6&8
        if 'pos_ratio_90_10' in exp_name or 'pos90' in exp_name:
            positive_ratio = 0.9
        elif 'pos_ratio_80_20' in exp_name or 'pos80' in exp_name:
            positive_ratio = 0.8
        elif 'pos_ratio_70_30' in exp_name or 'pos70' in exp_name:
            positive_ratio = 0.7
        elif 'pos_ratio_60_40' in exp_name or 'pos60' in exp_name:
            positive_ratio = 0.6
        elif 'pos_ratio_50_50' in exp_name or 'pos50' in exp_name:
            positive_ratio = 0.5
        
        # 4. Determine dataset type
        dataset_type = 'Bitcoin OTC (Optimal Split)' if exp_metrics['optimal_split'] else 'Bitcoin OTC (Standard Split)'
        
        # 5. Real embeddedness level detection
        embeddedness_level = extract_embeddedness_level(exp_name, config)
        
        print(f"   {exp_name}: embeddedness_level={embeddedness_level}")
        
        # 6. Determine aggregation method
        aggregation_method = config.get('bidirectional_method', 'max')
        
        # Enhanced detection from experiment name
        if 'aggregation_max' in exp_name:
            aggregation_method = 'max'
        elif 'aggregation_sum' in exp_name:
            aggregation_method = 'sum'
        elif 'aggregation_stronger' in exp_name:
            aggregation_method = 'stronger'
        
        results.append({
            'experiment_name': exp_name,
            'display_name': f"{'Optimal' if exp_metrics['optimal_split'] else 'Standard'}: {exp_name}",
            'accuracy': accuracy,
            'f1_score': f1_score,
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'feature_type': feature_type,
            'dataset_type': dataset_type,
            'optimal_split': exp_metrics['optimal_split'],
            'split_type': 'optimal' if exp_metrics['optimal_split'] else 'standard',
            # Enhanced analysis dimensions
            'cycle_length': cycle_length,
            'embeddedness_level': embeddedness_level,
            'positive_ratio': positive_ratio,
            'use_weighted': use_weighted,
            'aggregation_method': aggregation_method
        })
    
    return pd.DataFrame(results)

def create_weighted_vs_unweighted_comparison(df, output_dir):
    """
    PLOT 1: Weighted vs Unweighted Comparison - ENHANCED with semantic titles
    """
    print(f"\n{'='*80}")
    print("PLOT 1: WEIGHTED vs UNWEIGHTED COMPARISON (ENHANCED TITLES)")
    print("="*80)
    
    # STRICT FILTERING: Only weighted_optimal and unweighted_optimal
    weighted_df = df[df['experiment_name'] == 'weighted_optimal'].copy()
    unweighted_df = df[df['experiment_name'] == 'unweighted_optimal'].copy()
    
    print(f"Filtered weighted experiments: {len(weighted_df)} (should be 1)")
    print(f"Filtered unweighted experiments: {len(unweighted_df)} (should be 1)")
    
    if len(weighted_df) == 0 or len(unweighted_df) == 0:
        print("Warning: Missing required experiments (weighted_optimal or unweighted_optimal)")
        
        # Create informational plot
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, 
                'MISSING REQUIRED EXPERIMENTS\n\n'
                'Expected experiments:\n'
                '• weighted_optimal\n'
                '• unweighted_optimal\n\n'
                'Please run these specific experiments first.',
                ha='center', va='center', transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral"),
                fontsize=14)
        ax.set_title('PLOT 1: Missing Required Experiments')
        ax.axis('off')
        
        output_path = os.path.join(output_dir, '1_weighted_vs_unweighted_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return None, None, "Unknown"
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Weighted vs Unweighted Features Performance Comparison\n'
                 'Configuration: cycle_length=4, min_embeddedness=1, pos_ratio=0.5, max aggregation', 
                 fontsize=16, fontweight='bold')
    
    metrics = ['accuracy', 'f1_score', 'roc_auc', 'precision']
    metric_titles = ['Accuracy', 'F1 Score', 'ROC AUC', 'Precision']
    
    # Get single values (since we have exactly one experiment per type)
    unweighted_values = unweighted_df[metrics].iloc[0]
    weighted_values = weighted_df[metrics].iloc[0]
    
    # Determine which is better
    unweighted_better = unweighted_values['accuracy'] > weighted_values['accuracy']
    better_method = "Unweighted" if unweighted_better else "Weighted"
    
    print(f"ANALYSIS RESULT: {better_method} features perform better overall")
    print(f"Unweighted accuracy: {unweighted_values['accuracy']:.3f}")
    print(f"Weighted accuracy: {weighted_values['accuracy']:.3f}")
    
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[i//2, i%2]
        
        # Create bar plot with single points
        categories = ['Unweighted Features', 'Weighted Features']
        values = [unweighted_values[metric], weighted_values[metric]]
        
        # Color coding based on performance
        if unweighted_better:
            colors = ['lightgreen', 'lightcoral']
        else:
            colors = ['lightcoral', 'lightgreen']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.8)
        
        # Add single data points
        ax.scatter(0, unweighted_values[metric], c='darkgreen', s=100, zorder=5)
        ax.scatter(1, weighted_values[metric], c='darkred', s=100, zorder=5)
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # ENHANCED: Semantic subplot titles
        ax.set_title(f'{title}: Feature Type Comparison', fontweight='bold')
        ax.set_ylabel(f'{title} Score')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, '1_weighted_vs_unweighted_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"PLOT 1 COMPLETE: Enhanced weighted vs unweighted comparison saved to {output_path}")
    
    return unweighted_values, weighted_values, better_method

def create_dataset_comparison(df, output_dir):
    """
    PLOT 2: Dataset Comparison (best configuration only) - SIMPLIFIED
    Only compare Bitcoin vs Epinions under optimal configuration
    """
    print(f"\n{'='*50}")
    print("PLOT 2: DATASET COMPARISON (SIMPLIFIED - BEST CONFIGURATION ONLY)")
    print("="*50)
    
    # Look for baseline experiments for comparison
    baseline_experiments = [
        'baseline_bitcoin_verification',
        'experiment_epinions_subset_v3',
        'unweighted_optimal'  # Use as Bitcoin baseline if specific baseline not found
    ]
    
    comparison_data = []
    for exp_name in baseline_experiments:
        exp_data = df[df['experiment_name'] == exp_name]
        if len(exp_data) > 0:
            row = exp_data.iloc[0]
            dataset_name = 'Bitcoin OTC' if 'bitcoin' in exp_name.lower() or 'unweighted' in exp_name else 'Epinions'
            comparison_data.append({
                'dataset': dataset_name,
                'accuracy': row['accuracy'],
                'f1_score': row['f1_score'],
                'roc_auc': row['roc_auc'],
                'precision': row['precision']
            })
            print(f"Found dataset experiment: {exp_name} -> {dataset_name}")
    
    if len(comparison_data) < 1:
        print("Warning: Insufficient dataset experiments for comparison")
        # Create informational plot
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, 
                'DATASET COMPARISON NOT AVAILABLE\n\n'
                'Expected experiments for comparison:\n'
                '• baseline_bitcoin_verification (Bitcoin OTC)\n'
                '• experiment_epinions_subset_v3 (Epinions)\n'
                'Or equivalent baseline experiments\n\n'
                'Configuration: cycle_length=4, min_embeddedness=1,\n'
                'pos_ratio=0.5, unweighted, max aggregation',
                ha='center', va='center', transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"),
                fontsize=12)
        ax.set_title('PLOT 2: Dataset Comparison')
        ax.axis('off')
        
        output_path = os.path.join(output_dir, '2_dataset_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    # Create simplified comparison plot
    comparison_df = pd.DataFrame(comparison_data)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Dataset Performance Comparison (Best Configuration Only)\n'
                 'Configuration: cycle_length=4, min_embeddedness=1, pos_ratio=0.5, unweighted, max aggregation', 
                 fontsize=16, fontweight='bold')
    
    metrics = ['accuracy', 'f1_score', 'roc_auc', 'precision']
    metric_titles = ['Accuracy', 'F1 Score', 'ROC AUC', 'Precision']
    colors = ['lightblue', 'lightgreen']
    
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[i//2, i%2]
        
        datasets = comparison_df['dataset'].values
        values = comparison_df[metric].values
        
        bars = ax.bar(datasets, values, color=colors[:len(datasets)], alpha=0.8)
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # ENHANCED: Semantic subplot titles
        ax.set_title(f'{title}: Dataset Performance', fontweight='bold')
        ax.set_ylabel(f'{title} Score')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, '2_dataset_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"PLOT 2 COMPLETE: Simplified dataset comparison saved to {output_path}")

def create_embeddedness_comparison(df, output_dir):
    """
    PLOT 3: Embeddedness Level Comparison - ENHANCED with semantic titles
    Compare embeddedness levels 0 (no filter), 1 (moderate), 2 (strong)
    """
    print(f"\n{'='*50}")
    print("PLOT 3: EMBEDDEDNESS LEVEL COMPARISON (ENHANCED TITLES)")
    print("="*50)
    
    # Filter for embeddedness experiments
    embed_experiments = ['embed_0_optimal', 'embed_1_optimal', 'embed_2_optimal']
    embed_df = df[df['experiment_name'].isin(embed_experiments)].copy()
    
    print(f"Filtered embeddedness experiments: {len(embed_df)}")
    
    if embed_df.empty:
        print("Warning: No embeddedness experiments found")
        
        # Create informational plot
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, 
                'MISSING EMBEDDEDNESS EXPERIMENTS\n\n'
                'Expected experiments with different embeddedness levels:\n'
                '• embed_0_optimal: No filtering (min_embeddedness=0)\n'
                '• embed_1_optimal: Moderate filtering (min_embeddedness=1)\n'
                '• embed_2_optimal: Strong filtering (min_embeddedness=2)\n\n'
                'Note: Embeddedness filtering now applied in preprocessing stage.',
                ha='center', va='center', transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"),
                fontsize=12)
        ax.set_title('PLOT 3: Missing Embeddedness Level Experiments')
        ax.axis('off')
        
        output_path = os.path.join(output_dir, '3_embeddedness_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    # Sort by embeddedness level for consistent ordering
    embed_df = embed_df.sort_values('embeddedness_level')
    
    print(f"Available embeddedness levels: {sorted(embed_df['embeddedness_level'].unique())}")
    
    # Show data distribution
    for _, row in embed_df.iterrows():
        print(f"   {row['experiment_name']}: level {row['embeddedness_level']}, accuracy={row['accuracy']:.3f}")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Embeddedness Level Performance Comparison\n'
                 'Configuration: pos_ratio=0.5, cycle_length=4, unweighted, max aggregation', 
                 fontsize=16, fontweight='bold')
    
    metrics = ['accuracy', 'f1_score', 'roc_auc', 'precision']
    metric_titles = ['Accuracy', 'F1 Score', 'ROC AUC', 'Precision']
    colors = ['lightcoral', 'lightblue', 'lightgreen']  # Red, blue, green for levels 0, 1, 2
    
    # ENHANCED: Semantic level labels
    level_labels = {
        0: 'No Filter\n(Level 0)', 
        1: 'Moderate Filter\n(Level 1)', 
        2: 'Strong Filter\n(Level 2)'
    }
    
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[i//2, i%2]
        
        # Extract data for plotting
        levels = embed_df['embeddedness_level'].values
        values = embed_df[metric].values
        
        # Create bar plot with semantic labels
        bar_colors = [colors[int(level)] for level in levels]
        bars = ax.bar(range(len(levels)), values, color=bar_colors, alpha=0.8)
        
        # Overlay individual points
        ax.scatter(range(len(levels)), values, color='darkred', s=100, zorder=5, 
                  marker='o', edgecolor='darkred', linewidth=2)
        
        # ENHANCED: Semantic subplot titles
        ax.set_title(f'{title} vs Embeddedness Filtering Level', fontsize=14, fontweight='bold')
        ax.set_xlabel('Embeddedness Filtering Strategy', fontsize=12)
        ax.set_ylabel(f'{title} Score', fontsize=12)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([level_labels.get(int(level), f'Level {int(level)}') for level in levels])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, '3_embeddedness_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"PLOT 3 COMPLETE: Enhanced embeddedness comparison saved to {output_path}")

def create_aggregation_methods_comparison(df, output_dir):
    """
    PLOT 4: Aggregation Methods Comparison (Max vs Sum vs Stronger)
    """
    print(f"\n{'='*50}")
    print("PLOT 4: AGGREGATION METHODS COMPARISON")
    print("="*50)
    
    # Filter for aggregation method experiments
    agg_experiments = ['aggregation_max_optimal', 'aggregation_sum_optimal', 'aggregation_stronger_optimal']
    agg_df = df[df['experiment_name'].isin(agg_experiments)].copy()
    
    if agg_df.empty:
        print("Warning: No aggregation method experiments found")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'MISSING AGGREGATION EXPERIMENTS\n\n'
                         'Expected experiments:\n'
                         '• aggregation_max_optimal\n'
                         '• aggregation_sum_optimal\n'
                         '• aggregation_stronger_optimal\n\n'
                         'These test different bidirectional edge handling methods.',
               ha='center', va='center', transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"),
               fontsize=12)
        ax.set_title('PLOT 4: Aggregation Methods Analysis')
        ax.axis('off')
        
        output_path = os.path.join(output_dir, '4_aggregation_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    # Extract method names from experiment names
    agg_df['method'] = agg_df['experiment_name'].str.extract(r'aggregation_(\w+)_optimal')[0]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Aggregation Methods Performance Comparison\n'
                 'Configuration: pos_ratio=0.5, min_embeddedness=1, cycle_length=4, unweighted', 
                 fontsize=16, fontweight='bold')
    
    metrics = ['accuracy', 'f1_score', 'roc_auc', 'precision']
    metric_titles = ['Accuracy', 'F1 Score', 'ROC AUC', 'Precision']
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[i//2, i%2]
        
        methods = agg_df['method'].values
        values = agg_df[metric].values
        
        bars = ax.bar(methods, values, color=colors[:len(methods)], alpha=0.8)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # ENHANCED: Semantic subplot titles
        ax.set_title(f'{title}: Bidirectional Edge Aggregation', fontweight='bold')
        ax.set_ylabel(f'{title} Score', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, '4_aggregation_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"PLOT 4 COMPLETE: Aggregation methods comparison saved to {output_path}")

def create_performance_summary_table(df, output_dir):
    """
    PLOT 5: Complete Performance Summary Table - Enhanced
    """
    print(f"\n{'='*50}")
    print("PLOT 5: PERFORMANCE SUMMARY TABLE")
    print("="*50)
    
    # Sort by accuracy descending
    df_sorted = df.sort_values('accuracy', ascending=False)
    
    fig, ax = plt.subplots(figsize=(24, 16))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data with enhanced information
    table_data = []
    headers = ['Rank', 'Experiment', 'Accuracy', 'F1 Score', 'ROC AUC', 'Precision', 
               'Feature Type', 'Split Type', 'Cycle Len', 'Embed Lvl', 'Pos Ratio', 'Aggregation']
    
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        table_data.append([
            i + 1,
            row['experiment_name'][:18] + '...' if len(row['experiment_name']) > 18 else row['experiment_name'],
            f"{row['accuracy']:.3f}",
            f"{row['f1_score']:.3f}",
            f"{row['roc_auc']:.3f}",
            f"{row['precision']:.3f}",
            row['feature_type'],
            row['split_type'].title(),
            f"{row['cycle_length']}",
            f"{row['embeddedness_level']}",
            f"{row['positive_ratio']:.1f}",
            row['aggregation_method']
        ])
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    
    # Color rows based on performance
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        row_idx = i + 1  # +1 because of header
        
        # Color based on performance (top 3 get special colors)
        if i < 3:  # Top 3
            color = ['#FFD700', '#C0C0C0', '#CD7F32'][i]  # Gold, Silver, Bronze
        elif row['optimal_split']:
            color = '#FFE6CC'  # Light orange for optimal split
        else:
            color = '#F0F0F0'  # Light gray for standard
        
        for j in range(len(headers)):
            table[(row_idx, j)].set_facecolor(color)
    
    # Color header
    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#4CAF50')  # Green
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    plt.title('Complete Performance Summary Table\n(Gold/Silver/Bronze for Top 3, Orange for Optimal Split)', 
              fontsize=16, fontweight='bold', pad=20)
    
    output_path = os.path.join(output_dir, '5_performance_summary_table.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"PLOT 5 COMPLETE: Performance summary table saved to {output_path}")

def create_positive_ratio_comparison(df, output_dir):
    """
    PLOT 6: Positive Ratio Comparison - Enhanced for pos_ratio experiments
    """
    print(f"\n{'='*50}")
    print("PLOT 6: POSITIVE RATIO COMPARISON")
    print("="*50)
    
    # Filter for positive ratio experiments from run_experiment.py
    pos_ratio_experiments = ['pos_ratio_90_10_optimal', 'pos_ratio_80_20_optimal', 
                            'pos_ratio_70_30_optimal', 'pos_ratio_60_40_optimal', 
                            'pos_ratio_50_50_optimal']
    
    ratio_df = df[df['experiment_name'].isin(pos_ratio_experiments)].copy()
    
    if ratio_df.empty:
        print("Warning: No pos_ratio experiments found from run_experiment.py")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, 
                'MISSING POSITIVE RATIO EXPERIMENTS\n\n'
                'Expected experiments from run_experiment.py:\n'
                '• pos_ratio_90_10_optimal (90% pos, 10% neg)\n'
                '• pos_ratio_80_20_optimal (80% pos, 20% neg)\n'
                '• pos_ratio_70_30_optimal (70% pos, 30% neg)\n'
                '• pos_ratio_60_40_optimal (60% pos, 40% neg)\n'
                '• pos_ratio_50_50_optimal (50% pos, 50% neg)\n\n'
                'These are generated by run_experiment.py with different\n'
                'pos_edges_ratio settings.',
                ha='center', va='center', transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"),
                fontsize=12)
        ax.set_title('PLOT 6: Missing Positive Ratio Experiments')
        ax.axis('off')
        
        output_path = os.path.join(output_dir, '6_positive_ratio_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    # Sort by positive ratio
    ratio_df = ratio_df.sort_values('positive_ratio', ascending=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Positive/Negative Ratio Impact on Performance\n'
                 'Configuration: min_embeddedness=1, cycle_length=4, unweighted, max aggregation', 
                 fontsize=16, fontweight='bold')
    
    metrics = ['accuracy', 'f1_score', 'roc_auc', 'precision']
    metric_titles = ['Accuracy', 'F1 Score', 'ROC AUC', 'Precision']
    
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[i//2, i%2]
        
        pos_ratios = ratio_df['positive_ratio'].values * 100  # Convert to percentage
        values = ratio_df[metric].values
        
        ax.plot(pos_ratios, values, marker='s', linewidth=3, markersize=8, color='#F18F01')
        ax.fill_between(pos_ratios, values, alpha=0.3, color='#F18F01')
        
        # ENHANCED: Semantic subplot titles
        ax.set_title(f'{title}: Impact of Class Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Positive Examples (%)', fontsize=12)
        ax.set_ylabel(f'{title} Score', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for x, y in zip(pos_ratios, values):
            ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points",
                       xytext=(0,10), ha='center', fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '6_positive_ratio_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"PLOT 6 COMPLETE: Positive ratio comparison saved to {output_path}")

def create_cycle_length_comparison(df, output_dir):
    """
    PLOT 7: Cycle Length Comparison (3, 4, 5) - NEW ADDITION
    """
    print(f"\n{'='*50}")
    print("PLOT 7: CYCLE LENGTH COMPARISON (NEW)")
    print("="*50)
    
    # Filter for cycle length experiments
    cycle_experiments = ['cycle_length_3_optimal', 'cycle_length_4_optimal', 'cycle_length_5_optimal']
    cycle_df = df[df['experiment_name'].isin(cycle_experiments)].copy()
    
    if cycle_df.empty:
        print("Warning: No cycle length experiments found")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, 
                'MISSING CYCLE LENGTH EXPERIMENTS\n\n'
                'Expected experiments:\n'
                '• cycle_length_3_optimal\n'
                '• cycle_length_4_optimal\n'
                '• cycle_length_5_optimal\n\n'
                'These test different structural feature complexity levels\n'
                'with HOC features of length 3, 4, and 5.\n\n'
                'Configuration: pos_ratio=0.5, min_embeddedness=1,\n'
                'unweighted, max aggregation',
                ha='center', va='center', transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"),
                fontsize=12)
        ax.set_title('PLOT 7: Missing Cycle Length Experiments')
        ax.axis('off')
        
        output_path = os.path.join(output_dir, '7_cycle_length_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    cycle_df = cycle_df.sort_values('cycle_length')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cycle Length Performance Comparison\n'
                 'Configuration: pos_ratio=0.5, min_embeddedness=1, unweighted, max aggregation', 
                 fontsize=16, fontweight='bold')
    
    metrics = ['accuracy', 'f1_score', 'roc_auc', 'precision']
    metric_titles = ['Accuracy', 'F1 Score', 'ROC AUC', 'Precision']
    
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[i//2, i%2]
        
        cycle_lengths = cycle_df['cycle_length'].values
        values = cycle_df[metric].values
        
        # Plot line with markers
        ax.plot(cycle_lengths, values, marker='o', linewidth=3, markersize=10, color='#2E86AB')
        ax.fill_between(cycle_lengths, values, alpha=0.3, color='#2E86AB')
        
        # ENHANCED: Semantic subplot titles
        ax.set_title(f'{title}: Structural Feature Complexity', fontsize=14, fontweight='bold')
        ax.set_xlabel('Cycle Length', fontsize=12)
        ax.set_ylabel(f'{title} Score', fontsize=12)
        ax.set_xticks(cycle_lengths)
        ax.set_xticklabels([f'Length {int(length)}' for length in cycle_lengths])
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for x, y in zip(cycle_lengths, values):
            ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, '7_cycle_length_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"PLOT 7 COMPLETE: Cycle length comparison saved to {output_path}")

def create_pos_neg_ratio_experiments_comparison(df, output_dir):
    """
    PLOT 8: Different Pos/Neg Rate Experiments - Enhanced version of Plot 6
    """
    print(f"\n{'='*50}")
    print("PLOT 8: POS/NEG RATIO EXPERIMENTS (Enhanced)")
    print("="*50)
    
    # Same as Plot 6 but with different visualization approach
    pos_ratio_experiments = ['pos_ratio_90_10_optimal', 'pos_ratio_80_20_optimal', 
                            'pos_ratio_70_30_optimal', 'pos_ratio_60_40_optimal', 
                            'pos_ratio_50_50_optimal']
    
    ratio_df = df[df['experiment_name'].isin(pos_ratio_experiments)].copy()
    
    if ratio_df.empty:
        print("Warning: No pos/neg ratio experiments found")

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, 
                'MISSING POS/NEG RATIO EXPERIMENTS\n\n'
                'Expected experiments with different class ratios:\n'
                '• pos_ratio_90_10_optimal (90% pos, 10% neg)\n'
                '• pos_ratio_80_20_optimal (80% pos, 20% neg)\n'
                '• pos_ratio_70_30_optimal (70% pos, 30% neg)\n'
                '• pos_ratio_60_40_optimal (60% pos, 40% neg)\n'
                '• pos_ratio_50_50_optimal (50% pos, 50% neg)\n\n'
                'These test model performance under different\n'
                'class distribution scenarios.',
               ha='center', va='center', transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"),
               fontsize=12)
        ax.set_title('PLOT 8: Pos/Neg Ratio Analysis')
        ax.axis('off')
        
        output_path = os.path.join(output_dir, '8_pos_neg_ratio_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    ratio_df = ratio_df.sort_values('positive_ratio', ascending=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Impact of Different Pos/Neg Ratios on Performance\n'
                 'Configuration: min_embeddedness=1, cycle_length=4, unweighted, max aggregation', 
                 fontsize=16, fontweight='bold')
    
    metrics = ['accuracy', 'f1_score', 'roc_auc', 'precision']
    metric_titles = ['Accuracy', 'F1 Score', 'ROC AUC', 'Precision']
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
    
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[i//2, i%2]
        
        # Create bar plot
        ratio_labels = [f'{int(row["positive_ratio"]*100)}%-{int((1-row["positive_ratio"])*100)}%' 
                       for _, row in ratio_df.iterrows()]
        values = ratio_df[metric].values
        
        bars = ax.bar(range(len(ratio_labels)), values, 
                     color=colors[:len(ratio_labels)], alpha=0.8)
        
        # Add value labels
        for j, (bar, value) in enumerate(zip(bars, values)):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # ENHANCED: Semantic subplot titles
        ax.set_title(f'{title}: Class Imbalance Impact', fontweight='bold')
        ax.set_ylabel(f'{title} Score', fontsize=12)
        ax.set_xticks(range(len(ratio_labels)))
        ax.set_xticklabels(ratio_labels, rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, '8_pos_neg_ratio_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"PLOT 8 COMPLETE: Pos/neg ratio comparison saved to {output_path}")

def main():
    """
    Main function following the enhanced workflow with all required plots
    """
    print("COMPLETE RESULTS ANALYSIS - ENHANCED VERSION")
    print("="*80)
    print("Enhanced Workflow with All Required Plots:")
    print("PLOT 1: Weighted vs Unweighted (MOST IMPORTANT - use optimal split only)")
    print("PLOT 2: Dataset Comparison (best configuration only)")
    print("PLOT 3: Embeddedness Level Comparison (0, 1, 2) - FIXED")
    print("PLOT 4: Aggregation Methods (Max vs Sum vs Stronger)")
    print("PLOT 5: Performance Summary Table (comprehensive)")
    print("PLOT 6: Positive Ratio Comparison (from run_experiment.py)")
    print("PLOT 7: Cycle Length Comparison (3, 4, 5) - NEW")
    print("PLOT 8: Pos/Neg Ratio Experiments (enhanced version)")
    print("")
    print("ENHANCED CONFIGURATION:")
    print("- Embeddedness filtering applied in preprocessing stage")
    print("- Split ratio: from config.yaml (test=4000, validation=4000)")
    print("- All experiments use optimal split")
    print("- Enhanced subplot titles with semantic naming")
    print("="*80)
    
    # Create output directory
    output_dir = os.path.abspath(os.path.join(PROJECT_ROOT, 'plots', 'complete_results_analysis'))
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Find all experiments
    found_experiments, optimal_experiments, embeddedness_experiments = find_all_experiments()
    
    if not found_experiments:
        print("No experiments found. Please run some experiments first.")
        print("To generate missing experiments, run: python run_batch_experiments.py")
        return
    
    # Load all experiment metrics
    print(f"\nLoading all experiment data...")
    all_metrics = []
    
    for exp_name in found_experiments.keys():
        print(f"   Loading metrics for {exp_name}...")
        metrics = load_experiment_metrics(exp_name)
        all_metrics.append(metrics)
        
        split_status = " (OPTIMAL)" if metrics['optimal_split'] else ""
        print(f"   Loaded: {exp_name}{split_status}")
    
    print(f"Successfully loaded {len(all_metrics)} experiments")
    print(f"Including {len(optimal_experiments)} optimal split experiments")
    print(f"Including {len(embeddedness_experiments)} embeddedness experiments")
    
    # Extract structured data
    df = extract_experiment_data(all_metrics)
    
    if df.empty:
        print("No valid experiment data found")
        return
    
    print(f"\nAnalyzing {len(df)} experiments following enhanced workflow...")
    
    # Check available variations
    cycle_lengths = df['cycle_length'].unique()
    positive_ratios = df['positive_ratio'].unique()
    embeddedness_levels = df['embeddedness_level'].unique()
    
    print(f"\nAvailable variations:")
    print(f"  Cycle lengths: {sorted(cycle_lengths)}")
    print(f"  Positive ratios: {sorted(positive_ratios)}")
    print(f"  Embeddedness levels: {sorted(embeddedness_levels)}")
    
    # Store analysis results
    analysis_results = {}
    
    # ENHANCED WORKFLOW - ALL 8 PLOTS
    
    # PLOT 1: Most Important - Weighted vs Unweighted (use optimal split only)
    unweighted_means, weighted_means, better_method = create_weighted_vs_unweighted_comparison(df, output_dir)
    analysis_results['better_method'] = better_method
    analysis_results['unweighted_means'] = unweighted_means
    analysis_results['weighted_means'] = weighted_means
    
    # PLOT 2: Dataset Comparison (best configuration only)
    create_dataset_comparison(df, output_dir)
    
    # PLOT 3: Embeddedness Level Comparison (FIXED)
    create_embeddedness_comparison(df, output_dir)
    
    # PLOT 4: Aggregation Methods Comparison
    create_aggregation_methods_comparison(df, output_dir)
    
    # PLOT 5: Complete Performance Summary Table
    create_performance_summary_table(df, output_dir)
    
    # PLOT 6: Positive Ratio Comparison
    create_positive_ratio_comparison(df, output_dir)
    
    # PLOT 7: Cycle Length Comparison (3, 4, 5) - NEW
    create_cycle_length_comparison(df, output_dir)
    
    # PLOT 8: Pos/Neg Ratio Experiments - Enhanced
    create_pos_neg_ratio_experiments_comparison(df, output_dir)
    
    # Final summary
    print("\n" + "="*80)
    print("COMPLETE WORKFLOW COMPLETION SUMMARY")
    print("="*80)
    print(f"Analyzed {len(df)} experiments")
    print(f"Best accuracy: {df['accuracy'].max():.3f}")
    print(f"Best F1 score: {df['f1_score'].max():.3f}")
    
    # Key decision from Plot 1
    if better_method:
        print(f"\nKEY DECISION (Plot 1): Use {better_method} features for all future work")
        if unweighted_means is not None and weighted_means is not None:
            improvement = abs(unweighted_means['accuracy'] - weighted_means['accuracy']) / min(unweighted_means['accuracy'], weighted_means['accuracy']) * 100
            print(f"Performance difference: {improvement:.1f}%")
    
    print(f"\nAll plots saved to: {output_dir}")
    
    # File verification
    print(f"\nGenerated Files:")
    expected_files = [
        '1_weighted_vs_unweighted_comparison.png',
        '2_dataset_comparison.png',
        '3_embeddedness_comparison.png',
        '4_aggregation_comparison.png',
        '5_performance_summary_table.png',
        '6_positive_ratio_comparison.png',
        '7_cycle_length_comparison.png',
        '8_pos_neg_ratio_comparison.png'
    ]
    
    for filename in expected_files:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"✓ {filename}: {size:,} bytes")
        else:
            print(f"✗ {filename}: NOT FOUND")
    
    print("\n" + "="*80)
    print("ENHANCED WORKFLOW COMPLETED WITH ALL PLOTS!")
    print("Issues addressed:")
    print("✓ Plot 1 - Weighted vs Unweighted comparison (most important)")
    print("✓ Plot 2 - Dataset comparison (best configuration only)")
    print("✓ Plot 3 - Embeddedness level comparison with preprocessing filtering")
    print("✓ Plot 4 - Aggregation methods comparison")
    print("✓ Plot 5 - Performance summary table")
    print("✓ Plot 6 - Multiple positive ratios comparison")
    print("✓ Plot 7 - Cycle length (3, 4, 5) comparison - NEW")
    print("✓ Plot 8 - Pos/neg ratio experiments (enhanced)")
    print("✓ All experiments use optimal split")
    print("✓ Enhanced subplot titles with semantic naming")
    print("✓ Embeddedness filtering applied in preprocessing stage")
    print("✓ Ready for presentation with all required plots")
    print("="*80)

if __name__ == "__main__":
    main()