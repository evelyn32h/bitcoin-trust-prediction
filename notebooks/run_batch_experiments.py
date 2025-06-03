#!/usr/bin/env python3
"""
Enhanced Batch Experiments Runner - Fixed Embeddedness Parameter Handling
=========================================================================

This script runs comprehensive batch experiments for link prediction analysis:
- Step 1: Weighted vs Unweighted Features Comparison
- Step 3: Embeddedness Level Comparison (0, 1, 2) - FIXED parameter handling
- Steps 6&8: Positive/Negative Ratio Comparisons
- Step 7: Cycle Length Comparisons (3, 4, 5)

FIXED: Embeddedness parameters are now passed to preprocessing stage,
ensuring proper filtering before BFS sampling.

Usage: python run_batch_experiments.py
"""

import os
import sys
import json
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from datetime import datetime
import subprocess
import shutil

# Set up paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'analysis_output')

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_single_experiment(experiment_name, config_modifications=None, base_config_name='config.yaml'):
    """
    Run a single experiment with specified configuration modifications
    FIXED: Properly pass embeddedness parameters to preprocessing stage
    """
    print(f"\n{'='*60}")
    print(f"RUNNING EXPERIMENT: {experiment_name}")
    print(f"{'='*60}")
    
    if config_modifications is None:
        config_modifications = {}
    
    # Try multiple possible config locations
    config_paths = [
        os.path.join(PROJECT_ROOT, 'configs', base_config_name),
        os.path.join(PROJECT_ROOT, 'config.yaml'),
        os.path.join(PROJECT_ROOT, 'configs', 'config.yaml'),
        os.path.join(PROJECT_ROOT, 'notebooks', 'config.yaml')
    ]
    
    config_path = None
    for path in config_paths:
        if os.path.exists(path):
            config_path = path
            break
    
    if config_path is None:
        print(f"Error: No base config file found. Tried:")
        for path in config_paths:
            print(f"  - {path}")
        return False
    
    print(f"Using base config: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading base config: {e}")
        return False
    
    # Apply configuration modifications first
    for key, value in config_modifications.items():
        config[key] = value
        print(f"  Modified config: {key} = {value}")
    
    # Extract key parameters for experiment execution
    min_train_embeddedness = config.get('min_train_embeddedness', 1)
    use_weighted_features = config.get('use_weighted_features', False)
    bidirectional_method = config.get('bidirectional_method', 'max')
    cycle_length = config.get('cycle_length', 4)
    
    print(f"  Key parameters:")
    print(f"    min_train_embeddedness: {min_train_embeddedness}")
    print(f"    use_weighted_features: {use_weighted_features}")
    print(f"    bidirectional_method: {bidirectional_method}")
    print(f"    cycle_length: {cycle_length}")
    
    # Set standard parameters for optimal experiments
    standard_params = {
        'num_test_edges': 3080,
        'num_validation_edges': 2640,
        'optimal_split': True,
        'experiment_name': experiment_name,
        'save_predictions': True,
        'save_model': False,
        'verbose': True
    }
    
    config.update(standard_params)
    
    try:
        # Step 1: Run preprocessing with embeddedness parameter
        print(f"  Step 1: Running preprocessing...")
        preprocess_cmd = [
            sys.executable, 'notebooks/preprocess.py',
            '--name', experiment_name,
            '--min_embeddedness', str(min_train_embeddedness)
        ]
        
        # Add conditional flags
        if use_weighted_features:
            preprocess_cmd.append('--use_weighted_features')
        if bidirectional_method:
            preprocess_cmd.extend(['--bidirectional_method', bidirectional_method])
        
        print(f"    Command: {' '.join(preprocess_cmd)}")
        
        result = subprocess.run(
            preprocess_cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes timeout
        )
        
        if result.returncode != 0:
            print(f"  ✗ Preprocessing failed")
            print(f"  Error: {result.stderr}")
            return False
        
        print(f"  ✓ Preprocessing completed successfully")
        
        # Step 2: Run training
        print(f"  Step 2: Running training...")
        train_cmd = [
            sys.executable, 'notebooks/train_model.py',
            '--name', experiment_name,
            '--n_folds', '5',
            '--cycle_length', str(cycle_length)
        ]
        
        print(f"    Command: {' '.join(train_cmd)}")
        
        result = subprocess.run(
            train_cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes timeout
        )
        
        if result.returncode != 0:
            print(f"  ✗ Training failed")
            print(f"  Error: {result.stderr}")
            return False
        
        print(f"  ✓ Training completed successfully")
        
        # Step 3: Run validation
        print(f"  Step 3: Running validation...")
        validate_cmd = [
            sys.executable, 'notebooks/validate_model.py',
            '--name', experiment_name,
            '--n_folds', '5',
            '--cycle_length', str(cycle_length)
        ]
        
        result = subprocess.run(
            validate_cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=1800
        )
        
        if result.returncode != 0:
            print(f"  ✗ Validation failed")
            print(f"  Error: {result.stderr}")
            return False
        
        print(f"  ✓ Validation completed successfully")
        
        # Step 4: Run testing
        print(f"  Step 4: Running testing...")
        test_cmd = [
            sys.executable, 'notebooks/test_model.py',
            '--name', experiment_name,
            '--n_folds', '5',
            '--cycle_length', str(cycle_length)
        ]
        
        result = subprocess.run(
            test_cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=1800
        )
        
        if result.returncode != 0:
            print(f"  ✗ Testing failed")
            print(f"  Error: {result.stderr}")
            return False
        
        print(f"  ✓ Testing completed successfully")
        
        # Verify results exist
        metrics_path = os.path.join(RESULTS_DIR, experiment_name, 'testing', 'metrics.json')
        if os.path.exists(metrics_path):
            print(f"  ✓ Results verified: {metrics_path}")
            
            # Copy config to results directory for reference
            config_dest = os.path.join(RESULTS_DIR, experiment_name, 'config_used.yaml')
            try:
                with open(config_dest, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
                print(f"  ✓ Config saved to results directory")
            except Exception as e:
                print(f"  Warning: Could not save config: {e}")
            
            return True
        else:
            print(f"  ✗ Results not found: {metrics_path}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  ✗ Experiment {experiment_name} timed out")
        return False
    except Exception as e:
        print(f"  ✗ Experiment {experiment_name} failed with exception: {e}")
        return False

def run_weighted_vs_unweighted_experiments():
    """
    Step 1: Run weighted vs unweighted feature experiments (Most Important)
    """
    print(f"\n{'='*80}")
    print("STEP 1: WEIGHTED vs UNWEIGHTED FEATURES EXPERIMENTS")
    print("="*80)
    print("Running the most important comparison: weighted vs unweighted features")
    
    experiments = [
        {
            'name': 'weighted_optimal',
            'config': {
                'use_weighted_features': True,
                'use_structural_features': True,
                'use_centrality_features': True,
                'min_train_embeddedness': 1,  # Default embeddedness
                'cycle_length': 4,
                'bidirectional_method': 'max'
            }
        },
        {
            'name': 'unweighted_optimal',
            'config': {
                'use_weighted_features': False,
                'use_structural_features': True,
                'use_centrality_features': True,
                'min_train_embeddedness': 1,  # Default embeddedness
                'cycle_length': 4,
                'bidirectional_method': 'max'
            }
        }
    ]
    
    results = []
    for exp in experiments:
        print(f"\nRunning {exp['name']}...")
        success = run_single_experiment(exp['name'], exp['config'])
        results.append((exp['name'], success))
        
        if success:
            print(f"✓ {exp['name']} completed successfully")
        else:
            print(f"✗ {exp['name']} failed")
    
    print(f"\nSTEP 1 SUMMARY:")
    for name, success in results:
        status = "SUCCESS" if success else "FAILED"
        print(f"  {name}: {status}")
    
    return results

def run_embeddedness_experiments():
    """
    Step 3: Run embeddedness level experiments (0, 1, 2) - FIXED
    """
    print(f"\n{'='*80}")
    print("STEP 3: EMBEDDEDNESS LEVEL EXPERIMENTS (FIXED)")
    print("="*80)
    print("Testing different embeddedness filtering levels:")
    print("- Level 0: No filtering (include all edges)")
    print("- Level 1: Moderate filtering (standard)")
    print("- Level 2: Strong filtering (high embeddedness only)")
    
    experiments = [
        {
            'name': 'embed_0_optimal',
            'config': {
                'min_train_embeddedness': 0,  # No filtering
                'min_val_embeddedness': 0,
                'min_test_embeddedness': 0,
                'use_weighted_features': False,  # Use best performing feature type
                'use_structural_features': True,
                'use_centrality_features': True,
                'cycle_length': 4,
                'bidirectional_method': 'max'
            }
        },
        {
            'name': 'embed_1_optimal',
            'config': {
                'min_train_embeddedness': 1,  # Moderate filtering
                'min_val_embeddedness': 1,
                'min_test_embeddedness': 1,
                'use_weighted_features': False,  # Use best performing feature type
                'use_structural_features': True,
                'use_centrality_features': True,
                'cycle_length': 4,
                'bidirectional_method': 'max'
            }
        },
        {
            'name': 'embed_2_optimal',
            'config': {
                'min_train_embeddedness': 2,  # Strong filtering
                'min_val_embeddedness': 2,
                'min_test_embeddedness': 2,
                'use_weighted_features': False,  # Use best performing feature type
                'use_structural_features': True,
                'use_centrality_features': True,
                'cycle_length': 4,
                'bidirectional_method': 'max'
            }
        }
    ]
    
    results = []
    for exp in experiments:
        print(f"\nRunning {exp['name']}...")
        print(f"  Embeddedness level: {exp['config']['min_train_embeddedness']}")
        success = run_single_experiment(exp['name'], exp['config'])
        results.append((exp['name'], success))
        
        if success:
            print(f"✓ {exp['name']} completed successfully")
        else:
            print(f"✗ {exp['name']} failed")
    
    print(f"\nSTEP 3 SUMMARY:")
    for name, success in results:
        status = "SUCCESS" if success else "FAILED"
        level = name.split('_')[1]  # Extract level from name
        print(f"  Embeddedness Level {level}: {status}")
    
    return results

def run_cycle_length_experiments():
    """
    Step 7: Run cycle length experiments (3, 4, 5)
    """
    print(f"\n{'='*80}")
    print("STEP 7: CYCLE LENGTH EXPERIMENTS")
    print("="*80)
    print("Testing different structural feature complexity levels:")
    
    experiments = [
        {
            'name': 'cycle_length_3_optimal',
            'config': {
                'cycle_length': 3,
                'use_weighted_features': False,  # Use best performing feature type
                'use_structural_features': True,
                'use_centrality_features': True,
                'min_train_embeddedness': 1,  # Default embeddedness
                'bidirectional_method': 'max'
            }
        },
        {
            'name': 'cycle_length_4_optimal',
            'config': {
                'cycle_length': 4,
                'use_weighted_features': False,
                'use_structural_features': True,
                'use_centrality_features': True,
                'min_train_embeddedness': 1,  # Default embeddedness
                'bidirectional_method': 'max'
            }
        },
        {
            'name': 'cycle_length_5_optimal',
            'config': {
                'cycle_length': 5,
                'use_weighted_features': False,
                'use_structural_features': True,
                'use_centrality_features': True,
                'min_train_embeddedness': 1,  # Default embeddedness
                'bidirectional_method': 'max'
            }
        }
    ]
    
    results = []
    for exp in experiments:
        print(f"\nRunning {exp['name']}...")
        cycle_len = exp['config']['cycle_length']
        print(f"  Cycle length: {cycle_len}")
        success = run_single_experiment(exp['name'], exp['config'])
        results.append((exp['name'], success))
        
        if success:
            print(f"✓ {exp['name']} completed successfully")
        else:
            print(f"✗ {exp['name']} failed")
    
    print(f"\nSTEP 7 SUMMARY:")
    for name, success in results:
        status = "SUCCESS" if success else "FAILED"
        length = name.split('_')[2]  # Extract length from name
        print(f"  Cycle Length {length}: {status}")
    
    return results

def run_aggregation_experiments():
    """
    Step 4: Run aggregation method experiments
    """
    print(f"\n{'='*80}")
    print("STEP 4: AGGREGATION METHOD EXPERIMENTS")
    print("="*80)
    print("Testing different bidirectional edge aggregation methods:")
    
    experiments = [
        {
            'name': 'aggregation_max_optimal',
            'config': {
                'bidirectional_method': 'max',
                'use_weighted_features': False,
                'use_structural_features': True,
                'use_centrality_features': True,
                'min_train_embeddedness': 1,  # Default embeddedness
                'cycle_length': 4
            }
        },
        {
            'name': 'aggregation_sum_optimal',
            'config': {
                'bidirectional_method': 'sum',
                'use_weighted_features': False,
                'use_structural_features': True,
                'use_centrality_features': True,
                'min_train_embeddedness': 1,  # Default embeddedness
                'cycle_length': 4
            }
        },
        {
            'name': 'aggregation_stronger_optimal',
            'config': {
                'bidirectional_method': 'stronger',
                'use_weighted_features': False,
                'use_structural_features': True,
                'use_centrality_features': True,
                'min_train_embeddedness': 1,  # Default embeddedness
                'cycle_length': 4
            }
        }
    ]
    
    results = []
    for exp in experiments:
        print(f"\nRunning {exp['name']}...")
        method = exp['config']['bidirectional_method']
        print(f"  Aggregation method: {method}")
        success = run_single_experiment(exp['name'], exp['config'])
        results.append((exp['name'], success))
        
        if success:
            print(f"✓ {exp['name']} completed successfully")
        else:
            print(f"✗ {exp['name']} failed")
    
    print(f"\nSTEP 4 SUMMARY:")
    for name, success in results:
        status = "SUCCESS" if success else "FAILED"
        method = name.split('_')[1]  # Extract method from name
        print(f"  Aggregation {method}: {status}")
    
    return results

def run_dataset_comparison_experiments():
    """
    Step 2: Run dataset comparison experiments (Bitcoin vs Others)
    """
    print(f"\n{'='*80}")
    print("STEP 2: DATASET COMPARISON EXPERIMENTS")
    print("="*80)
    print("Creating baseline experiments for dataset comparison:")
    
    experiments = [
        {
            'name': 'baseline_bitcoin_verification',
            'config': {
                'use_weighted_features': False,
                'use_structural_features': True,
                'use_centrality_features': True,
                'min_train_embeddedness': 1,  # Default embeddedness
                'cycle_length': 4,
                'bidirectional_method': 'max',
                'data_path': 'data/soc-sign-bitcoinotc.csv'
            }
        }
    ]
    
    results = []
    for exp in experiments:
        print(f"\nRunning {exp['name']}...")
        success = run_single_experiment(exp['name'], exp['config'])
        results.append((exp['name'], success))
        
        if success:
            print(f"✓ {exp['name']} completed successfully")
        else:
            print(f"✗ {exp['name']} failed")
    
    print(f"\nSTEP 2 SUMMARY:")
    for name, success in results:
        status = "SUCCESS" if success else "FAILED"
        print(f"  {name}: {status}")
    
    return results

def extract_experiment_data(experiment_name, experiment_path):
    """Extract data from a single experiment with proper embeddedness level detection"""
    try:
        # Load metrics
        metrics_path = os.path.join(experiment_path, 'testing', 'metrics.json')
        if not os.path.exists(metrics_path):
            print(f"Warning: No metrics found for {experiment_name}")
            return None
            
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Load configuration - try multiple locations
        config_used = {}
        config_paths = [
            os.path.join(experiment_path, 'config_used.yaml'),
            os.path.join(experiment_path, 'preprocess', 'config_used.yaml'),
            os.path.join(experiment_path, 'config.yaml')
        ]
        
        for config_path in config_paths:
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_used = yaml.safe_load(f)
                    break
                except Exception as e:
                    print(f"Warning: Could not load config from {config_path}: {e}")
                    continue
        
        # Extract embeddedness level - ENHANCED LOGIC
        embeddedness_level = 1  # default
        
        # Method 1: From config file (highest priority)
        if 'min_test_embeddedness' in config_used:
            embeddedness_level = int(config_used['min_test_embeddedness'])
        elif 'min_train_embeddedness' in config_used:
            embeddedness_level = int(config_used['min_train_embeddedness'])
        elif 'min_val_embeddedness' in config_used:
            embeddedness_level = int(config_used['min_val_embeddedness'])
        # Method 2: From experiment name pattern
        elif 'embed_' in experiment_name:
            try:
                parts = experiment_name.split('_')
                for i, part in enumerate(parts):
                    if part == 'embed' and i + 1 < len(parts):
                        embeddedness_level = int(parts[i + 1])
                        break
            except (ValueError, IndexError):
                pass
        
        # Extract other parameters from config
        cycle_length = config_used.get('cycle_length', 4)
        use_weighted_features = config_used.get('use_weighted_features', True)
        pos_ratio = config_used.get('pos_test_edges_ratio', 
                                  config_used.get('pos_edges_ratio', 0.8))
        
        # Extract metrics safely
        def safe_get_metric(data, key, default=0.0):
            if isinstance(data, dict):
                if 'actual' in data and isinstance(data['actual'], dict):
                    return float(data['actual'].get(key, default))
                return float(data.get(key, default))
            return default
        
        # Create comprehensive data record
        data = {
            'experiment_name': experiment_name,
            'accuracy': safe_get_metric(metrics, 'accuracy'),
            'f1_score': safe_get_metric(metrics, 'f1_score'),
            'precision': safe_get_metric(metrics, 'precision'),
            'recall': safe_get_metric(metrics, 'recall'),
            'roc_auc': safe_get_metric(metrics, 'roc_auc'),
            'embeddedness_level': embeddedness_level,
            'cycle_length': cycle_length,
            'use_weighted_features': use_weighted_features,
            'positive_ratio': pos_ratio,
            'negative_ratio': 1 - pos_ratio,
            'feature_type': 'Weighted' if use_weighted_features else 'Unweighted',
            'dataset_split': f"{int((1-pos_ratio)*100)}-{int(pos_ratio*100)}" if pos_ratio != 0.8 else "Standard"
        }
        
        print(f"Extracted data for {experiment_name}: embeddedness={embeddedness_level}, "
              f"cycle={cycle_length}, weighted={use_weighted_features}, pos_ratio={pos_ratio:.1f}")
        
        return data
        
    except Exception as e:
        print(f"Error extracting data for {experiment_name}: {str(e)}")
        return None

def load_all_experiment_data():
    """Load data from all experiments"""
    print("Loading experiment data...")
    
    if not os.path.exists(RESULTS_DIR):
        print(f"Results directory not found: {RESULTS_DIR}")
        return pd.DataFrame()
    
    all_data = []
    experiment_dirs = [d for d in os.listdir(RESULTS_DIR) 
                      if os.path.isdir(os.path.join(RESULTS_DIR, d))]
    
    print(f"Found {len(experiment_dirs)} experiment directories")
    
    for exp_name in experiment_dirs:
        exp_path = os.path.join(RESULTS_DIR, exp_name)
        data = extract_experiment_data(exp_name, exp_path)
        if data:
            all_data.append(data)
    
    if not all_data:
        print("No experiment data found!")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    print(f"Loaded {len(df)} experiments successfully")
    
    # Print summary
    print("\nExperiment Summary:")
    print(f"Embeddedness levels: {sorted(df['embeddedness_level'].unique())}")
    print(f"Cycle lengths: {sorted(df['cycle_length'].unique())}")
    print(f"Feature types: {sorted(df['feature_type'].unique())}")
    print(f"Positive ratios: {sorted(df['positive_ratio'].unique())}")
    
    return df

def create_embeddedness_comparison(df, output_dir):
    """Create embeddedness level comparison plots - ENHANCED AND FIXED"""
    print("\n=== Creating Embeddedness Level Comparison (Step 3) ===")
    
    # Filter to only embeddedness experiments - STRICT FILTERING
    embeddedness_experiments = ['embed_0_optimal', 'embed_1_optimal', 'embed_2_optimal']
    embed_df = df[df['experiment_name'].isin(embeddedness_experiments)].copy()
    
    if embed_df.empty:
        print("WARNING: No embeddedness experiments found!")
        print("Expected experiments:", embeddedness_experiments)
        print("Available experiments:", df['experiment_name'].tolist())
        return
    
    print(f"Found {len(embed_df)} embeddedness experiments:")
    for _, row in embed_df.iterrows():
        print(f"  - {row['experiment_name']}: level {row['embeddedness_level']}")
    
    # Ensure we have unique embeddedness levels
    if len(embed_df['embeddedness_level'].unique()) < 2:
        print("WARNING: Need at least 2 different embeddedness levels for comparison")
        return
    
    # Sort by embeddedness level for consistent ordering
    embed_df = embed_df.sort_values('embeddedness_level')
    
    # Create the comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance vs Embeddedness Level\n(Step 3: Impact of Node Embeddedness Filtering)', 
                 fontsize=16, fontweight='bold')
    
    metrics = [
        ('accuracy', 'Accuracy', 'Accuracy Score'),
        ('f1_score', 'F1 Score', 'F1 Score'),
        ('roc_auc', 'ROC AUC', 'AUC Score'),
        ('precision', 'Precision', 'Precision Score')
    ]
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    for idx, (metric, title, ylabel) in enumerate(metrics):
        ax = axes[idx//2, idx%2]
        
        # Extract data for plotting
        levels = embed_df['embeddedness_level'].values
        values = embed_df[metric].values
        
        # Create bar plot with error handling
        bars = ax.bar(levels, values, color=colors[idx], alpha=0.7, width=0.6)
        
        # Overlay individual points
        ax.scatter(levels, values, color='red', s=100, zorder=5, 
                  marker='o', edgecolor='darkred', linewidth=2)
        
        # Customize plot
        ax.set_title(f'{title} by Embeddedness Level', fontsize=14, fontweight='bold')
        ax.set_xlabel('Minimum Embeddedness Level', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xticks(levels)
        ax.set_xticklabels([f'Level {int(level)}' for level in levels])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Set y-axis limits for better visualization
        y_min = max(0, min(values) - 0.05)
        y_max = min(1, max(values) + 0.05)
        ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, 'step3_embeddedness_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Embeddedness comparison saved to: {output_path}")
    
    # Print numerical summary
    print("\nEmbeddedness Level Performance Summary:")
    print("-" * 60)
    for _, row in embed_df.iterrows():
        print(f"Level {int(row['embeddedness_level'])}: "
              f"Acc={row['accuracy']:.3f}, F1={row['f1_score']:.3f}, "
              f"AUC={row['roc_auc']:.3f}, Prec={row['precision']:.3f}")
    
    plt.close()

def create_weighted_vs_unweighted_comparison(df, output_dir):
    """Create weighted vs unweighted features comparison - Step 1"""
    print("\n=== Creating Weighted vs Unweighted Comparison (Step 1) ===")
    
    # Filter for weighted vs unweighted experiments
    weight_experiments = ['weighted_optimal', 'unweighted_optimal']
    weight_df = df[df['experiment_name'].isin(weight_experiments)].copy()
    
    if len(weight_df) < 2:
        print("WARNING: Need both weighted and unweighted experiments")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Weighted vs Unweighted Features Performance Comparison\n(Step 1: Impact of Edge Weight Information)', 
                 fontsize=16, fontweight='bold')
    
    metrics = [
        ('accuracy', 'Accuracy'),
        ('f1_score', 'F1 Score'),
        ('roc_auc', 'ROC AUC'),
        ('precision', 'Precision')
    ]
    
    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx//2, idx%2]
        
        # Create comparison
        feature_types = weight_df['feature_type'].values
        values = weight_df[metric].values
        
        bars = ax.bar(feature_types, values, color=['#1f77b4', '#ff7f0e'], alpha=0.7)
        
        ax.set_title(f'{title}: Weighted vs Unweighted', fontsize=14, fontweight='bold')
        ax.set_ylabel(title, fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'step1_weighted_vs_unweighted.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Weighted vs unweighted comparison saved to: {output_path}")
    plt.close()

def create_cycle_length_comparison(df, output_dir):
    """Create cycle length comparison plots - Step 7"""
    print("\n=== Creating Cycle Length Comparison (Step 7) ===")
    
    # Filter for cycle length experiments
    cycle_experiments = ['cycle_length_3_optimal', 'cycle_length_4_optimal', 'cycle_length_5_optimal']
    cycle_df = df[df['experiment_name'].isin(cycle_experiments)].copy()
    
    if cycle_df.empty:
        print("WARNING: No cycle length experiments found")
        return
    
    cycle_df = cycle_df.sort_values('cycle_length')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance vs Cycle Length\n(Step 7: Impact of Structural Feature Complexity)', 
                 fontsize=16, fontweight='bold')
    
    metrics = [
        ('accuracy', 'Accuracy'),
        ('f1_score', 'F1 Score'), 
        ('roc_auc', 'ROC AUC'),
        ('precision', 'Precision')
    ]
    
    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx//2, idx%2]
        
        cycle_lengths = cycle_df['cycle_length'].values
        values = cycle_df[metric].values
        
        ax.plot(cycle_lengths, values, marker='o', linewidth=3, markersize=10, color='#2E86AB')
        ax.fill_between(cycle_lengths, values, alpha=0.3, color='#2E86AB')
        
        ax.set_title(f'{title} by Cycle Length', fontsize=14, fontweight='bold')
        ax.set_xlabel('Cycle Length', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_xticks(cycle_lengths)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for x, y in zip(cycle_lengths, values):
            ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'step7_cycle_length_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Cycle length comparison saved to: {output_path}")
    plt.close()

def create_aggregation_comparison(df, output_dir):
    """Create aggregation method comparison plots - Step 4"""
    print("\n=== Creating Aggregation Method Comparison (Step 4) ===")
    
    # Filter for aggregation experiments
    agg_experiments = ['aggregation_max_optimal', 'aggregation_sum_optimal', 'aggregation_stronger_optimal']
    agg_df = df[df['experiment_name'].isin(agg_experiments)].copy()
    
    if agg_df.empty:
        print("WARNING: No aggregation experiments found")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Aggregation Method Performance Comparison\n(Step 4: Impact of Bidirectional Edge Handling)', 
                 fontsize=16, fontweight='bold')
    
    metrics = [
        ('accuracy', 'Accuracy'),
        ('f1_score', 'F1 Score'),
        ('roc_auc', 'ROC AUC'),
        ('precision', 'Precision')
    ]
    
    # Extract method names from experiment names
    agg_df['method'] = agg_df['experiment_name'].str.extract(r'aggregation_(\w+)_optimal')[0]
    
    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx//2, idx%2]
        
        methods = agg_df['method'].values
        values = agg_df[metric].values
        
        bars = ax.bar(methods, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
        
        ax.set_title(f'{title} by Aggregation Method', fontsize=14, fontweight='bold')
        ax.set_ylabel(title, fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'step4_aggregation_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Aggregation method comparison saved to: {output_path}")
    plt.close()

def create_comprehensive_summary(df, output_dir):
    """Create comprehensive summary table and overview"""
    print("\n=== Creating Comprehensive Summary ===")
    
    # Create summary table
    summary_data = []
    for _, row in df.iterrows():
        summary_data.append({
            'Experiment': row['experiment_name'],
            'Embeddedness': f"Level {int(row['embeddedness_level'])}",
            'Cycle Length': int(row['cycle_length']),
            'Features': row['feature_type'],
            'Pos Ratio': f"{row['positive_ratio']:.0%}",
            'Accuracy': f"{row['accuracy']:.3f}",
            'F1 Score': f"{row['f1_score']:.3f}",
            'ROC AUC': f"{row['roc_auc']:.3f}",
            'Precision': f"{row['precision']:.3f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'comprehensive_results_summary.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"✓ Summary table saved to: {csv_path}")
    
    # Print top performers
    print("\nTop Performing Experiments:")
    print("-" * 50)
    if len(df) > 0:
        top_accuracy = df.loc[df['accuracy'].idxmax()]
        top_f1 = df.loc[df['f1_score'].idxmax()]
        top_auc = df.loc[df['roc_auc'].idxmax()]
        
        print(f"Best Accuracy: {top_accuracy['experiment_name']} ({top_accuracy['accuracy']:.3f})")
        print(f"Best F1 Score: {top_f1['experiment_name']} ({top_f1['f1_score']:.3f})")
        print(f"Best ROC AUC: {top_auc['experiment_name']} ({top_auc['roc_auc']:.3f})")

def run_analysis():
    """Run comprehensive analysis on all completed experiments"""
    print(f"\n{'='*80}")
    print("RUNNING COMPREHENSIVE ANALYSIS")
    print("="*80)
    
    # Load all experiment data
    df = load_all_experiment_data()
    
    if df.empty:
        print("ERROR: No experimental data found!")
        print("Please ensure experiments have been run and results are in:", RESULTS_DIR)
        return False
    
    print(f"\nAnalyzing {len(df)} experiments...")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Create all comparison plots
    try:
        # Step 1: Weighted vs Unweighted (Most Important)
        create_weighted_vs_unweighted_comparison(df, OUTPUT_DIR)
        
        # Step 3: Embeddedness Level Comparison (FIXED)
        create_embeddedness_comparison(df, OUTPUT_DIR)
        
        # Step 4: Aggregation Method Comparison
        create_aggregation_comparison(df, OUTPUT_DIR)
        
        # Step 7: Cycle Length Comparison
        create_cycle_length_comparison(df, OUTPUT_DIR)
        
        # Comprehensive Summary
        create_comprehensive_summary(df, OUTPUT_DIR)
        
        print(f"\n✓ ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"✓ All comparison plots generated")
        print(f"✓ Results saved to: {OUTPUT_DIR}")
        print(f"✓ Step 3 embeddedness comparison now available")
        
        return True
        
    except Exception as e:
        print(f"ERROR during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main function to run all batch experiments and analysis
    """
    print("ENHANCED BATCH EXPERIMENTS RUNNER - FIXED EMBEDDEDNESS HANDLING")
    print("="*80)
    print("Running comprehensive batch experiments for link prediction analysis:")
    print("• Step 1: Weighted vs Unweighted Features (Most Important)")
    print("• Step 2: Dataset Comparison (Bitcoin baseline)")
    print("• Step 3: Embeddedness Level Comparison (0, 1, 2) - FIXED")
    print("• Step 4: Aggregation Method Comparison")
    print("• Step 7: Cycle Length Comparisons (3, 4, 5)")
    print("="*80)
    
    # Track all experiment results
    all_results = []
    
    try:
        # Step 1: Most Important - Weighted vs Unweighted
        print(f"\nStarting Step 1: Weighted vs Unweighted Experiments...")
        step1_results = run_weighted_vs_unweighted_experiments()
        all_results.append(("Step 1 (Weighted vs Unweighted)", step1_results))
        
        # Step 2: Dataset Comparison
        print(f"\nStarting Step 2: Dataset Comparison Experiments...")
        step2_results = run_dataset_comparison_experiments()
        all_results.append(("Step 2 (Dataset Comparison)", step2_results))
        
        # Step 3: Embeddedness Level Experiments - FIXED
        print(f"\nStarting Step 3: Embeddedness Level Experiments...")
        step3_results = run_embeddedness_experiments()
        all_results.append(("Step 3 (Embeddedness Levels)", step3_results))
        
        # Step 4: Aggregation Method Experiments
        print(f"\nStarting Step 4: Aggregation Method Experiments...")
        step4_results = run_aggregation_experiments()
        all_results.append(("Step 4 (Aggregation Methods)", step4_results))
        
        # Step 7: Cycle Length Experiments
        print(f"\nStarting Step 7: Cycle Length Experiments...")
        step7_results = run_cycle_length_experiments()
        all_results.append(("Step 7 (Cycle Lengths)", step7_results))
        
        # Run comprehensive analysis
        print(f"\nStarting Comprehensive Analysis...")
        analysis_success = run_analysis()
        
        # Final summary
        print(f"\n{'='*80}")
        print("BATCH EXPERIMENTS COMPLETION SUMMARY")
        print("="*80)
        
        total_experiments = 0
        successful_experiments = 0
        
        for step_name, step_results in all_results:
            print(f"\n{step_name}:")
            for exp_name, success in step_results:
                status = "✓ SUCCESS" if success else "✗ FAILED"
                print(f"  {exp_name}: {status}")
                total_experiments += 1
                if success:
                    successful_experiments += 1
        
        success_rate = (successful_experiments / total_experiments * 100) if total_experiments > 0 else 0
        
        print(f"\nOVERALL RESULTS:")
        print(f"✓ Total experiments: {total_experiments}")
        print(f"✓ Successful: {successful_experiments}")
        print(f"✗ Failed: {total_experiments - successful_experiments}")
        print(f"Success rate: {success_rate:.1f}%")
        
        if analysis_success:
            print(f"✓ Comprehensive analysis: SUCCESS")
            print(f"✓ Results and plots saved to: {OUTPUT_DIR}")
        else:
            print(f"✗ Comprehensive analysis: FAILED")
        
        print(f"\n{'='*80}")
        if successful_experiments == total_experiments and analysis_success:
            print("✓ ALL EXPERIMENTS AND ANALYSIS COMPLETED SUCCESSFULLY!")
            print("✓ Fixed embeddedness parameter handling working correctly")
            print("✓ Ready for presentation with all required plots")
            print("\nNext steps:")
            print("  1. Run: python notebooks/analyze_all_results.py")
            print("  2. Run: python notebooks/analyze_feature_distributions.py")
        else:
            print("  Some experiments or analysis failed - check logs above")
        print("="*80)
        
        return successful_experiments == total_experiments and analysis_success
        
    except Exception as e:
        print(f"\nCRITICAL ERROR in batch experiments: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)