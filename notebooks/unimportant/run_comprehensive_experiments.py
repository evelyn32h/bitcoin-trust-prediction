#!/usr/bin/env python3
"""
Comprehensive Experiment Runner - Generate All Required Variations
================================================================

This script systematically generates all experiment variations needed to fix:
- Step 3: Embeddedness levels (0, 1, 2) 
- Step 6: Positive ratios (0.9, 0.8, 0.7, 0.6, 0.5)
- Step 7: Cycle lengths (3, 4, 5)
- Step 8: Pos/neg ratios (90%-10%, 80%-20%, 70%-30%, 60%-40%, 50%-50%)

All experiments use the OPTIMAL 74:12:14 split (test=3080, validation=2640)

Usage: python run_comprehensive_experiments.py
"""

import os
import sys
import subprocess
import time
import yaml
import itertools
from pathlib import Path

# Set project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(PROJECT_ROOT)

def create_experiment_config(base_config, variations, experiment_name):
    """
    Create a specific experiment configuration with given variations
    """
    config = base_config.copy()
    
    # Update with variations
    for key, value in variations.items():
        if key == 'cycle_length':
            config['cycle_length'] = value
        elif key == 'embeddedness_level':
            config['min_train_embeddedness'] = value
            config['min_val_embeddedness'] = value
            config['min_test_embeddedness'] = value
        elif key == 'positive_ratio':
            config['pos_train_edges_ratio'] = value
            config['pos_edges_ratio'] = value
            config['pos_test_edges_ratio'] = value
    
    # Set experiment name
    config['default_experiment_name'] = experiment_name
    
    # Ensure optimal split settings are maintained
    config['num_test_edges'] = 3080
    config['num_validation_edges'] = 2640
    config['split_ratio'] = {
        'train': 0.74,
        'validation': 0.12,
        'test': 0.14
    }
    config['optimal_split'] = True
    config['split_type'] = 'optimal_74_12_14'
    
    return config

def load_base_config():
    """
    Load base configuration
    """
    base_config = {
        'data_path': 'data/soc-sign-epinions-best-balance.txt',
        'default_n_folds': 5,
        'cycle_length': 4,  # Default
        'num_test_edges': 3080,
        'num_validation_edges': 2640,
        'default_threshold': 0.5,
        'min_train_embeddedness': 1,  # Default
        'pos_train_edges_ratio': 0.8,  # Default
        'nr_val_predictions': 200,
        'min_val_embeddedness': 1,  # Default
        'pos_edges_ratio': 0.8,  # Default
        'threshold_type': 'best_accuracy_threshold',
        'n_test_predictions': 300,
        'min_test_embeddedness': 1,  # Default
        'pos_test_edges_ratio': 0.8,  # Default
        'use_weighted_features': False,  # Unweighted performs better
        'weight_method': 'raw',
        'weight_bins': 5,
        'preserve_original_weights': True,
        'bidirectional_method': 'max',
        'enable_subset_sampling': True,
        'subset_sampling_method': 'bfs_sampling',
        'target_edge_count': 35000,
        'subset_preserve_structure': True,
        'bfs_seed_selection': 'random_moderate_degree',
        'bfs_degree_percentile': 70,
        'split_ratio': {
            'train': 0.74,
            'validation': 0.12,
            'test': 0.14
        },
        'memory_optimized': True,
        'parallel_processing': True,
        'optimal_split': True,
        'split_type': 'optimal_74_12_14'
    }
    return base_config

def generate_experiment_plans():
    """
    Generate all experiment plans to fix the identified issues
    """
    experiments = []
    
    # Step 3 Fix: Embeddedness Level Comparison (0, 1, 2)
    # Keep other parameters at optimal values
    for embeddedness in [0, 1, 2]:
        exp_name = f"optimal_embed_{embeddedness}_cycle_4_pos_80"
        variations = {
            'embeddedness_level': embeddedness,
            'cycle_length': 4,
            'positive_ratio': 0.8
        }
        experiments.append({
            'name': exp_name,
            'variations': variations,
            'purpose': f'Step 3: Embeddedness level {embeddedness} comparison',
            'priority': 'high'
        })
    
    # Step 6 Fix: Positive Ratio Comparison (0.9, 0.8, 0.7, 0.6, 0.5)
    # Keep other parameters at optimal values
    for pos_ratio in [0.9, 0.8, 0.7, 0.6, 0.5]:
        exp_name = f"optimal_embed_1_cycle_4_pos_{int(pos_ratio*100)}"
        variations = {
            'embeddedness_level': 1,
            'cycle_length': 4,
            'positive_ratio': pos_ratio
        }
        ratio_desc = f"{pos_ratio:.0%}-{1-pos_ratio:.0%}"
        experiments.append({
            'name': exp_name,
            'variations': variations,
            'purpose': f'Step 6: Positive ratio {ratio_desc} comparison',
            'priority': 'high'
        })
    
    # Step 7 Fix: Cycle Length Comparison (3, 4, 5)
    # Keep other parameters at optimal values
    for cycle_len in [3, 4, 5]:
        exp_name = f"optimal_embed_1_cycle_{cycle_len}_pos_80"
        variations = {
            'embeddedness_level': 1,
            'cycle_length': cycle_len,
            'positive_ratio': 0.8
        }
        experiments.append({
            'name': exp_name,
            'variations': variations,
            'purpose': f'Step 7: Cycle length {cycle_len} comparison',
            'priority': 'high'
        })
    
    # Step 8 Fix: Comprehensive Pos/Neg Ratio Experiments
    # Fixed optimal splitting scheme with varied ratios
    ratios_step8 = [0.9, 0.8, 0.7, 0.6, 0.5]
    for pos_ratio in ratios_step8:
        exp_name = f"optimal_step8_pos_{int(pos_ratio*100)}_neg_{int((1-pos_ratio)*100)}"
        variations = {
            'embeddedness_level': 1,  # Optimal
            'cycle_length': 4,        # Optimal
            'positive_ratio': pos_ratio
        }
        ratio_desc = f"{pos_ratio:.0%}-{1-pos_ratio:.0%}"
        experiments.append({
            'name': exp_name,
            'variations': variations,
            'purpose': f'Step 8: Pos/neg ratio {ratio_desc} with fixed optimal split',
            'priority': 'high'
        })
    
    # Additional baseline experiments for comparison
    # Weighted vs Unweighted with optimal split (Step 1)
    for use_weighted in [False, True]:
        weight_type = "weighted" if use_weighted else "unweighted"
        exp_name = f"optimal_baseline_{weight_type}_embed_1_cycle_4_pos_80"
        variations = {
            'embeddedness_level': 1,
            'cycle_length': 4,
            'positive_ratio': 0.8,
            'use_weighted_features': use_weighted
        }
        experiments.append({
            'name': exp_name,
            'variations': variations,
            'purpose': f'Step 1: {weight_type.title()} features baseline',
            'priority': 'medium'
        })
    
    return experiments

def save_experiment_config(config, config_path):
    """
    Save experiment configuration to file
    """
    config_dir = os.path.dirname(config_path)
    os.makedirs(config_dir, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

def run_single_experiment(experiment, base_config, dry_run=False):
    """
    Run a single experiment
    """
    exp_name = experiment['name']
    variations = experiment['variations']
    purpose = experiment['purpose']
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"PURPOSE: {purpose}")
    print(f"VARIATIONS: {variations}")
    print(f"{'='*60}")
    
    # Create experiment-specific config
    exp_config = create_experiment_config(base_config, variations, exp_name)
    
    # Save config
    config_path = f"configs/experiment_{exp_name}.yaml"
    save_experiment_config(exp_config, config_path)
    print(f"Config saved to: {config_path}")
    
    if dry_run:
        print("DRY RUN: Skipping actual experiment execution")
        return True
    
    # Run experiment
    try:
        start_time = time.time()
        
        # Check if experiment script exists
        experiment_script = "notebooks/run_experiment.py"
        if not os.path.exists(experiment_script):
            print(f"Warning: Experiment script not found: {experiment_script}")
            return False
        
        # Run experiment command
        cmd = [
            sys.executable, 
            experiment_script,
            "--config", config_path,
            "--name", exp_name
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, 
                              capture_output=False, text=True)
        
        elapsed = time.time() - start_time
        print(f"✅ Experiment completed successfully in {elapsed:.1f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"❌ Experiment failed after {elapsed:.1f} seconds with exit code {e.returncode}")
        return False
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Experiment failed after {elapsed:.1f} seconds: {str(e)}")
        return False

def main():
    """
    Main function to run comprehensive experiments
    """
    print("🚀 COMPREHENSIVE EXPERIMENT RUNNER")
    print("="*80)
    print("Generating experiments to fix all identified issues:")
    print("• Step 3: Embeddedness levels (0, 1, 2)")
    print("• Step 6: Positive ratios (90%, 80%, 70%, 60%, 50%)")
    print("• Step 7: Cycle lengths (3, 4, 5)")
    print("• Step 8: Pos/neg ratios with fixed optimal split")
    print("• All using OPTIMAL 74:12:14 split (test=3080, val=2640)")
    print("="*80)
    
    # Load base configuration
    base_config = load_base_config()
    
    # Generate experiment plans
    experiments = generate_experiment_plans()
    
    print(f"\nGenerated {len(experiments)} experiments:")
    for i, exp in enumerate(experiments, 1):
        priority = exp['priority'].upper()
        print(f"{i:2d}. [{priority:6s}] {exp['name']}: {exp['purpose']}")
    
    # Ask user for confirmation
    response = input(f"\nRun all {len(experiments)} experiments? (y/n/d for dry-run): ").lower().strip()
    
    if response not in ['y', 'yes', 'd', 'dry']:
        print("Experiment run cancelled by user")
        return
    
    dry_run = response in ['d', 'dry']
    if dry_run:
        print("Running in DRY RUN mode - configs will be generated but experiments won't run")
    
    # Create configs directory
    os.makedirs("configs", exist_ok=True)
    
    # Run experiments
    start_time = time.time()
    successful = 0
    failed = 0
    
    for i, experiment in enumerate(experiments, 1):
        print(f"\n📊 PROGRESS: {i}/{len(experiments)} experiments")
        
        success = run_single_experiment(experiment, base_config, dry_run)
        
        if success:
            successful += 1
        else:
            failed += 1
            
        # Small delay between experiments
        if not dry_run and i < len(experiments):
            print("Waiting 2 seconds before next experiment...")
            time.sleep(2)
    
    # Final summary
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print("🎯 COMPREHENSIVE EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"✅ Successful: {successful}/{len(experiments)}")
    print(f"❌ Failed: {failed}/{len(experiments)}")
    
    if successful == len(experiments):
        print(f"\n🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print(f"📁 Results saved to: results/")
        print(f"📊 Ready for analysis with fixed parameters")
        print(f"\nNext steps:")
        print(f"1. Run: python notebooks/analyze_all_results.py")
        print(f"2. Run: python notebooks/analyze_feature_distributions.py")
        print(f"3. All plots will show proper comparisons for Steps 3, 6, 7, 8")
    elif successful > 0:
        print(f"\n⚠️  PARTIAL SUCCESS: {successful} experiments completed")
        print(f"You can still run analysis on completed experiments")
    else:
        print(f"\n❌ ALL EXPERIMENTS FAILED")
        print(f"Please check the error messages above")
    
    print(f"\n🔧 ISSUES ADDRESSED:")
    print(f"• Step 3: Embeddedness comparison (0, 1, 2) - {3} experiments")
    print(f"• Step 6: Positive ratio comparison - {5} experiments")
    print(f"• Step 7: Cycle length comparison (3, 4, 5) - {3} experiments")
    print(f"• Step 8: Pos/neg ratio experiments - {5} experiments")
    print(f"• Baseline weighted/unweighted - {2} experiments")
    print(f"📈 Total variations: {len(experiments)} experiments")
    print("="*80)

if __name__ == "__main__":
    main()