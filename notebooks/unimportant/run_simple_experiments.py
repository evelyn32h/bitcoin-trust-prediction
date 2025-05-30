#!/usr/bin/env python3
"""
Simple Experiment Runner - Windows Compatible
===========================================

Simplified version without Unicode characters for Windows compatibility.
Runs weighted vs unweighted experiments with optimal 74:12:14 split.

Usage: python run_simple_experiments.py
"""

import os
import sys
import yaml
import shutil

# Set project root as working directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(PROJECT_ROOT)

def load_config():
    """Load base configuration"""
    config_path = os.path.join(PROJECT_ROOT, 'config.yaml')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return None

def save_config(config):
    """Save configuration"""
    config_path = os.path.join(PROJECT_ROOT, 'config.yaml')
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False

def backup_config():
    """Backup original config"""
    original = os.path.join(PROJECT_ROOT, 'config.yaml')
    backup = os.path.join(PROJECT_ROOT, 'config_backup.yaml')
    
    if os.path.exists(original):
        shutil.copy2(original, backup)
        print("Config backed up successfully")
        return True
    return False

def restore_config():
    """Restore original config"""
    original = os.path.join(PROJECT_ROOT, 'config.yaml')
    backup = os.path.join(PROJECT_ROOT, 'config_backup.yaml')
    
    if os.path.exists(backup):
        shutil.copy2(backup, original)
        print("Config restored successfully")

def create_weighted_config(base_config):
    """Create weighted features configuration"""
    config = base_config.copy()
    
    # Set weighted features
    config['use_weighted_features'] = True
    config['default_experiment_name'] = 'weighted_optimal_test'
    
    # Ensure optimal split
    config['num_test_edges'] = 3080
    config['num_validation_edges'] = 2640
    config['min_train_embeddedness'] = 1
    config['min_val_embeddedness'] = 1
    config['min_test_embeddedness'] = 1
    
    return config

def create_unweighted_config(base_config):
    """Create unweighted features configuration"""
    config = base_config.copy()
    
    # Set unweighted features
    config['use_weighted_features'] = False
    config['default_experiment_name'] = 'unweighted_optimal_test'
    
    # Ensure optimal split
    config['num_test_edges'] = 3080
    config['num_validation_edges'] = 2640
    config['min_train_embeddedness'] = 1
    config['min_val_embeddedness'] = 1
    config['min_test_embeddedness'] = 1
    
    return config

def print_config_summary(config):
    """Print configuration summary"""
    print("\n" + "="*50)
    print("CONFIGURATION SUMMARY")
    print("="*50)
    print(f"Experiment name: {config.get('default_experiment_name', 'Unknown')}")
    print(f"Use weighted features: {config.get('use_weighted_features', False)}")
    print(f"Test edges: {config.get('num_test_edges', 'Not set')}")
    print(f"Validation edges: {config.get('num_validation_edges', 'Not set')}")
    print(f"Embeddedness level: {config.get('min_train_embeddedness', 'Not set')}")
    print(f"Data path: {config.get('data_path', 'Not set')}")
    print("="*50)

def main():
    """Main function"""
    print("SIMPLE EXPERIMENT RUNNER")
    print("="*50)
    print("This will create configurations for weighted vs unweighted comparison")
    print("Using optimal 74:12:14 split ratio")
    print("="*50)
    
    # Load base config
    base_config = load_config()
    if not base_config:
        print("Failed to load config.yaml")
        return
    
    # Backup original config
    if not backup_config():
        print("Failed to backup config")
        return
    
    try:
        # Create weighted config
        print("\n1. Creating WEIGHTED features configuration...")
        weighted_config = create_weighted_config(base_config)
        print_config_summary(weighted_config)
        
        # Save weighted config
        save_config(weighted_config)
        print("Weighted config saved to config.yaml")
        print("\nNow run: python notebooks/run_experiment.py --name weighted_optimal_test")
        
        input("\nPress Enter after running weighted experiment...")
        
        # Create unweighted config
        print("\n2. Creating UNWEIGHTED features configuration...")
        unweighted_config = create_unweighted_config(base_config)
        print_config_summary(unweighted_config)
        
        # Save unweighted config
        save_config(unweighted_config)
        print("Unweighted config saved to config.yaml")
        print("\nNow run: python notebooks/run_experiment.py --name unweighted_optimal_test")
        
        input("\nPress Enter after running unweighted experiment...")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Restore original config
        restore_config()
        print("Original config restored")
    
    print("\n" + "="*50)
    print("NEXT STEPS:")
    print("="*50)
    print("1. Compare results in results/weighted_optimal_test/ and results/unweighted_optimal_test/")
    print("2. Run: python notebooks/run_complete_analysis.py")
    print("3. Check plots in plots/complete_results_analysis/")
    print("="*50)

if __name__ == "__main__":
    main()