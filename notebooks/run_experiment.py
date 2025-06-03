#!/usr/bin/env python3
"""
Single Experiment Runner with Parameter Override Support
======================================================

This script runs individual experiments with specific parameter overrides.
Designed for pos_ratio experiments and other parameter variations.

Key Features:
- Parameter override via --override "key=value;key2=value2" syntax
- Embeddedness filtering applied in preprocessing stage
- Optimal split ratios (74:12:14)
- Automatic experiment naming based on parameters

Usage Examples:
python notebooks/run_experiment.py --override "pos_edges_ratio=0.5;min_train_embeddedness=1"
python notebooks/run_experiment.py --override "pos_edges_ratio=0.6;min_train_embeddedness=1"
python notebooks/run_experiment.py --override "cycle_length=3;min_train_embeddedness=1"
"""

import os
import sys
import argparse
import yaml
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime

# Set project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def parse_override_string(override_str):
    """
    Parse override string like "pos_edges_ratio=0.5;min_train_embeddedness=1"
    Returns dictionary of parameter overrides
    """
    if not override_str:
        return {}
    
    overrides = {}
    parts = override_str.split(';')
    
    for part in parts:
        part = part.strip()
        if '=' in part:
            key, value = part.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            # Convert values to appropriate types
            if value.lower() in ['true', 'false']:
                overrides[key] = value.lower() == 'true'
            elif value.replace('.', '').replace('-', '').isdigit():
                if '.' in value:
                    overrides[key] = float(value)
                else:
                    overrides[key] = int(value)
            else:
                overrides[key] = value
                
            print(f"Parsed override: {key} = {overrides[key]} (type: {type(overrides[key]).__name__})")
    
    return overrides

def load_and_modify_config(config_path, overrides, experiment_name):
    """
    Load base config and apply overrides for this specific experiment
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        return None
    
    print(f"Base config loaded successfully")
    
    # Apply overrides
    for key, value in overrides.items():
        original_value = config.get(key, "not set")
        config[key] = value
        print(f"Override applied: {key} = {value} (was: {original_value})")
    
    # Set experiment-specific parameters
    config['default_experiment_name'] = experiment_name
    
    # Ensure optimal split configuration
    config['num_test_edges'] = 3080
    config['num_validation_edges'] = 2640
    config['optimal_split'] = True
    
    # Default values for consistency
    if 'min_train_embeddedness' not in config:
        config['min_train_embeddedness'] = 1
    if 'min_val_embeddedness' not in config:
        config['min_val_embeddedness'] = 1
    if 'min_test_embeddedness' not in config:
        config['min_test_embeddedness'] = 1
    if 'cycle_length' not in config:
        config['cycle_length'] = 4
    if 'bidirectional_method' not in config:
        config['bidirectional_method'] = 'max'
    if 'use_weighted_features' not in config:
        config['use_weighted_features'] = False
    
    return config

def generate_experiment_name(overrides):
    """
    Generate experiment name based on key parameters
    Priority: pos_edges_ratio > min_train_embeddedness > cycle_length > use_weighted_features
    """
    print(f"Generating experiment name from overrides: {overrides}")
    
    # Extract key parameters
    pos_ratio = overrides.get('pos_edges_ratio')
    min_embeddedness = overrides.get('min_train_embeddedness')
    cycle_length = overrides.get('cycle_length')
    weighted = overrides.get('use_weighted_features')
    bidirectional_method = overrides.get('bidirectional_method')
    
    # Generate name based on most significant parameter
    if pos_ratio is not None:
        pos_pct = int(pos_ratio * 100)
        neg_pct = 100 - pos_pct
        name = f"pos_ratio_{pos_pct}_{neg_pct}_optimal"
        print(f"Generated name based on pos_ratio: {name}")
        return name
        
    elif min_embeddedness is not None and min_embeddedness != 1:  # Only if different from default
        name = f"embed_{min_embeddedness}_optimal"
        print(f"Generated name based on embeddedness: {name}")
        return name
        
    elif cycle_length is not None and cycle_length != 4:  # Only if different from default
        name = f"cycle_length_{cycle_length}_optimal"
        print(f"Generated name based on cycle_length: {name}")
        return name
        
    elif weighted is not None:
        feature_type = "weighted" if weighted else "unweighted"
        name = f"{feature_type}_optimal"
        print(f"Generated name based on feature type: {name}")
        return name
        
    elif bidirectional_method is not None and bidirectional_method != 'max':  # Only if different from default
        name = f"aggregation_{bidirectional_method}_optimal"
        print(f"Generated name based on aggregation method: {name}")
        return name
        
    else:
        name = "single_experiment_optimal"
        print(f"Generated default name: {name}")
        return name

def run_experiment_pipeline(config, experiment_name):
    """
    Run the complete experiment pipeline: preprocess -> train -> validate -> test
    """
    print(f"\n{'='*60}")
    print(f"RUNNING EXPERIMENT PIPELINE: {experiment_name}")
    print(f"{'='*60}")
    
    # Extract key parameters for command line
    min_embeddedness = config.get('min_train_embeddedness', 1)
    use_weighted = config.get('use_weighted_features', False)
    bidirectional_method = config.get('bidirectional_method', 'max')
    cycle_length = config.get('cycle_length', 4)
    
    print(f"Key experiment parameters:")
    print(f"  Experiment name: {experiment_name}")
    print(f"  min_train_embeddedness: {min_embeddedness}")
    print(f"  use_weighted_features: {use_weighted}")
    print(f"  bidirectional_method: {bidirectional_method}")
    print(f"  cycle_length: {cycle_length}")
    
    try:
        # Step 1: Preprocessing with embeddedness filtering
        print(f"\nStep 1: Running preprocessing...")
        preprocess_cmd = [
            sys.executable, 'notebooks/preprocess.py',
            '--name', experiment_name,
            '--min_embeddedness', str(min_embeddedness),
            '--bidirectional_method', bidirectional_method
        ]
        
        if use_weighted:
            preprocess_cmd.append('--use_weighted_features')
        
        print(f"  Command: {' '.join(preprocess_cmd)}")
        result = subprocess.run(preprocess_cmd, cwd=PROJECT_ROOT, 
                               capture_output=True, text=True, timeout=1800)
        
        if result.returncode != 0:
            print(f"  ✗ Preprocessing failed")
            print(f"  Error: {result.stderr}")
            return False
        print(f"  ✓ Preprocessing completed successfully")
        
        # Step 2: Training
        print(f"\nStep 2: Running training...")
        train_cmd = [
            sys.executable, 'notebooks/train_model.py',
            '--name', experiment_name,
            '--cycle_length', str(cycle_length)
        ]
        
        print(f"  Command: {' '.join(train_cmd)}")
        result = subprocess.run(train_cmd, cwd=PROJECT_ROOT,
                               capture_output=True, text=True, timeout=1800)
        
        if result.returncode != 0:
            print(f"  ✗ Training failed")
            print(f"  Error: {result.stderr}")
            return False
        print(f"  ✓ Training completed successfully")
        
        # Step 3: Validation
        print(f"\nStep 3: Running validation...")
        validate_cmd = [
            sys.executable, 'notebooks/validate_model.py',
            '--name', experiment_name,
            '--cycle_length', str(cycle_length)
        ]
        
        result = subprocess.run(validate_cmd, cwd=PROJECT_ROOT,
                               capture_output=True, text=True, timeout=1800)
        
        if result.returncode != 0:
            print(f"  ✗ Validation failed")
            print(f"  Error: {result.stderr}")
            return False
        print(f"  ✓ Validation completed successfully")
        
        # Step 4: Testing
        print(f"\nStep 4: Running testing...")
        test_cmd = [
            sys.executable, 'notebooks/test_model.py',
            '--name', experiment_name,
            '--cycle_length', str(cycle_length)
        ]
        
        result = subprocess.run(test_cmd, cwd=PROJECT_ROOT,
                               capture_output=True, text=True, timeout=1800)
        
        if result.returncode != 0:
            print(f"  ✗ Testing failed")
            print(f"  Error: {result.stderr}")
            return False
        print(f"  ✓ Testing completed successfully")
        
        # Verify results
        metrics_path = os.path.join(PROJECT_ROOT, 'results', experiment_name, 'testing', 'metrics.json')
        if os.path.exists(metrics_path):
            print(f"  ✓ Results verified: {metrics_path}")
            
            # Load and display key metrics
            try:
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                
                if 'actual' in metrics:
                    actual = metrics['actual']
                    print(f"  Key Results:")
                    print(f"    Accuracy: {actual.get('accuracy', 0):.4f}")
                    print(f"    F1 Score: {actual.get('f1_score', 0):.4f}")
                    print(f"    ROC AUC: {actual.get('roc_auc', 0):.4f}")
                
            except Exception as e:
                print(f"  Warning: Could not load metrics: {e}")
            
            # Save config to results directory
            config_save_path = os.path.join(PROJECT_ROOT, 'results', experiment_name, 'config_used.yaml')
            os.makedirs(os.path.dirname(config_save_path), exist_ok=True)
            with open(config_save_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False)
            print(f"  ✓ Configuration saved for future reference")
            
            return True
        else:
            print(f"  ✗ Results not found: {metrics_path}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  ✗ Pipeline step timed out")
        return False
    except Exception as e:
        print(f"  ✗ Pipeline failed with exception: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Run single experiment with parameter overrides",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run pos_ratio experiments
  python notebooks/run_experiment.py --override "pos_edges_ratio=0.5;min_train_embeddedness=1"
  python notebooks/run_experiment.py --override "pos_edges_ratio=0.6;min_train_embeddedness=1"
  python notebooks/run_experiment.py --override "pos_edges_ratio=0.7;min_train_embeddedness=1"

  # Run embeddedness experiments  
  python notebooks/run_experiment.py --override "min_train_embeddedness=0;pos_edges_ratio=0.5"
  python notebooks/run_experiment.py --override "min_train_embeddedness=1;pos_edges_ratio=0.5"
  python notebooks/run_experiment.py --override "min_train_embeddedness=2;pos_edges_ratio=0.5"

  # Run cycle length experiments
  python notebooks/run_experiment.py --override "cycle_length=3;min_train_embeddedness=1"
  python notebooks/run_experiment.py --override "cycle_length=4;min_train_embeddedness=1"
  python notebooks/run_experiment.py --override "cycle_length=5;min_train_embeddedness=1"

  # Run weighted vs unweighted experiments
  python notebooks/run_experiment.py --override "use_weighted_features=true;min_train_embeddedness=1"
  python notebooks/run_experiment.py --override "use_weighted_features=false;min_train_embeddedness=1"

  # Run aggregation method experiments
  python notebooks/run_experiment.py --override "bidirectional_method=max;min_train_embeddedness=1"
  python notebooks/run_experiment.py --override "bidirectional_method=sum;min_train_embeddedness=1"
  python notebooks/run_experiment.py --override "bidirectional_method=stronger;min_train_embeddedness=1"
        """
    )
    
    parser.add_argument('--config', type=str, default='config.yaml',
                       help="Base config file to use")
    parser.add_argument('--override', type=str, required=True,
                       help="Parameter overrides in format 'key=value;key2=value2'")
    parser.add_argument('--name', type=str, default=None,
                       help="Experiment name (auto-generated if not provided)")
    
    args = parser.parse_args()
    
    # Find config file
    config_paths = [
        os.path.join(PROJECT_ROOT, args.config),
        os.path.join(PROJECT_ROOT, 'config.yaml'),
        os.path.join(PROJECT_ROOT, 'configs', args.config)
    ]
    
    config_path = None
    for path in config_paths:
        if os.path.exists(path):
            config_path = path
            break
    
    if not config_path:
        print(f"Error: Config file not found. Tried:")
        for path in config_paths:
            print(f"  - {path}")
        return False
    
    print(f"Using config file: {config_path}")
    
    # Parse overrides
    overrides = parse_override_string(args.override)
    if not overrides:
        print("Error: No valid overrides found")
        return False
    
    print(f"Parsed overrides: {overrides}")
    
    # Generate experiment name
    experiment_name = args.name or generate_experiment_name(overrides)
    print(f"Experiment name: {experiment_name}")
    
    # Load and modify config
    config = load_and_modify_config(config_path, overrides, experiment_name)
    if not config:
        return False
    
    # Print final configuration summary
    print(f"\nFinal Configuration Summary:")
    key_params = ['pos_edges_ratio', 'min_train_embeddedness', 'cycle_length', 
                  'use_weighted_features', 'bidirectional_method']
    for param in key_params:
        if param in config:
            print(f"  {param}: {config[param]}")
    
    # Run experiment
    start_time = time.time()
    success = run_experiment_pipeline(config, experiment_name)
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    if success:
        print(f"✓ EXPERIMENT {experiment_name} COMPLETED SUCCESSFULLY!")
        print(f"  Execution time: {elapsed_time/60:.1f} minutes")
        print(f"  Results saved to: results/{experiment_name}/")
        print(f"  Key files generated:")
        print(f"    - results/{experiment_name}/testing/metrics.json")
        print(f"    - results/{experiment_name}/config_used.yaml")
    else:
        print(f"✗ EXPERIMENT {experiment_name} FAILED!")
        print(f"  Execution time: {elapsed_time/60:.1f} minutes")
        print(f"  Check error messages above for details")
    print(f"{'='*60}")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)