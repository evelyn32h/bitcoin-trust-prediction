#!/usr/bin/env python3
"""
run_experiment.py - Orchestrates the full machine learning pipeline

This script runs the complete experiment pipeline consisting of:
1. Preprocessing (preprocess.py)
2. Training (train_model.py) 
3. Validation (validate_model.py)
4. Testing (test_model.py)
5. Analysis (analyze_results.py)

You can selectively enable/disable pipeline steps using command line arguments.
You can also specify a custom config file to override default parameters.
"""

import os
import sys
import argparse
import subprocess
import time
import yaml
from datetime import datetime

# Set project root as working directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(PROJECT_ROOT)

# Add project root to sys.path for src imports
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def load_config(config_path=None):
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file. If None, uses default config.yaml
    
    Returns:
        dict: Configuration dictionary
    """
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, 'config.yaml')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    print(f"Loading configuration from: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def run_command(cmd, description, cwd=None):
    """
    Run a shell command and handle errors gracefully.
    
    Args:
        cmd: List of command arguments
        description: Human-readable description of the command
        cwd: Working directory (defaults to PROJECT_ROOT)
    
    Returns:
        bool: True if command succeeded, False otherwise
    """
    if cwd is None:
        cwd = PROJECT_ROOT
    
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=False)
        elapsed = time.time() - start_time
        print(f"\nSUCCESS: {description} completed successfully in {elapsed:.1f} seconds")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\nFAILED: {description} failed after {elapsed:.1f} seconds with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\nFAILED: {description} failed: Command not found")
        return False

def run_preprocess(args, config):
    """Run the preprocessing step."""
    cmd = ['python', 'notebooks/preprocess.py', '--name', args.name]
    
    # Add optional arguments
    if args.bidirectional_method:
        cmd.extend(['--bidirectional_method', args.bidirectional_method])
    if args.use_weighted_features:
        cmd.append('--use_weighted_features')
    if args.weight_method:
        cmd.extend(['--weight_method', args.weight_method])
    
    return run_command(cmd, "Data Preprocessing")

def run_training(args, config):
    """Run the training step."""
    cmd = ['python', 'notebooks/train_model.py', 
           '--name', args.name,
           '--n_folds', str(args.n_folds),
           '--cycle_length', str(args.cycle_length)]
    
    return run_command(cmd, "Model Training")

def run_validation(args, config):
    """Run the validation step."""
    cmd = ['python', 'notebooks/validate_model.py',
           '--name', args.name,
           '--n_folds', str(args.n_folds),
           '--cycle_length', str(args.cycle_length)]
    
    # Add optional validation-specific arguments
    if args.predictions_per_fold:
        cmd.extend(['--predictions_per_fold', str(args.predictions_per_fold)])
    if args.pos_edges_ratio:
        cmd.extend(['--pos_edges_ratio', str(args.pos_edges_ratio)])
    
    return run_command(cmd, "Model Validation")

def run_testing(args, config):
    """Run the testing step."""
    cmd = ['python', 'notebooks/test_model.py',
           '--name', args.name,
           '--n_folds', str(args.n_folds),
           '--cycle_length', str(args.cycle_length)]
    
    # Add optional testing-specific arguments
    if args.threshold_type:
        cmd.extend(['--threshold_type', args.threshold_type])
    
    return run_command(cmd, "Model Testing")

def run_analysis(args, config):
    """Run the results analysis step."""
    cmd = ['python', 'notebooks/analyze_results.py', '--name', args.name]
    
    return run_command(cmd, "Results Analysis")

def print_experiment_summary(args, config, results, total_time):
    """Print a summary of the experiment run."""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"Experiment Name: {args.name}")
    print(f"Config File: {args.config or 'config.yaml (default)'}")
    print(f"Start Time: {args._start_time}")
    print(f"Total Runtime: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Configuration:")
    print(f"  - N Folds: {args.n_folds}")
    print(f"  - Cycle Length: {args.cycle_length}")
    print(f"  - Weighted Features: {args.use_weighted_features}")
    if args.use_weighted_features:
        print(f"  - Weight Method: {args.weight_method}")
        print(f"  - Bidirectional Method: {args.bidirectional_method}")
    
    # Print key config parameters if available
    if config:
        if 'min_train_embeddedness' in config:
            print(f"  - Min Train Embeddedness: {config['min_train_embeddedness']}")
        if 'min_test_embeddedness' in config:
            print(f"  - Min Test Embeddedness: {config['min_test_embeddedness']}")
        if 'use_weighted_features' in config:
            print(f"  - Config Weighted Features: {config['use_weighted_features']}")
    
    print(f"\nPipeline Steps:")
    step_names = ['Preprocessing', 'Training', 'Validation', 'Testing', 'Analysis']
    step_flags = [args.preprocess, args.train, args.validate, args.test, args.analyze]
    
    for name, enabled, result in zip(step_names, step_flags, results):
        if enabled:
            status = "SUCCESS" if result else "FAILED"
            print(f"  - {name}: {status}")
        else:
            print(f"  - {name}: SKIPPED")
    
    # Overall status
    if all(result for result, enabled in zip(results, step_flags) if enabled):
        print(f"\nSUCCESS: Experiment completed successfully!")
    else:
        print(f"\nWARNING: Experiment completed with some failures.")
    
    print(f"{'='*80}")

def main():
    parser = argparse.ArgumentParser(
        description="Run the complete machine learning experiment pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with default config
  python run_experiment.py --name my_experiment

  # Run with custom config file
  python run_experiment.py --name my_experiment --config config_custom.yaml

  # Run only preprocessing and training (using short flags)
  python run_experiment.py --name my_experiment -p -tr

  # Run only preprocessing, training and validation (using short flags)
  python run_experiment.py --name my_experiment -p -tr -v

  # Run only testing and analysis (using short flags)
  python run_experiment.py --name my_experiment -te -a

  # Run all except validation and testing (using --no-* flags)
  python run_experiment.py --name my_experiment --no-validate --no-test

  # Run with weighted features
  python run_experiment.py --name weighted_exp --use_weighted_features --weight_method raw

  # Run only analysis on existing results
  python run_experiment.py --name existing_exp -a
        """
    )
    
    # Configuration file argument - ADD THIS FIRST
    parser.add_argument('--config', type=str, 
                       help="Path to YAML config file (overrides default config.yaml)")
    
    # Load config early to get defaults
    # We need to parse known args first to get the config path
    known_args, remaining_args = parser.parse_known_args()
    config = load_config(known_args.config)
    
    # Extract defaults from config
    DEFAULT_EXPERIMENT_NAME = config.get('default_experiment_name', 'default_experiment')
    N_FOLDS = config.get('default_n_folds', 5)
    CYCLE_LENGTH = config.get('cycle_length', 4)
    
    # Experiment configuration
    parser.add_argument('--name', type=str, default=DEFAULT_EXPERIMENT_NAME,
                       help=f"Experiment name (default: {DEFAULT_EXPERIMENT_NAME})")
    parser.add_argument('--n_folds', type=int, default=N_FOLDS,
                       help=f"Number of folds for cross-validation (default: {N_FOLDS})")
    parser.add_argument('--cycle_length', type=int, default=CYCLE_LENGTH,
                       help=f"Cycle length for feature extraction (default: {CYCLE_LENGTH})")
    
    # Pipeline step control - two modes:
    # Mode 1: Disable specific steps (--no-* flags) - all enabled by default
    parser.add_argument('--no-preprocess', dest='preprocess', action='store_false',
                       help="Skip preprocessing step")
    parser.add_argument('--no-train', dest='train', action='store_false',
                       help="Skip training step")
    parser.add_argument('--no-validate', dest='validate', action='store_false',
                       help="Skip validation step")
    parser.add_argument('--no-test', dest='test', action='store_false',
                       help="Skip testing step")
    parser.add_argument('--no-analyze', dest='analyze', action='store_false',
                       help="Skip analysis step")
    
    # Mode 2: Enable only specific steps (short flags) - all disabled by default when any short flag is used
    parser.add_argument('-p', '--only-preprocess', dest='only_preprocess', action='store_true',
                       help="Run only preprocessing step")
    parser.add_argument('-tr', '--only-train', dest='only_train', action='store_true',
                       help="Run only training step")
    parser.add_argument('-v', '--only-validate', dest='only_validate', action='store_true',
                       help="Run only validation step")
    parser.add_argument('-te', '--only-test', dest='only_test', action='store_true',
                       help="Run only testing step")
    parser.add_argument('-a', '--only-analyze', dest='only_analyze', action='store_true',
                       help="Run only analysis step")
    
    # Set defaults for pipeline steps (all enabled by default unless "only" flags are used)
    parser.set_defaults(preprocess=True, train=True, validate=True, test=True, analyze=True)
    
    # Preprocessing arguments
    parser.add_argument('--bidirectional_method', type=str,
                       help="Method for handling bidirectional edges")
    parser.add_argument('--use_weighted_features', action='store_true',
                       help="Enable weighted features")
    parser.add_argument('--weight_method', type=str,
                       help="Weight processing method: sign, raw, binned")
    
    # Validation arguments
    parser.add_argument('--predictions_per_fold', type=int,
                       help="Number of validation predictions per fold")
    parser.add_argument('--pos_edges_ratio', type=float,
                       help="Positive edge ratio for validation sampling")
    
    # Testing arguments
    parser.add_argument('--threshold_type', type=str,
                       help="Threshold type for testing (e.g., default_threshold, best_f1_threshold)")
    
    # Parse all arguments now
    args = parser.parse_args()
    
    # Reload config in case it changed
    config = load_config(args.config)
    
    # Handle "only" mode: if any -p, -tr, -v, -te, -a flags are used,
    # disable all steps first, then enable only the specified ones
    only_flags_used = any([args.only_preprocess, args.only_train, args.only_validate, 
                          args.only_test, args.only_analyze])
    
    if only_flags_used:
        # Disable all steps first
        args.preprocess = False
        args.train = False
        args.validate = False
        args.test = False
        args.analyze = False
        
        # Enable only the specified steps
        if args.only_preprocess:
            args.preprocess = True
        if args.only_train:
            args.train = True
        if args.only_validate:
            args.validate = True
        if args.only_test:
            args.test = True
        if args.only_analyze:
            args.analyze = True
    
    # Record start time for summary
    args._start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    experiment_start = time.time()
    
    # Print experiment configuration
    print(f"Starting experiment: {args.name}")
    print(f"Using config: {args.config or 'config.yaml (default)'}")
    print(f"Pipeline steps to run: ", end="")
    steps = []
    if args.preprocess: steps.append("preprocess")
    if args.train: steps.append("train")  
    if args.validate: steps.append("validate")
    if args.test: steps.append("test")
    if args.analyze: steps.append("analyze")
    print(" → ".join(steps))
    
    # Print key configuration parameters
    print(f"\nKey Configuration Parameters:")
    if 'min_train_embeddedness' in config:
        print(f"  - Min Train Embeddedness: {config['min_train_embeddedness']}")
    if 'min_test_embeddedness' in config:
        print(f"  - Min Test Embeddedness: {config['min_test_embeddedness']}")
    if 'use_weighted_features' in config:
        print(f"  - Use Weighted Features: {config['use_weighted_features']}")
    if 'cycle_length' in config:
        print(f"  - Cycle Length: {config['cycle_length']}")
    
    # Run pipeline steps
    results = []
    
    if args.preprocess:
        results.append(run_preprocess(args, config))
    else:
        results.append(True)  # Skipped steps count as successful
        
    if args.train:
        if not args.preprocess:
            print("\nNote: Training without preprocessing - ensure preprocess data exists")
        results.append(run_training(args, config))
    else:
        results.append(True)
        
    if args.validate:
        if not args.train and not args.preprocess:
            print("\nNote: Validation without training/preprocessing - ensure trained models exist")
        results.append(run_validation(args, config))
    else:
        results.append(True)
        
    if args.test:
        if not args.validate and not args.train and not args.preprocess:
            print("\nNote: Testing without previous steps - ensure all prerequisites exist")
        results.append(run_testing(args, config))
    else:
        results.append(True)
        
    if args.analyze:
        results.append(run_analysis(args, config))
    else:
        results.append(True)
    
    # Print summary
    total_time = time.time() - experiment_start
    print_experiment_summary(args, config, results, total_time)
    
    # Exit with error code if any step failed
    if not all(result for result, enabled in zip(results, [args.preprocess, args.train, args.validate, args.test, args.analyze]) if enabled):
        sys.exit(1)

if __name__ == "__main__":
    main()