"""
Automated Epinions Experiment Runner
====================================

Automates the complete workflow:
1. Analyze Epinions dataset
2. Run baseline experiment on Bitcoin OTC
3. Run main experiment on Epinions
4. Compare results and generate analysis

This script ensures reproducible results for original paper dataset evaluation.
"""

import os
import sys
import subprocess
import time
import yaml
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def run_command(cmd, description, check_success=True):
    """
    Run a shell command and handle errors gracefully.
    
    Parameters:
    cmd: Command to run
    description: Description for logging
    check_success: Whether to stop on failure
    
    Returns:
    success: Boolean indicating if command succeeded
    """
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    print()
    
    try:
        # Run command from project root directory with proper encoding
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=True, text=True,
                              cwd=PROJECT_ROOT, encoding='utf-8', errors='replace')
        print(f"[SUCCESS] {description}")
        if result.stdout:
            print("Output:")
            print(result.stdout[-1000:])  # Last 1000 characters
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"[FAILED] {description}")
        print(f"Error code: {e.returncode}")
        if e.stdout:
            print("STDOUT:")
            print(e.stdout[-500:])
        if e.stderr:
            print("STDERR:")
            print(e.stderr[-500:])
        
        if check_success:
            print(f"\nStopping execution due to failure in: {description}")
            return False
        else:
            print(f"\nContinuing despite failure in: {description}")
            return False
    
    except Exception as e:
        print(f"[ERROR] {description}")
        print(f"Error: {str(e)}")
        if check_success:
            return False
        else:
            return False

def check_data_files():
    """
    Check if required data files exist.
    
    Returns:
    success: Boolean indicating if all files exist
    """
    print("="*60)
    print("CHECKING DATA FILES")
    print("="*60)
    
    bitcoin_path = os.path.join(PROJECT_ROOT, 'data', 'soc-sign-bitcoinotc.csv')
    epinions_path = os.path.join(PROJECT_ROOT, 'data', 'soc-sign-epinions.txt')
    
    bitcoin_exists = os.path.exists(bitcoin_path)
    epinions_exists = os.path.exists(epinions_path)
    
    print(f"Bitcoin OTC dataset: {'[OK]' if bitcoin_exists else '[MISSING]'} {bitcoin_path}")
    print(f"Epinions dataset: {'[OK]' if epinions_exists else '[MISSING]'} {epinions_path}")
    
    if not bitcoin_exists:
        print("\n[ERROR] Bitcoin OTC dataset missing!")
        print("Please download from: https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html")
        print(f"Save as: {bitcoin_path}")
    
    if not epinions_exists:
        print("\n[ERROR] Epinions dataset missing!")
        print("Please download from: https://snap.stanford.edu/data/soc-sign-epinions.html")
        print(f"Save as: {epinions_path}")
    
    return bitcoin_exists and epinions_exists

def update_config_for_dataset(dataset_type):
    """
    Update config.yaml for specific dataset.
    
    Parameters:
    dataset_type: 'bitcoin' or 'epinions'
    """
    config_path = os.path.join(PROJECT_ROOT, 'config.yaml')
    
    # Read current config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Update for dataset
    if dataset_type == 'bitcoin':
        config['data_path'] = 'data/soc-sign-bitcoinotc.csv'
        config['default_experiment_name'] = 'baseline_bitcoin'
        config['num_test_edges'] = 2000
        config['num_validation_edges'] = 1000
        config['n_test_predictions'] = 500
        config['nr_val_predictions'] = 300
    elif dataset_type == 'epinions':
        config['data_path'] = 'data/soc-sign-epinions.txt'
        config['default_experiment_name'] = 'experiment_epinions'
        config['num_test_edges'] = 5000
        config['num_validation_edges'] = 3000
        config['n_test_predictions'] = 1000
        config['nr_val_predictions'] = 500
    
    # Write updated config
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"[SUCCESS] Updated config.yaml for {dataset_type} dataset")

def run_full_pipeline(experiment_name, description):
    """
    Run the complete ML pipeline for an experiment.
    
    Parameters:
    experiment_name: Name of the experiment
    description: Description for logging
    
    Returns:
    success: Boolean indicating if pipeline succeeded
    """
    print(f"\n{'='*80}")
    print(f"RUNNING FULL PIPELINE: {description}")
    print(f"{'='*80}")
    
    # Step 1: Preprocessing
    if not run_command(f"python preprocess.py --name {experiment_name}", 
                      f"Preprocessing for {experiment_name}"):
        return False
    
    # Step 2: Training
    if not run_command(f"python train_model.py --name {experiment_name}", 
                      f"Training for {experiment_name}"):
        return False
    
    # Step 3: Validation
    if not run_command(f"python validate_model.py --name {experiment_name}", 
                      f"Validation for {experiment_name}"):
        return False
    
    # Step 4: Testing
    if not run_command(f"python test_model.py --name {experiment_name}", 
                      f"Testing for {experiment_name}"):
        return False
    
    print(f"[SUCCESS] PIPELINE COMPLETE: {description}")
    return True

def run_analysis_scripts():
    """
    Run the analysis and comparison scripts.
    
    Returns:
    success: Boolean indicating if analysis succeeded
    """
    print(f"\n{'='*80}")
    print("RUNNING ANALYSIS SCRIPTS")
    print(f"{'='*80}")
    
    # Run Epinions dataset analysis
    if not run_command("python notebooks/analyze_epinions_dataset.py", 
                      "Epinions dataset analysis", check_success=False):
        print("[WARNING] Epinions analysis failed, but continuing...")
    
    # Run dataset comparison
    if not run_command("python notebooks/compare_epinions_bitcoin.py", 
                      "Dataset comparison analysis"):
        return False
    
    print("[SUCCESS] ANALYSIS COMPLETE")
    return True

def cleanup_old_results():
    """
    Optional: Clean up old results if requested by user.
    """
    results_dir = os.path.join(PROJECT_ROOT, 'results')
    
    if os.path.exists(results_dir):
        print(f"\nFound existing results directory: {results_dir}")
        response = input("Do you want to clean up old results? (y/n): ").lower().strip()
        
        if response == 'y':
            import shutil
            try:
                shutil.rmtree(results_dir)
                print("[SUCCESS] Old results cleaned up")
            except Exception as e:
                print(f"[WARNING] Could not clean up results: {e}")
                print("Continuing with existing results...")

def estimate_runtime():
    """
    Provide runtime estimates for the experiments.
    """
    print("\n" + "="*60)
    print("RUNTIME ESTIMATES")
    print("="*60)
    print("Based on dataset sizes:")
    print("  Bitcoin OTC (~35K edges): ~10-15 minutes per pipeline")
    print("  Epinions (~841K edges): ~30-60 minutes per pipeline")
    print()
    print("Total estimated time: 1-2 hours")
    print("Recommendation: Run during free time, check progress periodically")
    print("="*60)

def main():
    """
    Main function to run the complete automated workflow.
    """
    start_time = time.time()
    
    print("="*80)
    print("AUTOMATED EPINIONS EXPERIMENT RUNNER")
    print("="*80)
    print("This script will:")
    print("1. Check data files")
    print("2. Analyze Epinions dataset")
    print("3. Run Bitcoin OTC baseline experiment")
    print("4. Run Epinions main experiment")
    print("5. Compare results and generate analysis")
    print()
    
    # Runtime estimates
    estimate_runtime()
    
    # Ask for confirmation
    response = input("\nDo you want to proceed? (y/n): ").lower().strip()
    if response != 'y':
        print("Execution cancelled by user.")
        return
    
    # Optional cleanup
    cleanup_old_results()
    
    # Step 1: Check data files
    if not check_data_files():
        print("\n[ERROR] DATA FILES MISSING")
        print("Please download the required datasets before running this script.")
        print("\nDownload instructions:")
        print("1. Bitcoin OTC: https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html")
        print("2. Epinions: https://snap.stanford.edu/data/soc-sign-epinions.html")
        print("3. Place files in data/ directory with exact names shown above")
        return
    
    print("\n[SUCCESS] All data files found!")
    
    # Step 2: Analyze Epinions dataset
    print(f"\n{'='*80}")
    print("STEP 1: ANALYZING EPINIONS DATASET")
    print(f"{'='*80}")
    
    if not run_command("python notebooks/analyze_epinions_dataset.py", 
                      "Epinions dataset analysis", check_success=False):
        print("[WARNING] Epinions analysis had issues, but continuing...")
    
    # Step 3: Run Bitcoin OTC baseline
    print(f"\n{'='*80}")
    print("STEP 2: RUNNING BITCOIN OTC BASELINE")
    print(f"{'='*80}")
    
    update_config_for_dataset('bitcoin')
    
    if not run_full_pipeline('baseline_bitcoin', 'Bitcoin OTC Baseline'):
        print("[ERROR] Bitcoin OTC baseline failed!")
        print("This experiment is required for comparison.")
        return
    
    # Step 4: Run Epinions main experiment
    print(f"\n{'='*80}")
    print("STEP 3: RUNNING EPINIONS MAIN EXPERIMENT")
    print(f"{'='*80}")
    
    update_config_for_dataset('epinions')
    
    if not run_full_pipeline('experiment_epinions', 'Epinions Main Experiment'):
        print("[ERROR] Epinions main experiment failed!")
        print("This is the core experiment for the original paper dataset.")
        return
    
    # Step 5: Optional weighted features experiment
    print(f"\n{'='*80}")
    print("STEP 4: OPTIONAL WEIGHTED FEATURES EXPERIMENT")
    print(f"{'='*80}")
    
    response = input("Run weighted features experiment on Epinions? (y/n): ").lower().strip()
    if response == 'y':
        if not run_command("python preprocess.py --name experiment_epinions_weighted --use_weighted_features --weight_method raw", 
                          "Preprocessing weighted features experiment"):
            print("[WARNING] Weighted preprocessing failed, skipping weighted experiment")
        else:
            # Continue with the rest of the weighted pipeline
            if not run_command("python train_model.py --name experiment_epinions_weighted", 
                              "Training weighted features experiment"):
                print("[WARNING] Weighted training failed, but continuing with main results")
            elif not run_command("python validate_model.py --name experiment_epinions_weighted", 
                                "Validation weighted features experiment"):
                print("[WARNING] Weighted validation failed, but continuing with main results")
            elif not run_command("python test_model.py --name experiment_epinions_weighted", 
                                "Testing weighted features experiment"):
                print("[WARNING] Weighted testing failed, but continuing with main results")
            else:
                print("[SUCCESS] Weighted features experiment completed")
    
    # Step 6: Run comparison analysis
    print(f"\n{'='*80}")
    print("STEP 5: GENERATING COMPARISON ANALYSIS")
    print(f"{'='*80}")
    
    if not run_analysis_scripts():
        print("[ERROR] Analysis scripts failed!")
        print("Results are available, but automated analysis incomplete.")
        return
    
    # Summary
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    
    print(f"\n{'='*80}")
    print("EXPERIMENT WORKFLOW COMPLETED!")
    print(f"{'='*80}")
    print(f"Total runtime: {hours}h {minutes}m")
    print()
    print("Results available in:")
    print("  - results/baseline_bitcoin/          (Bitcoin OTC baseline)")
    print("  - results/experiment_epinions/       (Epinions main experiment)")
    print("  - results/dataset_comparison/        (Comparison analysis)")
    print("  - results/epinions_analysis/         (Dataset analysis)")
    print()
    print("Key files to check:")
    print("  - results/dataset_comparison/dataset_comparison.csv")
    print("  - results/dataset_comparison/dataset_comparison.png")
    print("  - results/epinions_analysis/epinions_analysis_report.txt")
    print()
    print("Next steps:")
    print("1. Review comparison results in dataset_comparison.csv")
    print("2. Check visualizations in the results directories")
    print("3. Use findings to address teacher feedback")
    print("4. Demonstrate method effectiveness on original paper dataset")

if __name__ == "__main__":
    main()