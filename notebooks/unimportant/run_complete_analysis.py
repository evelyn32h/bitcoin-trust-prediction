#!/usr/bin/env python3
"""
Run Complete Analysis - Fixed Version
====================================

This script runs the fixed analysis to generate all required plots:

RESULT PLOTS (7 types):
1. Weighted vs Unweighted Comparison (MOST IMPORTANT)
2. Dataset Comparison (best configuration only)
3. Aggregation Methods Comparison
4. Complete Performance Summary Table
5. Positive Ratio Comparison
6. Cycle Length Comparison (3, 4, 5)
7. Pos/Neg Ratio Experiments

FIXES APPLIED:
- Embeddedness level fixed at 1 (no comparison)
- All experiments use optimal split (74:12:14)
- Enhanced experiment detection
- Improved analysis workflow

Usage: python run_complete_analysis.py
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Set project root as working directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(PROJECT_ROOT)

def run_script(script_path, description):
    """
    Run a Python script and handle errors gracefully
    """
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Script: {script_path}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Check if script exists
        if not os.path.exists(script_path):
            print(f"ERROR: Script not found: {script_path}")
            return False
        
        # Run the script
        result = subprocess.run([sys.executable, script_path], 
                              cwd=PROJECT_ROOT, 
                              check=True, 
                              capture_output=False)
        
        elapsed = time.time() - start_time
        print(f"\nSUCCESS: {description} completed successfully in {elapsed:.1f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\nFAILED: {description} failed after {elapsed:.1f} seconds with exit code {e.returncode}")
        return False
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\nERROR: {description} failed after {elapsed:.1f} seconds: {str(e)}")
        return False

def check_requirements():
    """
    Check if required directories and files exist
    """
    print("Checking requirements...")
    
    # Check for results directory
    results_paths = [
        os.path.join(PROJECT_ROOT, 'results'),
        '../results/',
        'results/'
    ]
    
    results_found = False
    experiment_count = 0
    
    for path in results_paths:
        if os.path.exists(path):
            experiments = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            if experiments:
                experiment_count = len(experiments)
                print(f"SUCCESS: Found results directory: {path}")
                print(f"   Available experiments: {experiments}")
                results_found = True
                break
    
    if not results_found:
        print("WARNING: No experiment results found")
        print("   Please run some experiments first using:")
        print("   python notebooks/run_experiment.py --name your_experiment")
        print("   OR run: python run_batch_experiments.py")
        return False, 0
    
    # Check for analysis script
    analysis_script = os.path.join(PROJECT_ROOT, 'notebooks/analyze_all_results.py')
    if os.path.exists(analysis_script):
        print(f"SUCCESS: Found analysis script: notebooks/analyze_all_results.py")
    else:
        print(f"ERROR: Missing analysis script: notebooks/analyze_all_results.py")
        return False, experiment_count
    
    return True, experiment_count

def create_output_directories():
    """
    Create output directories for plots
    """
    print("\nCreating output directories...")
    
    directories = [
        os.path.join(PROJECT_ROOT, 'plots', 'complete_results_analysis')
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"SUCCESS: Created directory: {directory}")

def check_experiment_variations():
    """
    Check what experiment variations are available
    """
    print("\nChecking experiment variations...")
    
    results_dir = os.path.join(PROJECT_ROOT, 'results')
    if not os.path.exists(results_dir):
        return False, []
    
    experiments = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    
    # Check for cycle length variations
    cycle_experiments = [exp for exp in experiments if 'cycle_length' in exp or 'cycle' in exp]
    
    # Check for positive ratio variations  
    ratio_experiments = [exp for exp in experiments if 'pos_ratio' in exp or 'pos_' in exp]
    
    # Check for weighted vs unweighted
    weighted_experiments = [exp for exp in experiments if 'weighted' in exp or 'unweighted' in exp]
    
    variations = {
        'cycle_length': cycle_experiments,
        'positive_ratio': ratio_experiments,
        'weighted_vs_unweighted': weighted_experiments
    }
    
    print(f"Cycle length experiments: {len(cycle_experiments)}")
    for exp in cycle_experiments:
        print(f"  {exp}")
    
    print(f"Positive ratio experiments: {len(ratio_experiments)}")
    for exp in ratio_experiments:
        print(f"  {exp}")
    
    print(f"Weighted vs unweighted experiments: {len(weighted_experiments)}")
    for exp in weighted_experiments:
        print(f"  {exp}")
    
    # Check if we have sufficient variations
    sufficient = len(cycle_experiments) >= 2 or len(ratio_experiments) >= 2 or len(weighted_experiments) >= 2
    
    return sufficient, variations

def verify_outputs():
    """
    Verify that expected output files were created
    """
    print("\nVerifying generated plots...")
    
    # Expected result plots
    result_plots_dir = os.path.join(PROJECT_ROOT, 'plots', 'complete_results_analysis')
    expected_result_plots = [
        '1_weighted_vs_unweighted_comparison.png',
        '2_dataset_comparison.png', 
        '3_aggregation_comparison.png',
        '4_performance_summary_table.png',
        '5_positive_ratio_comparison.png',
        '6_cycle_length_comparison.png',
        '7_pos_neg_ratio_comparison.png'
    ]
    
    result_count = 0
    
    print(f"\nRESULT PLOTS ({result_plots_dir}):")
    for plot in expected_result_plots:
        filepath = os.path.join(result_plots_dir, plot)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"SUCCESS: {plot}: {size:,} bytes")
            result_count += 1
        else:
            print(f"MISSING: {plot}: NOT FOUND")
    
    total_expected = len(expected_result_plots)
    total_found = result_count
    
    print(f"\nSUMMARY:")
    print(f"Result plots: {result_count}/{total_expected} generated")
    
    if total_found == total_expected:
        print("SUCCESS: ALL PLOTS GENERATED SUCCESSFULLY!")
    else:
        print("WARNING: Some plots were not generated. Check the analysis output above.")
    
    return total_found, total_expected

def main():
    """
    Main function to run complete analysis
    """
    print("RUNNING COMPLETE ANALYSIS (FIXED VERSION)")
    print("="*80)
    print("This script will generate all required plots with fixes:")
    print("• 7 Result plots addressing all workflow requirements")
    print("• Fixed embeddedness level at 1 (no comparison)")
    print("• All experiments use optimal split (74:12:14)")
    print("• Enhanced experiment detection")
    print("• High-resolution plots ready for presentation")
    print("="*80)
    
    overall_start = time.time()
    
    # Step 1: Check requirements
    requirements_ok, experiment_count = check_requirements()
    if not requirements_ok:
        print("\nERROR: Requirements check failed. Please address the issues above.")
        return False
    
    print(f"\nFound {experiment_count} experiments")
    
    # Step 2: Check experiment variations
    sufficient_variations, variations = check_experiment_variations()
    
    if not sufficient_variations:
        print("\nWARNING: INSUFFICIENT EXPERIMENT VARIATIONS")
        print("To generate missing experiments for complete comparison plots, run:")
        print("   python run_batch_experiments.py")
        print("\nThis will create:")
        print("   • Cycle length experiments (3, 4, 5)")
        print("   • Positive ratio experiments (90%, 80%, 70%, 60%, 50%)")
        print("   • Weighted vs unweighted experiments")
        print("   • All using fixed optimal split configuration")
        print("\nProceeding with available experiments...")
    else:
        print("\nSUCCESS: Sufficient experiment variations found for comparisons")
    
    # Step 3: Create output directories
    create_output_directories()
    
    # Step 4: Run fixed results analysis
    results_success = run_script(
        'notebooks/analyze_all_results.py',
        'Fixed Results Analysis (7 plots with improvements)'
    )
    
    # Step 5: Verify outputs
    plots_found, plots_expected = verify_outputs()
    
    # Final summary
    overall_elapsed = time.time() - overall_start
    
    print(f"\n{'='*80}")
    print("COMPLETE ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"Total runtime: {overall_elapsed:.1f} seconds ({overall_elapsed/60:.1f} minutes)")
    print(f"Results analysis: {'SUCCESS' if results_success else 'FAILED'}")
    print(f"Plots generated: {plots_found}/{plots_expected}")
    
    if results_success and plots_found == plots_expected:
        print(f"\nCOMPLETE SUCCESS!")
        print(f"All plots saved to:")
        print(f"   • ../plots/complete_results_analysis/ (result plots)")
        print(f"\nREADY FOR PRESENTATION!")
        print(f"   • All workflow requirements addressed")
        print(f"   • Fixed embeddedness level at 1")
        print(f"   • All experiments use optimal split (74:12:14)")
        print(f"   • Enhanced experiment detection")
        print(f"   • High-resolution academic-quality plots")
        
        # Provide next steps
        print(f"\nWHAT'S GENERATED:")
        print(f"   1. Weighted vs Unweighted Comparison (MOST IMPORTANT)")
        print(f"   2. Dataset Comparison (best configuration only)")
        print(f"   3. Aggregation Methods Comparison")
        print(f"   4. Performance Summary Table")
        print(f"   5. Positive Ratio Comparison")
        print(f"   6. Cycle Length Comparison (3, 4, 5)")
        print(f"   7. Pos/Neg Ratio Experiments")
        
        if not sufficient_variations:
            print(f"\nTO GET MORE COMPARISONS:")
            print(f"   Run: python run_batch_experiments.py")
            print(f"   Then re-run this analysis for complete plots")
        
        return True
    else:
        print(f"\nPARTIAL SUCCESS")
        print(f"   Some components failed or plots are missing")
        
        if not results_success:
            print(f"   ERROR: Results analysis failed")
        
        if plots_found < plots_expected:
            print(f"   ERROR: Only {plots_found}/{plots_expected} plots generated")
        
        print(f"\nTROUBLESHOOTING:")
        
        if not results_success:
            print(f"   • Check if analyze_all_results.py script exists")
            print(f"   • Verify experiment data format")
            print(f"   • Check for missing dependencies")
        
        if plots_found < plots_expected:
            print(f"   • Check experiment data completeness")
            print(f"   • Verify output directory permissions")
            print(f"   • Review analysis script output for errors")
        
        if not sufficient_variations:
            print(f"   • Run: python run_batch_experiments.py")
            print(f"   • This will generate missing experiment variations")
        
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)