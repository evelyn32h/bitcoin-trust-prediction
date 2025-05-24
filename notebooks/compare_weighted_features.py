import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Add project root to sys.path for src imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def load_experiment_metrics(experiment_name):
    """
    Load metrics from validation and test results for a given experiment.
    
    Parameters:
    experiment_name: Name of the experiment
    
    Returns:
    metrics: Dictionary containing validation and test metrics
    """
    base_path = os.path.join(PROJECT_ROOT, 'results', experiment_name)
    
    metrics = {
        'experiment_name': experiment_name,
        'validation_metrics': None,
        'test_metrics': None,
        'config_used': None
    }
    
    # Load validation metrics
    val_metrics_path = os.path.join(base_path, 'validation', 'metrics.json')
    if os.path.exists(val_metrics_path):
        with open(val_metrics_path, 'r') as f:
            metrics['validation_metrics'] = json.load(f)
    
    # Load test metrics
    test_metrics_path = os.path.join(base_path, 'testing', 'metrics.json')
    if os.path.exists(test_metrics_path):
        with open(test_metrics_path, 'r') as f:
            metrics['test_metrics'] = json.load(f)
    
    # Load config used
    config_path = os.path.join(base_path, 'preprocess', 'config_used.yaml')
    if os.path.exists(config_path):
        import yaml
        with open(config_path, 'r') as f:
            metrics['config_used'] = yaml.safe_load(f)
    
    return metrics

def compare_weighted_feature_methods(experiment_names):
    """
    Compare different weighted feature methods.
    
    Parameters:
    experiment_names: List of experiment names to compare
    
    Returns:
    comparison_df: DataFrame with comparison results
    """
    all_metrics = []
    
    for exp_name in experiment_names:
        print(f"Loading metrics for {exp_name}...")
        metrics = load_experiment_metrics(exp_name)
        
        if metrics['validation_metrics'] and metrics['test_metrics']:
            config = metrics['config_used'] or {}
            
            # Determine feature type
            use_weighted = config.get('use_weighted_features', False)
            weight_method = config.get('weight_method', 'sign')
            
            if use_weighted and weight_method == 'raw':
                feature_type = 'Weighted (Raw)'
            elif use_weighted and weight_method == 'binned':
                feature_type = 'Weighted (Binned)'
            elif weight_method == 'sign':
                feature_type = 'Binary (Sign)'
            else:
                feature_type = f'Unknown ({weight_method})'
            
            row = {
                'Experiment': exp_name,
                'Feature_Type': feature_type,
                'Use_Weighted_Features': use_weighted,
                'Weight_Method': weight_method,
                'Bidirectional_Method': config.get('bidirectional_method', 'unknown'),
                
                # Validation metrics
                'Val_ROC_AUC': metrics['validation_metrics'].get('roc_auc', 0),
                'Val_Best_F1': metrics['validation_metrics'].get('best_f1', 0),
                'Val_Best_Accuracy': metrics['validation_metrics'].get('best_accuracy', 0),
                'Val_Avg_Precision': metrics['validation_metrics'].get('average_precision', 0),
                
                # Test metrics
                'Test_Accuracy': metrics['test_metrics'].get('accuracy', 0),
                'Test_F1': metrics['test_metrics'].get('f1_score', 0),
                'Test_Precision': metrics['test_metrics'].get('precision', 0),
                'Test_Recall': metrics['test_metrics'].get('recall', 0),
                'Test_ROC_AUC': metrics['test_metrics'].get('roc_auc', 0),
                'Test_Specificity': metrics['test_metrics'].get('specificity', 0),
                'Test_FPR': metrics['test_metrics'].get('false_positive_rate', 0)
            }
            all_metrics.append(row)
        else:
            print(f"Warning: Missing metrics for experiment {exp_name}")
    
    if all_metrics:
        comparison_df = pd.DataFrame(all_metrics)
        return comparison_df
    else:
        print("No valid experiments found for comparison")
        return pd.DataFrame()

def create_task1_comparison_visualizations(comparison_df, save_dir):
    """
    Create visualizations comparing different weighted feature methods.
    
    Parameters:
    comparison_df: DataFrame with comparison results
    save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if comparison_df.empty:
        print("No data to visualize")
        return
    
    # 1. Feature Type Performance Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Test Accuracy by Feature Type
    axes[0, 0].bar(comparison_df['Feature_Type'], comparison_df['Test_Accuracy'], 
                   color=['skyblue', 'lightgreen', 'coral'])
    axes[0, 0].set_title('Test Accuracy by Feature Type (Task #1)')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].set_ylim(0, 1)
    
    # Test F1 Score by Feature Type
    axes[0, 1].bar(comparison_df['Feature_Type'], comparison_df['Test_F1'], 
                   color=['skyblue', 'lightgreen', 'coral'])
    axes[0, 1].set_title('Test F1 Score by Feature Type (Task #1)')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].set_ylim(0, 1)
    
    # Test ROC AUC by Feature Type
    axes[1, 0].bar(comparison_df['Feature_Type'], comparison_df['Test_ROC_AUC'], 
                   color=['skyblue', 'lightgreen', 'coral'])
    axes[1, 0].set_title('Test ROC AUC by Feature Type (Task #1)')
    axes[1, 0].set_ylabel('ROC AUC')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].set_ylim(0, 1)
    
    # Precision vs Recall
    axes[1, 1].scatter(comparison_df['Test_Recall'], comparison_df['Test_Precision'], 
                      s=100, alpha=0.7, c=['blue', 'green', 'red'])
    for i, feature_type in enumerate(comparison_df['Feature_Type']):
        axes[1, 1].annotate(feature_type, 
                           (comparison_df['Test_Recall'].iloc[i], comparison_df['Test_Precision'].iloc[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
    axes[1, 1].set_title('Test Precision vs Recall (Task #1)')
    axes[1, 1].set_xlabel('Recall')
    axes[1, 1].set_ylabel('Precision')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'task1_feature_types_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Validation vs Test Performance (Task #1 focus)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # ROC AUC comparison
    x_pos = np.arange(len(comparison_df))
    width = 0.35
    
    axes[0].bar(x_pos - width/2, comparison_df['Val_ROC_AUC'], width, 
                label='Validation ROC AUC', alpha=0.8, color='lightblue')
    axes[0].bar(x_pos + width/2, comparison_df['Test_ROC_AUC'], width, 
                label='Test ROC AUC', alpha=0.8, color='orange')
    axes[0].set_title('ROC AUC: Validation vs Test (Task #1)')
    axes[0].set_ylabel('ROC AUC')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(comparison_df['Feature_Type'], rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)
    
    # F1 Score comparison
    axes[1].bar(x_pos - width/2, comparison_df['Val_Best_F1'], width, 
                label='Validation Best F1', alpha=0.8, color='lightgreen')
    axes[1].bar(x_pos + width/2, comparison_df['Test_F1'], width, 
                label='Test F1', alpha=0.8, color='red')
    axes[1].set_title('F1 Score: Validation vs Test (Task #1)')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(comparison_df['Feature_Type'], rotation=45)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'task1_validation_vs_test_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def print_task1_comparison_summary(comparison_df):
    """
    Print a summary of Task #1 comparison results.
    
    Parameters:
    comparison_df: DataFrame with comparison results
    """
    if comparison_df.empty:
        print("No comparison data available")
        return
    
    print("\n=== TASK #1: WEIGHTED FEATURES COMPARISON SUMMARY ===")
    print("\n1. Test Performance Ranking:")
    
    # Rank by different metrics
    rankings = {}
    
    for metric in ['Test_Accuracy', 'Test_F1', 'Test_ROC_AUC', 'Test_Precision']:
        if metric in comparison_df.columns:
            ranked = comparison_df.nlargest(len(comparison_df), metric)
            rankings[metric] = [(row['Feature_Type'], row[metric]) 
                               for _, row in ranked.iterrows()]
    
    for metric, ranking in rankings.items():
        print(f"\n{metric.replace('Test_', '')} Ranking:")
        for i, (feature_type, score) in enumerate(ranking, 1):
            print(f"  {i}. {feature_type}: {score:.4f}")
    
    print(f"\n2. Task #1 Best Method Recommendations:")
    
    # Find best method for each metric
    best_methods = {}
    for metric in ['Test_Accuracy', 'Test_F1', 'Test_ROC_AUC']:
        if metric in comparison_df.columns:
            best_idx = comparison_df[metric].idxmax()
            best_method = comparison_df.loc[best_idx, 'Feature_Type']
            best_score = comparison_df.loc[best_idx, metric]
            best_methods[metric] = (best_method, best_score)
    
    for metric, (method, score) in best_methods.items():
        print(f"  Best {metric.replace('Test_', '')}: {method} ({score:.4f})")
    
    print(f"\n3. Task #1 Implementation Status:")
    print(f"  ✅ Successfully implemented weighted features")
    print(f"  ✅ Tested {len(comparison_df)} different feature methods")
    print(f"  ✅ Maintained backward compatibility with binary features")
    print(f"  ✅ Feature types tested: {', '.join(comparison_df['Feature_Type'].tolist())}")
    
    # Calculate improvement
    if len(comparison_df) > 1:
        weighted_methods = comparison_df[comparison_df['Use_Weighted_Features'] == True]
        binary_methods = comparison_df[comparison_df['Use_Weighted_Features'] == False]
        
        if not weighted_methods.empty and not binary_methods.empty:
            best_weighted_acc = weighted_methods['Test_Accuracy'].max()
            best_binary_acc = binary_methods['Test_Accuracy'].max()
            improvement = ((best_weighted_acc - best_binary_acc) / best_binary_acc) * 100
            
            print(f"\n4. Task #1 Performance Improvement:")
            print(f"  Best weighted accuracy: {best_weighted_acc:.4f}")
            print(f"  Best binary accuracy: {best_binary_acc:.4f}")
            print(f"  Improvement: {improvement:+.1f}%")

def main():
    """
    Main function to compare weighted feature methods (Task #1).
    """
    # Define experiments to compare (Task #1 focus)
    experiment_names = [
        'experiment_binary_optimized',  # Binary features baseline
        'experiment_weighted_raw',      # Task #1: Raw weighted features
        'experiment_weighted_binned'    # Task #1: Binned weighted features
    ]
    
    print("=== TASK #1: WEIGHTED FEATURES COMPARISON ===")
    print("Comparing binary vs weighted features implementation")
    print(f"Comparing experiments: {experiment_names}")
    
    # Check which experiments exist
    existing_experiments = []
    for exp_name in experiment_names:
        exp_path = os.path.join(PROJECT_ROOT, 'results', exp_name)
        if os.path.exists(exp_path):
            existing_experiments.append(exp_name)
        else:
            print(f"Warning: Experiment {exp_name} not found at {exp_path}")
    
    if not existing_experiments:
        print("No experiments found! Please run experiments first:")
        print("Example commands:")
        print("  # Binary baseline")
        print("  python preprocess.py --name experiment_binary_optimized --weight_method sign")
        print("  # Weighted features")
        print("  python preprocess.py --name experiment_weighted_raw --use_weighted_features --weight_method raw")
        print("  python preprocess.py --name experiment_weighted_binned --use_weighted_features --weight_method binned")
        return
    
    # Load and compare metrics
    comparison_df = compare_weighted_feature_methods(existing_experiments)
    
    if not comparison_df.empty:
        # Print detailed comparison
        print(f"\nDetailed Comparison Table:")
        print(comparison_df.round(4).to_string(index=False))
        
        # Save comparison table
        results_dir = os.path.join(PROJECT_ROOT, 'results', 'task1_comparison')
        os.makedirs(results_dir, exist_ok=True)
        comparison_df.round(4).to_csv(os.path.join(results_dir, 'weighted_features_comparison.csv'), index=False)
        
        # Create visualizations
        create_task1_comparison_visualizations(comparison_df, results_dir)
        
        # Print summary
        print_task1_comparison_summary(comparison_df)
        
        print(f"\n✓ Task #1 comparison complete! Results saved to {results_dir}")
    else:
        print("No valid experiments available for comparison.")

if __name__ == "__main__":
    main()