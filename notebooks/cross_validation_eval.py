import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold

# Add src directory to Python path
sys.path.append(os.path.join('..'))

# Import custom modules
from src.data_loader import load_bitcoin_data
from src.preprocessing import filter_neutral_edges, map_to_unweighted_graph, ensure_connectivity, reindex_nodes_sequentially
from src.feature_extraction import feature_matrix_from_graph
from src.models import train_edge_sign_classifier, predict_edge_signs
from src.evaluation import evaluate_sign_predictor, plot_roc_curve, plot_confusion_matrix

def create_comparison_plot(results_k3, results_k4):
    """
    Create a comparison plot of metrics for k=3 and k=4
    """
    metrics = ['accuracy', 'false_positive_rate', 'precision', 'recall', 'f1_score']
    labels = ['Accuracy', 'False Positive Rate', 'Precision', 'Recall', 'F1 Score']
    
    # Use CV results if available, otherwise use simple results
    k3_values = []
    k4_values = []
    
    for metric in metrics:
        if f'avg_{metric}' in results_k3:
            k3_values.append(results_k3[f'avg_{metric}'])
            k4_values.append(results_k4[f'avg_{metric}'])
        else:
            k3_values.append(results_k3[metric])
            k4_values.append(results_k4[metric])
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, k3_values, width, label='k=3', alpha=0.8)
    bars2 = ax.bar(x + width/2, k4_values, width, label='k=4', alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title('Performance Comparison: k=3 vs k=4')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    results_dir = os.path.join('..', 'results')
    plt.savefig(os.path.join(results_dir, 'comparison_k3_vs_k4.png'), dpi=300)
    plt.close()

def create_fpr_comparison_plot(results_k3, results_k4):
    """
    Create a specific plot for false positive rate comparison
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Get FPR values
    if 'avg_false_positive_rate' in results_k3:
        fpr_k3 = results_k3['avg_false_positive_rate']
        fpr_k4 = results_k4['avg_false_positive_rate']
        std_k3 = results_k3.get('std_false_positive_rate', 0)
        std_k4 = results_k4.get('std_false_positive_rate', 0)
    else:
        fpr_k3 = results_k3['false_positive_rate']
        fpr_k4 = results_k4['false_positive_rate']
        std_k3 = 0
        std_k4 = 0
    
    colors = ['#3498db', '#e74c3c']
    bars = ax.bar(['k=3', 'k=4'], [fpr_k3, fpr_k4], color=colors, alpha=0.8)
    
    # Add error bars if available
    if std_k3 > 0 or std_k4 > 0:
        ax.errorbar(['k=3', 'k=4'], [fpr_k3, fpr_k4], yerr=[std_k3, std_k4], 
                    fmt='none', color='black', capsize=5)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    ax.set_ylabel('False Positive Rate')
    ax.set_title('False Positive Rate Comparison')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(fpr_k3, fpr_k4) * 1.2)
    
    plt.tight_layout()
    results_dir = os.path.join('..', 'results')
    plt.savefig(os.path.join(results_dir, 'fpr_comparison.png'), dpi=300)
    plt.close()

def create_summary_table(cv_results_k3, cv_results_k4):
    """
    Create a summary table for the report
    """
    metrics_display = {
        'avg_accuracy': 'Accuracy',
        'avg_auc': 'AUC',
        'avg_false_positive_rate': 'False Positive Rate',
        'avg_precision': 'Precision',
        'avg_recall': 'Recall',
        'avg_f1_score': 'F1 Score'
    }
    
    results_dir = os.path.join('..', 'results')
    
    # Create a table figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table
    table_data = []
    for metric_key, metric_name in metrics_display.items():
        if metric_key in cv_results_k3:
            k3_val = cv_results_k3[metric_key]
            k3_std = cv_results_k3.get(metric_key.replace('avg_', 'std_'), 0)
            k4_val = cv_results_k4[metric_key]
            k4_std = cv_results_k4.get(metric_key.replace('avg_', 'std_'), 0)
            
            improvement = ((k4_val - k3_val) / k3_val * 100) if metric_key != 'avg_false_positive_rate' else ((k3_val - k4_val) / k3_val * 100)
            
            table_data.append([
                metric_name,
                f'{k3_val:.4f} ± {k3_std:.4f}',
                f'{k4_val:.4f} ± {k4_std:.4f}',
                f'{improvement:+.2f}%'
            ])
    
    table = ax.table(cellText=table_data,
                     colLabels=['Metric', 'k=3', 'k=4', 'Improvement'],
                     cellLoc='center',
                     loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    plt.title('Cross-Validation Results Summary', fontsize=16, pad=20)
    plt.savefig(os.path.join(results_dir, 'results_summary_table.png'), dpi=300, bbox_inches='tight')
    plt.close()

def run_cross_validation(G, n_splits=10, cycle_length=3):
    """
    Run k-fold cross-validation for edge sign prediction
    """
    print(f"Running {n_splits}-fold cross-validation with cycle length k={cycle_length}...")
    
    # Extract features for all edges
    print("Extracting features for all edges...")
    X, y, edges = feature_matrix_from_graph(G, k=cycle_length)
    
    # Initialize k-fold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Store results from each fold
    fold_results = []
    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        print(f"\nProcessing fold {fold}/{n_splits}...")
        
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train model
        model = train_edge_sign_classifier(X_train, y_train)
        
        # Make predictions
        y_pred, y_prob = predict_edge_signs(model, X_test)
        
        # Store predictions for overall evaluation
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)
        
        # Evaluate
        metrics = evaluate_sign_predictor(y_test, y_pred, y_prob)
        fold_results.append(metrics)
        
        print(f"Fold {fold} - Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics.get('auc', 'N/A'):.4f}")
    
    # Calculate average metrics
    avg_results = {}
    for metric in fold_results[0].keys():
        if isinstance(fold_results[0][metric], (int, float)):
            values = [r[metric] for r in fold_results]
            avg_results[f'avg_{metric}'] = np.mean(values)
            avg_results[f'std_{metric}'] = np.std(values)
    
    # Create results directory
    results_dir = os.path.join('..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Convert to arrays
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_prob = np.array(all_y_prob)
    
    # Generate plots
    plot_roc_curve(all_y_true, all_y_prob, 
                   save_path=os.path.join(results_dir, f'roc_curve_k{cycle_length}.png'))
    plot_confusion_matrix(all_y_true, all_y_pred,
                         save_path=os.path.join(results_dir, f'confusion_matrix_k{cycle_length}.png'))
    
    return avg_results

def main():
    # Load data
    data_path = os.path.join('..', 'data', 'soc-sign-bitcoinotc.csv')
    G, df = load_bitcoin_data(data_path)
    
    # Preprocess
    print("Preprocessing graph...")
    G = filter_neutral_edges(G)
    G_signed = map_to_unweighted_graph(G)
    G_connected = ensure_connectivity(G_signed)
    G_processed = reindex_nodes_sequentially(G_connected)
    
    print(f"Final graph: {G_processed.number_of_nodes()} nodes, {G_processed.number_of_edges()} edges")
    
    # Run experiments for both k=3 and k=4
    print("\n" + "="*50)
    print("Running experiments for k=3...")
    cv_results_k3 = run_cross_validation(G_processed, n_splits=10, cycle_length=3)
    
    print("\n" + "="*50)
    print("Running experiments for k=4...")
    cv_results_k4 = run_cross_validation(G_processed, n_splits=10, cycle_length=4)
    
    # Create comparison visualizations
    print("\nCreating comparison plots...")
    create_comparison_plot(cv_results_k3, cv_results_k4)
    create_fpr_comparison_plot(cv_results_k3, cv_results_k4)
    create_summary_table(cv_results_k3, cv_results_k4)
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY FOR REPORT")
    print("="*50)
    
    print("\nCross-validation results (k=3):")
    for key, value in cv_results_k3.items():
        if key.startswith('avg_'):
            print(f"{key}: {value:.4f}")
    
    print("\nCross-validation results (k=4):")
    for key, value in cv_results_k4.items():
        if key.startswith('avg_'):
            print(f"{key}: {value:.4f}")
    
    print("\nKey improvements from k=3 to k=4:")
    acc_improvement = (cv_results_k4['avg_accuracy'] - cv_results_k3['avg_accuracy']) * 100
    auc_improvement = (cv_results_k4['avg_auc'] - cv_results_k3['avg_auc']) * 100
    fpr_reduction = (cv_results_k3['avg_false_positive_rate'] - cv_results_k4['avg_false_positive_rate']) * 100
    
    print(f"Accuracy: +{acc_improvement:.2f}%")
    print(f"AUC: +{auc_improvement:.2f}%")
    print(f"FPR Reduction: {fpr_reduction:.2f}%")
    
    print("\n✓ All experiments completed!")
    print("✓ Results saved in 'results' folder")
    print("✓ Ready for report writing")

if __name__ == "__main__":
    main()