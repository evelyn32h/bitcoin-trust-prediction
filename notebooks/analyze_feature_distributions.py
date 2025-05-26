#!/usr/bin/env python3
"""
Feature Distribution Analysis for Bitcoin Trust Prediction
==========================================================

Requested by Vide to visualize feature distributions and calculate metrics.
This script loads preprocessed data and analyzes feature distributions
to justify preprocessing choices in the report.

Usage: python notebooks/analyze_feature_distributions.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

# Add project root to sys.path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import our modules
from src.data_loader import load_undirected_graph_from_csv
from src.preprocessing import reindex_nodes
from src.feature_extraction import feature_matrix_from_graph

def load_experiment_datasets(experiment_name):
    """
    Load training, validation, and test datasets from experiment preprocess results
    
    Parameters:
    experiment_name: Name of the experiment
    
    Returns:
    dict: Contains G_train, G_val, G_test graphs
    """
    base_path = Path("results") / experiment_name / "preprocess"
    
    if not base_path.exists():
        raise FileNotFoundError(f"Experiment {experiment_name} preprocess results not found")
    
    datasets = {}
    
    # Load test set
    test_path = base_path / "test.csv"
    if test_path.exists():
        G_test, _ = load_undirected_graph_from_csv(str(test_path))
        datasets['test'] = reindex_nodes(G_test)
        print(f"✅ Loaded test set: {datasets['test'].number_of_nodes()} nodes, {datasets['test'].number_of_edges()} edges")
    else:
        print("⚠️  Test set not found")
        datasets['test'] = None
    
    # Load fold 0 training and validation sets
    train_path = base_path / "fold_0_train.csv"
    val_path = base_path / "fold_0_val.csv"
    
    if train_path.exists():
        G_train, _ = load_undirected_graph_from_csv(str(train_path))
        datasets['train'] = reindex_nodes(G_train)
        print(f"✅ Loaded train set: {datasets['train'].number_of_nodes()} nodes, {datasets['train'].number_of_edges()} edges")
    else:
        print("⚠️  Train set not found")
        datasets['train'] = None
    
    if val_path.exists():
        G_val, _ = load_undirected_graph_from_csv(str(val_path))
        datasets['val'] = reindex_nodes(G_val)
        print(f"✅ Loaded validation set: {datasets['val'].number_of_nodes()} nodes, {datasets['val'].number_of_edges()} edges")
    else:
        print("⚠️  Validation set not found")
        datasets['val'] = None
    
    return datasets

def extract_features_from_datasets(datasets, cycle_length=4, sample_size=1000):
    """
    Extract features from all available datasets for analysis
    
    Parameters:
    datasets: Dictionary containing graphs
    cycle_length: k value for feature extraction
    sample_size: Number of edges to sample from each dataset
    
    Returns:
    dict: Feature matrices for each dataset
    """
    features = {}
    
    for dataset_name, G in datasets.items():
        if G is None:
            continue
            
        print(f"\n🔍 Extracting features from {dataset_name} dataset...")
        
        try:
            # Sample edges for feature extraction
            all_edges = list(G.edges(data=True))
            if len(all_edges) > sample_size:
                sampled_edges = np.random.choice(len(all_edges), sample_size, replace=False)
                edges_to_analyze = [all_edges[i] for i in sampled_edges]
            else:
                edges_to_analyze = all_edges
            
            print(f"   Analyzing {len(edges_to_analyze)} edges...")
            
            # Extract features
            X, y, feature_names = feature_matrix_from_graph(
                G, 
                edges=edges_to_analyze, 
                k=cycle_length,
                use_weighted_features=False  # Start with binary features
            )
            
            features[dataset_name] = {
                'X': X,
                'y': y,
                'feature_names': feature_names,
                'n_edges_analyzed': len(edges_to_analyze)
            }
            
            print(f"   ✅ Features extracted: {X.shape}")
            
        except Exception as e:
            print(f"   ❌ Error extracting features: {e}")
            features[dataset_name] = None
    
    return features

def calculate_feature_statistics(X, dataset_name):
    """
    Calculate comprehensive statistics for feature matrix
    
    Parameters:
    X: Feature matrix (numpy array)
    dataset_name: Name of the dataset
    
    Returns:
    dict: Statistical metrics
    """
    if X is None or X.size == 0:
        return None
    
    stats = {
        'dataset': dataset_name,
        'shape': X.shape,
        'mean': np.mean(X, axis=0).tolist(),
        'std': np.std(X, axis=0).tolist(),
        'min': np.min(X, axis=0).tolist(),
        'max': np.max(X, axis=0).tolist(),
        'median': np.median(X, axis=0).tolist(),
        'q25': np.percentile(X, 25, axis=0).tolist(),
        'q75': np.percentile(X, 75, axis=0).tolist(),
        'n_zeros': np.sum(X == 0, axis=0).tolist(),
        'n_positives': np.sum(X > 0, axis=0).tolist(),
        'n_negatives': np.sum(X < 0, axis=0).tolist(),
        'missing_values': np.sum(np.isnan(X), axis=0).tolist() if np.any(np.isnan(X)) else [0] * X.shape[1]
    }
    
    # Calculate skewness and kurtosis safely
    try:
        stats['skewness'] = [pd.Series(X[:, i]).skew() for i in range(X.shape[1])]
        stats['kurtosis'] = [pd.Series(X[:, i]).kurtosis() for i in range(X.shape[1])]
    except:
        stats['skewness'] = [0] * X.shape[1]
        stats['kurtosis'] = [0] * X.shape[1]
    
    return stats

def create_feature_distribution_plots(features_data, experiment_name, save_dir):
    """
    Create comprehensive feature distribution visualizations
    
    Parameters:
    features_data: Dictionary containing feature matrices for each dataset
    experiment_name: Name of the experiment
    save_dir: Directory to save plots
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    available_datasets = [(name, data) for name, data in features_data.items() 
                         if data is not None and data['X'] is not None]
    
    if not available_datasets:
        print("⚠️  No feature data available for visualization")
        return
    
    # 1. Overall feature distribution comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Feature Distribution Analysis - {experiment_name}', fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot 1: Overall distribution comparison
    for i, (dataset_name, data) in enumerate(available_datasets):
        if i >= len(colors):
            break
            
        X = data['X']
        color = colors[i]
        
        # Flatten all features for overall distribution
        X_flat = X.flatten()
        X_flat = X_flat[~np.isnan(X_flat)]  # Remove NaN values
        
        if len(X_flat) == 0:
            continue
        
        # Distribution plot
        axes[0, 0].hist(X_flat, bins=50, alpha=0.6, label=f'{dataset_name} ({len(X_flat)} values)', 
                       color=color, density=True)
        axes[0, 0].set_title('Overall Feature Value Distribution', fontweight='bold')
        axes[0, 0].set_xlabel('Feature Value')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Box plots by dataset
    box_data = []
    box_labels = []
    for dataset_name, data in available_datasets:
        X = data['X']
        X_flat = X.flatten()
        X_flat = X_flat[~np.isnan(X_flat)]
        if len(X_flat) > 0:
            box_data.append(X_flat)
            box_labels.append(dataset_name)
    
    if box_data:
        axes[0, 1].boxplot(box_data, labels=box_labels, patch_artist=True)
        axes[0, 1].set_title('Feature Value Ranges by Dataset', fontweight='bold')
        axes[0, 1].set_ylabel('Feature Value')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Feature-wise statistics (using first available dataset)
    if available_datasets:
        first_dataset_name, first_data = available_datasets[0]
        X = first_data['X']
        
        feature_means = np.mean(X, axis=0)
        feature_stds = np.std(X, axis=0)
        
        feature_indices = range(len(feature_means))
        axes[1, 0].scatter(feature_indices, feature_means, alpha=0.7, label='Mean', s=50, color='blue')
        axes[1, 0].scatter(feature_indices, feature_stds, alpha=0.7, label='Std Dev', s=50, color='red')
        axes[1, 0].set_title(f'Feature-wise Statistics ({first_dataset_name})', fontweight='bold')
        axes[1, 0].set_xlabel('Feature Index')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Class distribution
    if available_datasets:
        for i, (dataset_name, data) in enumerate(available_datasets):
            if i >= len(colors):
                break
            y = data['y']
            unique, counts = np.unique(y, return_counts=True)
            
            # Bar plot for this dataset
            x_pos = np.arange(len(unique)) + i * 0.8 / len(available_datasets)
            axes[1, 1].bar(x_pos, counts, width=0.8/len(available_datasets), 
                          label=dataset_name, alpha=0.7, color=colors[i])
        
        axes[1, 1].set_title('Class Distribution by Dataset', fontweight='bold')
        axes[1, 1].set_xlabel('Class (Edge Sign)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_xticks(np.arange(len(unique)))
        axes[1, 1].set_xticklabels(['+1' if x == 1 else '-1' for x in unique])
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{experiment_name}_feature_overview.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Individual feature analysis
    create_individual_feature_plots(features_data, experiment_name, save_dir)

def create_individual_feature_plots(features_data, experiment_name, save_dir):
    """
    Create detailed plots for individual features
    """
    available_datasets = [(name, data) for name, data in features_data.items() 
                         if data is not None and data['X'] is not None]
    
    if not available_datasets:
        return
    
    # Use the first dataset to determine number of features
    first_dataset_name, first_data = available_datasets[0]
    X = first_data['X']
    n_features = X.shape[1]
    
    if n_features == 0:
        return
    
    # Create subplot grid
    n_cols = min(4, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    fig.suptitle(f'Individual Feature Distributions - {experiment_name}', fontsize=14, fontweight='bold')
    
    # Handle single subplot case
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for feature_idx in range(n_features):
        row = feature_idx // n_cols
        col = feature_idx % n_cols
        
        ax = axes[row, col]
        
        for i, (dataset_name, data) in enumerate(available_datasets):
            if i >= len(colors):
                break
                
            X = data['X']
            feature_data = X[:, feature_idx]
            feature_data = feature_data[~np.isnan(feature_data)]  # Remove NaN
            
            if len(feature_data) > 0:
                ax.hist(feature_data, bins=20, alpha=0.6, label=dataset_name, 
                       color=colors[i], density=True)
        
        ax.set_title(f'Feature {feature_idx}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        if feature_idx == 0:  # Only show legend on first plot
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for feature_idx in range(n_features, n_rows * n_cols):
        row = feature_idx // n_cols
        col = feature_idx % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{experiment_name}_individual_features.png", dpi=300, bbox_inches='tight')
    plt.close()

def compare_scalers_impact(features_data, experiment_name, save_dir):
    """
    Compare different scaling methods and their impact on feature distributions
    """
    # Use training data for scaler comparison
    if 'train' not in features_data or features_data['train'] is None:
        print("⚠️  No training data available for scaler comparison")
        return
    
    X_train = features_data['train']['X']
    
    scalers = {
        'Original (No Scaling)': None,
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler()
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Impact of Different Scalers - {experiment_name}', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    scaler_results = {}
    
    for i, (scaler_name, scaler) in enumerate(scalers.items()):
        if scaler is None:
            X_scaled = X_train.copy()
        else:
            try:
                X_scaled = scaler.fit_transform(X_train)
            except Exception as e:
                print(f"⚠️  Could not apply {scaler_name}: {e}")
                continue
        
        # Plot distribution of scaled features
        X_flat = X_scaled.flatten()
        X_flat = X_flat[~np.isnan(X_flat)]
        
        if len(X_flat) == 0:
            continue
        
        axes[i].hist(X_flat, bins=50, alpha=0.7, edgecolor='black', color=f'C{i}')
        axes[i].set_title(f'{scaler_name}', fontweight='bold')
        axes[i].set_xlabel('Scaled Feature Value')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = np.mean(X_flat)
        std_val = np.std(X_flat)
        min_val = np.min(X_flat)
        max_val = np.max(X_flat)
        
        axes[i].axvline(mean_val, color='red', linestyle='--', alpha=0.8, linewidth=2)
        
        stats_text = f'μ={mean_val:.3f}\nσ={std_val:.3f}\nmin={min_val:.3f}\nmax={max_val:.3f}'
        axes[i].text(0.02, 0.98, stats_text, 
                    transform=axes[i].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
                    fontsize=10)
        
        # Store results for summary
        scaler_results[scaler_name] = {
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'range': max_val - min_val
        }
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{experiment_name}_scaler_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return scaler_results

def generate_feature_report(experiment_name, save_dir="plots"):
    """
    Generate comprehensive feature analysis report for an experiment
    """
    print(f"\n📊 Generating Feature Analysis Report for {experiment_name}")
    print("="*70)
    
    try:
        # Load datasets
        datasets = load_experiment_datasets(experiment_name)
        available_datasets = {k: v for k, v in datasets.items() if v is not None}
        
        if not available_datasets:
            print(f"❌ No valid datasets found for {experiment_name}")
            return None
        
        print(f"📋 Available datasets: {list(available_datasets.keys())}")
        
        # Extract features
        print(f"🔍 Extracting features for analysis...")
        features_data = extract_features_from_datasets(available_datasets, cycle_length=4, sample_size=1000)
        
        # Calculate statistics
        stats_report = {}
        for dataset_name, data in features_data.items():
            if data is not None:
                stats = calculate_feature_statistics(data['X'], dataset_name)
                stats_report[dataset_name] = stats
                print(f"✅ Calculated statistics for {dataset_name}")
        
        # Create visualizations
        print(f"📈 Creating feature distribution plots...")
        create_feature_distribution_plots(features_data, experiment_name, save_dir)
        
        print(f"⚖️  Comparing different scalers...")
        scaler_results = compare_scalers_impact(features_data, experiment_name, save_dir)
        
        # Save detailed statistics
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Save feature statistics
        with open(f"{save_dir}/{experiment_name}_feature_statistics.json", 'w') as f:
            json.dump(stats_report, f, indent=2)
        
        # Save scaler comparison results
        if scaler_results:
            with open(f"{save_dir}/{experiment_name}_scaler_results.json", 'w') as f:
                json.dump(scaler_results, f, indent=2)
        
        print(f"✅ Feature analysis complete for {experiment_name}!")
        print(f"📁 Results saved to {save_dir}/")
        
        return {
            'stats': stats_report,
            'scalers': scaler_results,
            'experiment': experiment_name
        }
        
    except Exception as e:
        print(f"❌ Error analyzing {experiment_name}: {e}")
        return None

def main():
    """
    Main function to run feature analysis on available experiments
    """
    print("🔍 Bitcoin Trust Prediction - Feature Distribution Analysis")
    print("="*70)
    print("Analyzing feature distributions as requested by Vide for the project report")
    print("Focus: Understanding feature characteristics and justifying preprocessing choices\n")
    
    # Check if results directory exists
    if not Path("results").exists():
        print("❌ Results directory not found. Please run experiments first.")
        return
    
    # Find available experiments
    results_dir = Path("results")
    experiments = [d.name for d in results_dir.iterdir() 
                  if d.is_dir() and (d / "preprocess").exists()]
    
    if not experiments:
        print("❌ No experiment results with preprocess data found")
        return
    
    print(f"🔍 Found {len(experiments)} experiments with preprocess data:")
    for exp in experiments:
        print(f"  - {exp}")
    print()
    
    # Analyze each experiment
    successful_analyses = 0
    all_results = {}
    
    # Create main plots directory
    plots_dir = "plots/feature_analysis"
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    
    for experiment in experiments:
        try:
            result = generate_feature_report(experiment, plots_dir)
            if result is not None:
                successful_analyses += 1
                all_results[experiment] = result
            print()
        except Exception as e:
            print(f"❌ Failed to analyze {experiment}: {e}\n")
    
    # Generate summary report
    if all_results:
        print("📝 Generating summary report...")
        generate_summary_report(all_results, plots_dir)
    
    print(f"🎯 Feature analysis complete!")
    print(f"✅ Successfully analyzed {successful_analyses}/{len(experiments)} experiments")
    print(f"📁 All results saved to {plots_dir}/")
    print("📋 Use these results to justify preprocessing choices in the report")
    print("🎨 Feature distribution plots ready for inclusion in report")

def generate_summary_report(all_results, save_dir):
    """
    Generate a summary report comparing all experiments
    """
    summary_path = Path(save_dir) / "feature_analysis_summary.md"
    
    with open(summary_path, 'w') as f:
        f.write("# Feature Distribution Analysis Summary\n\n")
        f.write("Generated for project report as requested by Vide\n\n")
        
        f.write("## Experiments Analyzed\n\n")
        for exp_name in all_results.keys():
            f.write(f"- {exp_name}\n")
        f.write("\n")
        
        f.write("## Key Findings\n\n")
        f.write("### Feature Distribution Characteristics\n")
        f.write("- Feature distributions show [to be filled based on visual inspection]\n")
        f.write("- Class imbalance observed: [to be filled]\n")
        f.write("- Scaling impact: [to be filled based on scaler comparison]\n\n")
        
        f.write("### Preprocessing Justification\n")
        f.write("Based on the feature analysis:\n")
        f.write("1. **Scaler Choice**: [Recommend based on distribution analysis]\n")
        f.write("2. **Feature Engineering**: [Observations about feature effectiveness]\n")
        f.write("3. **Data Quality**: [Notes about missing values, outliers]\n\n")
        
        f.write("## Files Generated\n\n")
        for exp_name in all_results.keys():
            f.write(f"### {exp_name}\n")
            f.write(f"- `{exp_name}_feature_overview.png`\n")
            f.write(f"- `{exp_name}_individual_features.png`\n")
            f.write(f"- `{exp_name}_scaler_comparison.png`\n")
            f.write(f"- `{exp_name}_feature_statistics.json`\n")
            f.write(f"- `{exp_name}_scaler_results.json`\n\n")
    
    print(f"📄 Summary report saved to {summary_path}")

if __name__ == "__main__":
    main()