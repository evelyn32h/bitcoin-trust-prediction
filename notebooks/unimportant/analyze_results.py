import os
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import argparse
import yaml
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.data_loader import load_prediction_results, load_undirected_graph_from_csv
from src.preprocessing import reindex_nodes
from src.evaluation import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall,
    plot_accuracy_vs_threshold,
    plot_f1_vs_threshold,
    plot_calibration_curve,
    plot_feature_distributions_from_graph
)

# Load config from YAML - Fixed to read from config instead of hardcoding
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
with open(os.path.join(PROJECT_ROOT, 'config.yaml'), 'r') as f:
    config = yaml.safe_load(f)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
DEFAULT_EXPERIMENT_NAME = config['default_experiment_name']

def analyze_preprocess_output(experiment_name):
    print('\n--- Preprocess Output Analysis ---')
    preprocess_dir = os.path.join(RESULTS_DIR, experiment_name, 'preprocess')
    test_path = os.path.join(preprocess_dir, 'test.csv')
    train_path = os.path.join(preprocess_dir, 'fold_0_train.csv')
    val_path = os.path.join(preprocess_dir, 'fold_0_val.csv')
    plots_dir = os.path.join(preprocess_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    for name, path in [('Test', test_path), ('Train (fold 0)', train_path), ('Val (fold 0)', val_path)]:
        if os.path.exists(path):
            G, df = load_undirected_graph_from_csv(path)
            print(f"{name}: {df['source'].nunique()} nodes, {len(df)} edges")
            print(f"  Class balance: {df['weight'].value_counts().to_dict()}")
            # Save class balance bar plot
            ax = df['weight'].value_counts().plot(kind='bar', title=f'{name} Class Balance')
            ax.set_xlabel('Weight')
            ax.set_ylabel('Count')
            fig = ax.get_figure()
            fig.tight_layout()
            fig.savefig(os.path.join(plots_dir, f'{name.lower().replace(" ", "_")}_class_balance.png'))
            plt.close(fig)
        else:
            print(f"{name}: File not found at {path}")
    
    # Plot feature distributions for test set
    if os.path.exists(test_path):
        G_test, _ = load_undirected_graph_from_csv(test_path)
        G_test = reindex_nodes(G_test)
        feature_plot_path = os.path.join(plots_dir, 'test_feature_distributions.png')
        plot_feature_distributions_from_graph(G_test, save_path=feature_plot_path, k=4, show=False)
    else:
        print(f"Test split not found at {test_path}")

def analyze_train_output(experiment_name):
    print('\n--- Training Output Analysis ---')
    training_dir = os.path.join(RESULTS_DIR, experiment_name, 'training')
    plots_dir = os.path.join(training_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    if os.path.exists(training_dir):
        print("No analysis implemented for training output yet.")
    else:
        print(f"Training directory not found at {training_dir}")

def analyze_validation_output(experiment_name):
    print('\n--- Validation Output Analysis ---')
    validation_dir = os.path.join(RESULTS_DIR, experiment_name, 'validation')
    metrics_path = os.path.join(validation_dir, 'metrics.json')  # Fixed: Changed to .json
    plots_dir = os.path.join(validation_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    if not os.path.exists(validation_dir):
        print(f"No validation directory found at {validation_dir}")
        return
    
    prediction_file = os.path.join(validation_dir, 'prediction_results.csv')
    if not os.path.exists(prediction_file):
        print(f"No validation predictions found at {prediction_file}")
        return
    
    y_true, y_pred, y_prob = load_prediction_results(validation_dir)
    print(f"Validation set: {len(y_true)} samples")
    plot_confusion_matrix(y_true, y_pred, save_path=os.path.join(plots_dir, 'confusion_matrix.png'), show=False)
    plot_roc_curve(y_true, y_prob, save_path=os.path.join(plots_dir, 'roc_curve.png'), show=False)
    plot_precision_recall(y_true, y_prob, save_path=os.path.join(plots_dir, 'precision_recall_curve.png'), show=False)
    plot_accuracy_vs_threshold(y_true, y_prob, save_path=os.path.join(plots_dir, 'accuracy_vs_threshold.png'), show=False)
    plot_f1_vs_threshold(y_true, y_prob, save_path=os.path.join(plots_dir, 'f1_vs_threshold.png'), show=False)
    plot_calibration_curve(y_true, y_prob, save_path=os.path.join(plots_dir, 'calibration_curve.png'), show=False)
    
    if os.path.exists(metrics_path):
        print(f"Metrics from {metrics_path}:")
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

def analyze_test_output(experiment_name):
    print('\n--- Test Output Analysis ---')
    testing_dir = os.path.join(RESULTS_DIR, experiment_name, 'testing')
    metrics_path = os.path.join(testing_dir, 'metrics.json')  # Fixed: Changed to .json
    plots_dir = os.path.join(testing_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    if not os.path.exists(testing_dir):
        print(f"No test directory found at {testing_dir}")
        return
    
    prediction_file = os.path.join(testing_dir, 'prediction_results.csv')
    if not os.path.exists(prediction_file):
        print(f"No test predictions found at {prediction_file}")
        return
    
    y_true, y_pred, y_prob = load_prediction_results(testing_dir)
    print(f"Test set: {len(y_true)} samples")
    plot_confusion_matrix(y_true, y_pred, save_path=os.path.join(plots_dir, 'confusion_matrix.png'), show=False)
    plot_roc_curve(y_true, y_prob, save_path=os.path.join(plots_dir, 'roc_curve.png'), show=False)
    plot_precision_recall(y_true, y_prob, save_path=os.path.join(plots_dir, 'precision_recall_curve.png'), show=False)
    plot_accuracy_vs_threshold(y_true, y_prob, save_path=os.path.join(plots_dir, 'accuracy_vs_threshold.png'), show=False)
    plot_f1_vs_threshold(y_true, y_prob, save_path=os.path.join(plots_dir, 'f1_vs_threshold.png'), show=False)
    plot_calibration_curve(y_true, y_prob, save_path=os.path.join(plots_dir, 'calibration_curve.png'), show=False)
    
    if os.path.exists(metrics_path):
        print(f"Metrics from {metrics_path}:")
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results and generate plots.")
    parser.add_argument('--name', type=str, default=DEFAULT_EXPERIMENT_NAME, 
                       help=f"Name of the experiment to analyze (default: {DEFAULT_EXPERIMENT_NAME})")
    args = parser.parse_args()
    
    experiment_name = args.name
    print(f"Analyzing experiment: {experiment_name}")
    
    analyze_preprocess_output(experiment_name)
    analyze_train_output(experiment_name)
    analyze_validation_output(experiment_name)
    analyze_test_output(experiment_name)

if __name__ == '__main__':
    main()