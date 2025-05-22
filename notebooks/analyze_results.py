import os
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

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

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
EXPERIMENT = 'experiment_v2'  # Change as needed
PREPROCESS_DIR = os.path.join(RESULTS_DIR, EXPERIMENT, 'preprocess')
TRAINING_DIR = os.path.join(RESULTS_DIR, EXPERIMENT, 'training')
VALIDATION_DIR = os.path.join(RESULTS_DIR, EXPERIMENT, 'validation')
TESTING_DIR = os.path.join(RESULTS_DIR, EXPERIMENT, 'testing')


def analyze_preprocess_output():
    print('\n--- Preprocess Output Analysis ---')
    test_path = os.path.join(PREPROCESS_DIR, 'test.csv')
    train_path = os.path.join(PREPROCESS_DIR, 'fold_0_train.csv')
    val_path = os.path.join(PREPROCESS_DIR, 'fold_0_val.csv')
    plots_dir = os.path.join(PREPROCESS_DIR, 'plots')
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
    # Plot feature distributions for test set
    if os.path.exists(test_path):
        G_test, _ = load_undirected_graph_from_csv(test_path)
        G_test = reindex_nodes(G_test)
        feature_plot_path = os.path.join(plots_dir, 'test_feature_distributions.png')
        plot_feature_distributions_from_graph(G_test, save_path=feature_plot_path, k=4, show=False)
    else:
        print(f"Test split not found at {test_path}")


def analyze_train_output():
    print('\n--- Training Output Analysis ---')
    plots_dir = os.path.join(TRAINING_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    if os.path.exists(TRAINING_DIR):
        print("No analysis implemented for training output yet.")
    else:
        print(f"Train split not found at {train_path}")


def analyze_validation_output():
    print('\n--- Validation Output Analysis ---')
    metrics_path = os.path.join(VALIDATION_DIR, 'metrics.csv')
    plots_dir = os.path.join(VALIDATION_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    if not os.path.exists(VALIDATION_DIR):
        print(f"No validation predictions found at {VALIDATION_DIR}")
        return
    y_true, y_pred, y_prob = load_prediction_results(VALIDATION_DIR)
    print(f"Validation set: {len(y_true)} samples")
    plot_confusion_matrix(y_true, y_pred, save_path=os.path.join(plots_dir, 'confusion_matrix.png'), show=False)
    plot_roc_curve(y_true, y_prob, save_path=os.path.join(plots_dir, 'roc_curve.png'), show=False)
    plot_precision_recall(y_true, y_prob, save_path=os.path.join(plots_dir, 'precision_recall_curve.png'), show=False)
    plot_accuracy_vs_threshold(y_true, y_prob, save_path=os.path.join(plots_dir, 'accuracy_vs_threshold.png'), show=False)
    plot_f1_vs_threshold(y_true, y_prob, save_path=os.path.join(plots_dir, 'f1_vs_threshold.png'), show=False)
    plot_calibration_curve(y_true, y_prob, save_path=os.path.join(plots_dir, 'calibration_curve.png'), show=False)
    if os.path.exists(metrics_path):
        print(f"Metrics from {metrics_path}:")
        print(pd.read_csv(metrics_path))


def analyze_test_output():
    print('\n--- Test Output Analysis ---')
    from src.data_loader import load_prediction_results
    metrics_path = os.path.join(TESTING_DIR, 'metrics.csv')
    plots_dir = os.path.join(TESTING_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    if not os.path.exists(TESTING_DIR):
        print(f"No test predictions found at {TESTING_DIR}")
        return
    y_true, y_pred, y_prob = load_prediction_results(TESTING_DIR)
    print(f"Test set: {len(y_true)} samples")
    plot_confusion_matrix(y_true, y_pred, save_path=os.path.join(plots_dir, 'confusion_matrix.png'), show=False)
    plot_roc_curve(y_true, y_prob, save_path=os.path.join(plots_dir, 'roc_curve.png'), show=False)
    plot_precision_recall(y_true, y_prob, save_path=os.path.join(plots_dir, 'precision_recall_curve.png'), show=False)
    plot_accuracy_vs_threshold(y_true, y_prob, save_path=os.path.join(plots_dir, 'accuracy_vs_threshold.png'), show=False)
    plot_f1_vs_threshold(y_true, y_prob, save_path=os.path.join(plots_dir, 'f1_vs_threshold.png'), show=False)
    plot_calibration_curve(y_true, y_prob, save_path=os.path.join(plots_dir, 'calibration_curve.png'), show=False)
    if os.path.exists(metrics_path):
        print(f"Metrics from {metrics_path}:")
        print(pd.read_csv(metrics_path))


def main():
    analyze_preprocess_output()
    analyze_train_output()
    analyze_validation_output()
    analyze_test_output()

if __name__ == '__main__':
    main()
