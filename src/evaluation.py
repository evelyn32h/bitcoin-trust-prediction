import csv
import datetime
import json
import logging
import os
import pickle
import sys
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, precision_score, recall_score, f1_score
from sklearn.calibration import calibration_curve
from sklearn.model_selection import KFold
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
from collections import Counter
import time

# Set project root as working directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(PROJECT_ROOT)

from src.feature_extraction import feature_matrix_from_graph
from src.models import predict_edge_signs
from src.data_loader import load_bitcoin_data
from src.preprocessing import (map_to_unweighted_graph, ensure_connectivity, 
                             filter_neutral_edges, reindex_nodes)

# Configure matplotlib and seaborn for better plots
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
sns.set_palette("husl")

# Load config from YAML
with open(os.path.join(PROJECT_ROOT, 'config.yaml'), 'r') as f:
    config = yaml.safe_load(f)

DEFAULT_THRESHOLD = config['default_threshold']

def evaluate_sign_predictor(y_true, y_pred, y_prob=None):
    """
    Evaluate performance of edge sign prediction
    
    Parameters:
    y_true: True edge signs
    y_pred: Predicted edge signs
    y_prob: Predicted probabilities for positive class (optional)
    
    Returns:
    metrics: Dictionary containing evaluation metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate false positive rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Calculate F1 score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'accuracy': accuracy,
        'false_positive_rate': fpr,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positive': tp,
        'false_positive': fp,
        'true_negative': tn,
        'false_negative': fn
    }
    
    # Calculate AUC if probabilities are provided
    if y_prob is not None:
        y_prob = np.array(y_prob)
        # Convert labels to binary for AUC calculation
        y_binary = (y_true + 1) / 2  # Convert from {-1, 1} to {0, 1}
        auc_score = roc_auc_score(y_binary, y_prob)
        metrics['auc'] = auc_score
    
    return metrics


def calculate_best_f1_threshold(y_true, y_prob, thresholds=None):
    """
    Find the best threshold for converting probabilities to labels based on F1 score.
    Returns the best threshold, best F1 score, all thresholds, and all F1 scores.
    FIXED: Handle cases where labels might include 0 or other unexpected values
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    # Filter out any zero or unexpected labels to ensure binary classification
    valid_indices = np.isin(y_true, [-1, 1])
    
    if not np.any(valid_indices):
        print("Warning: No valid binary labels found for F1 threshold calculation")
        return 0.5, 0.0, [0.5], [0.0]
    
    # Filter to only valid binary labels
    y_true_filtered = y_true[valid_indices]
    y_prob_filtered = y_prob[valid_indices]
    
    if thresholds is None:
        thresholds = np.linspace(0, 1, 101)
    best_f1 = -np.inf
    best_threshold = 0.5
    f1_scores = {}
    for thresh in thresholds:
        y_pred = np.where(y_prob_filtered >= thresh, 1, -1)
        tp = np.sum((y_pred == 1) & (y_true_filtered == 1))
        fp = np.sum((y_pred == 1) & (y_true_filtered == -1))
        fn = np.sum((y_pred == -1) & (y_true_filtered == 1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores[thresh] = f1
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    return best_threshold, best_f1, f1_scores

def calculate_best_accuracy_threshold(y_true, y_prob, thresholds=None):
    """
    Find the best threshold for converting probabilities to labels based on accuracy.
    Returns the best threshold, best accuracy, all thresholds, and all accuracy scores.
    FIXED: Handle cases where labels might include 0 or other unexpected values
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    # Filter out any zero or unexpected labels to ensure binary classification
    valid_indices = np.isin(y_true, [-1, 1])
    
    if not np.any(valid_indices):
        print("Warning: No valid binary labels found for accuracy threshold calculation")
        return 0.5, 0.0, [0.5], [0.0]
    
    # Filter to only valid binary labels
    y_true_filtered = y_true[valid_indices]
    y_prob_filtered = y_prob[valid_indices]
    
    if thresholds is None:
        thresholds = np.linspace(0, 1, 101)
    best_acc = -np.inf
    best_threshold = 0.5
    acc_scores = {}
    for thresh in thresholds:
        y_pred = np.where(y_prob_filtered >= thresh, 1, -1)
        acc = (y_pred == y_true_filtered).mean()
        acc_scores[thresh] = acc
        if acc > best_acc:
            best_acc = acc
            best_threshold = thresh
    return best_threshold, best_acc, acc_scores

def calculate_evaluation_metrics(y_true, y_pred, y_prob, thresholds=None, default_threshold=None):
    """
    Calculate evaluation metrics for a range of thresholds and return best F1 and accuracy thresholds and their metrics.
    Returns a dictionary with best F1/accuracy, thresholds, and metrics at the default threshold.
    FIXED: Handle cases where labels might include 0 or other unexpected values
    """
    if default_threshold is None:
        default_threshold = DEFAULT_THRESHOLD
        
    if thresholds is None:
        thresholds = np.linspace(0, 1, 101)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    
    # Filter out any zero or unexpected labels to ensure binary classification
    valid_indices = np.isin(y_true, [-1, 1]) & np.isin(y_pred, [-1, 1])
    
    if not np.any(valid_indices):
        print("Warning: No valid binary labels found in evaluation metrics")
        return {
            'default_threshold': default_threshold,
            'best_f1_threshold': 0.5,
            'best_f1': 0.0,
            'best_accuracy_threshold': 0.5,
            'best_accuracy': 0.0,
            'roc_auc': 0.0,
            'average_precision': 0.0,
        }
    
    # Filter to only valid binary labels
    y_true_filtered = y_true[valid_indices]
    y_pred_filtered = y_pred[valid_indices]
    y_prob_filtered = y_prob[valid_indices]
    
    print(f"Evaluation metrics: using {len(y_true_filtered)} valid samples from {len(y_true)} total")
    
    # Check if we have both classes
    if len(np.unique(y_true_filtered)) != 2:
        print("Warning: Number of classes in evaluation data is not 2")
        return {
            'default_threshold': default_threshold,
            'best_f1_threshold': 0.5,
            'best_f1': 0.0,
            'best_accuracy_threshold': 0.5,
            'best_accuracy': 0.0,
            'roc_auc': 0.0,
            'average_precision': 0.0,
        }
    
    # find best threshold values
    best_f1_thresh, best_f1, f1_scores = calculate_best_f1_threshold(y_true_filtered, y_prob_filtered, thresholds)
    best_acc_thresh, best_acc, acc_scores = calculate_best_accuracy_threshold(y_true_filtered, y_prob_filtered, thresholds)
    
    # threshold independent metrics
    try:
        y_binary = (y_true_filtered + 1) / 2  # Convert from {-1, 1} to {0, 1}
        roc_auc = roc_auc_score(y_binary, y_prob_filtered)
        average_precision = average_precision_score(y_binary, y_prob_filtered)
    except ValueError as e:
        print(f"Warning: Could not calculate ROC AUC or AP in evaluation metrics: {e}")
        roc_auc = 0.0
        average_precision = 0.0
    
    return {
        'default_threshold': default_threshold,
        'default_f1': f1_scores.get(default_threshold, 0.0),
        'default_accuracy': acc_scores.get(default_threshold, 0.0),
        'best_f1_threshold': best_f1_thresh,
        'best_f1': best_f1,
        'best_accuracy_threshold': best_acc_thresh,
        'best_accuracy': best_acc,
        'roc_auc': roc_auc,
        'average_precision': average_precision,
    }
    

    
def calculate_test_metrics(y_true, y_pred, y_prob=None):
    """
    Calculate and return test metrics that are dependent on the threshold.
    Returns a dictionary with accuracy, confusion matrix, F1, precision, recall, specificity, and false positive rate.
    Uses sklearn metrics where possible.
    
    FIXED: Handle cases where labels might include 0 or other unexpected values
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Filter out any zero or unexpected labels to ensure binary classification
    valid_indices = np.isin(y_true, [-1, 1]) & np.isin(y_pred, [-1, 1])
    
    if not np.any(valid_indices):
        print("Warning: No valid binary labels found in test data")
        return {
            'accuracy': 0.0,
            'f1_score': 0.0, 
            'precision': 0.0,
            'recall': 0.0,
            'specificity': 0.0,
            'false_positive_rate': 0.0,
            'true_positive': 0,
            'false_positive': 0,
            'true_negative': 0,
            'false_negative': 0,
        }
    
    # Filter to only valid binary labels
    y_true_filtered = y_true[valid_indices]
    y_pred_filtered = y_pred[valid_indices]
    
    if y_prob is not None:
        y_prob = np.array(y_prob)
        y_prob_filtered = y_prob[valid_indices]
    else:
        y_prob_filtered = None
    
    print(f"Filtered test data: {len(y_true_filtered)} valid samples from {len(y_true)} total")
    
    # Calculate basic metrics
    acc = accuracy_score(y_true_filtered, y_pred_filtered)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=[-1, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Handle edge case where only one class is present
        tn = fp = fn = tp = 0
        if (y_true_filtered == 1).all():
            tp = (y_pred_filtered == 1).sum()
            fn = (y_pred_filtered == -1).sum()
        elif (y_true_filtered == -1).all():
            tn = (y_pred_filtered == -1).sum()
            fp = (y_pred_filtered == 1).sum()
    
    # Calculate precision, recall, F1 with explicit average parameter for binary classification
    precision = precision_score(y_true_filtered, y_pred_filtered, pos_label=1, average='binary', zero_division=0)
    recall = recall_score(y_true_filtered, y_pred_filtered, pos_label=1, average='binary', zero_division=0)
    f1 = f1_score(y_true_filtered, y_pred_filtered, pos_label=1, average='binary', zero_division=0)
    
    # Calculate specificity and false positive rate
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    metrics = {
        'accuracy': float(acc),
        'f1_score': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity),
        'false_positive_rate': float(fpr),
        'true_positive': int(tp),
        'false_positive': int(fp),
        'true_negative': int(tn),
        'false_negative': int(fn),
    }
    
    # Add ROC AUC and average precision if probabilities are provided
    if y_prob_filtered is not None and len(np.unique(y_true_filtered)) > 1:
        try:
            y_binary = (y_true_filtered + 1) / 2  # Convert from {-1, 1} to {0, 1}
            metrics['roc_auc'] = float(roc_auc_score(y_binary, y_prob_filtered))
            metrics['average_precision'] = float(average_precision_score(y_binary, y_prob_filtered))
        except ValueError as e:
            print(f"Warning: Could not calculate ROC AUC or AP: {e}")
            metrics['roc_auc'] = 0.0
            metrics['average_precision'] = 0.0
    
    return metrics
    
#### NETWORK ANALYSIS FUNCTIONS ####

def analyze_network(G, graph_name="Graph"):
    """
    Calculate and return comprehensive statistics of the network
    
    Parameters:
    G: NetworkX graph
    graph_name: Name for the graph being analyzed
    
    Returns:
    stats: Dictionary containing statistical information
    """
    print(f"\n{'='*60}")
    print(f"NETWORK ANALYSIS: {graph_name}")
    print(f"{'='*60}")
    
    stats = {}
    
    # Basic information
    stats['num_nodes'] = G.number_of_nodes()
    stats['num_edges'] = G.number_of_edges()
    stats['graph_density'] = nx.density(G)
    
    print(f"Nodes: {stats['num_nodes']:,}")
    print(f"Edges: {stats['num_edges']:,}")
    print(f"Density: {stats['graph_density']:.6f}")
    
    # Edge weight analysis
    if G.edges():
        weights = [data.get('weight', 1) for _, _, data in G.edges(data=True)]
        stats['positive_edges'] = sum(1 for w in weights if w > 0)
        stats['negative_edges'] = sum(1 for w in weights if w < 0)
        stats['zero_edges'] = sum(1 for w in weights if w == 0)
        stats['positive_ratio'] = stats['positive_edges'] / stats['num_edges']
        stats['negative_ratio'] = stats['negative_edges'] / stats['num_edges']
        
        stats['min_weight'] = min(weights)
        stats['max_weight'] = max(weights)
        stats['mean_weight'] = np.mean(weights)
        stats['std_weight'] = np.std(weights)
        
        print(f"\nEdge Weight Distribution:")
        print(f"  Positive edges: {stats['positive_edges']:,} ({stats['positive_ratio']:.2%})")
        print(f"  Negative edges: {stats['negative_edges']:,} ({stats['negative_ratio']:.2%})")
        if stats['zero_edges'] > 0:
            print(f"  Zero weight edges: {stats['zero_edges']:,}")
        print(f"  Weight range: [{stats['min_weight']}, {stats['max_weight']}]")
        print(f"  Mean weight: {stats['mean_weight']:.3f} ± {stats['std_weight']:.3f}")
    
    # Degree analysis
    if isinstance(G, nx.DiGraph):
        in_degrees = [d for n, d in G.in_degree()]
        out_degrees = [d for n, d in G.out_degree()]
        
        stats['avg_in_degree'] = np.mean(in_degrees)
        stats['max_in_degree'] = np.max(in_degrees)
        stats['avg_out_degree'] = np.mean(out_degrees)
        stats['max_out_degree'] = np.max(out_degrees)
        
        print(f"\nDegree Statistics (Directed):")
        print(f"  Average in-degree: {stats['avg_in_degree']:.2f}")
        print(f"  Maximum in-degree: {stats['max_in_degree']}")
        print(f"  Average out-degree: {stats['avg_out_degree']:.2f}")
        print(f"  Maximum out-degree: {stats['max_out_degree']}")
    else:
        degrees = [d for n, d in G.degree()]
        stats['avg_degree'] = np.mean(degrees)
        stats['max_degree'] = np.max(degrees)
        
        print(f"\nDegree Statistics (Undirected):")
        print(f"  Average degree: {stats['avg_degree']:.2f}")
        print(f"  Maximum degree: {stats['max_degree']}")
    
    # Connectivity analysis
    if isinstance(G, nx.DiGraph):
        stats['weakly_connected_components'] = nx.number_weakly_connected_components(G)
        stats['strongly_connected_components'] = nx.number_strongly_connected_components(G)
        
        if stats['weakly_connected_components'] > 0:
            largest_wcc = max(nx.weakly_connected_components(G), key=len)
            stats['largest_wcc_size'] = len(largest_wcc)
            stats['largest_wcc_ratio'] = stats['largest_wcc_size'] / stats['num_nodes']
        
        if stats['strongly_connected_components'] > 0:
            largest_scc = max(nx.strongly_connected_components(G), key=len)
            stats['largest_scc_size'] = len(largest_scc)
            stats['largest_scc_ratio'] = stats['largest_scc_size'] / stats['num_nodes']
        
        print(f"\nConnectivity (Directed):")
        print(f"  Weakly connected components: {stats['weakly_connected_components']}")
        print(f"  Largest WCC size: {stats.get('largest_wcc_size', 0):,} ({stats.get('largest_wcc_ratio', 0):.2%})")
        print(f"  Strongly connected components: {stats['strongly_connected_components']}")
        print(f"  Largest SCC size: {stats.get('largest_scc_size', 0):,} ({stats.get('largest_scc_ratio', 0):.2%})")
    else:
        stats['connected_components'] = nx.number_connected_components(G)
        if stats['connected_components'] > 0:
            largest_cc = max(nx.connected_components(G), key=len)
            stats['largest_cc_size'] = len(largest_cc)
            stats['largest_cc_ratio'] = stats['largest_cc_size'] / stats['num_nodes']
        
        print(f"\nConnectivity (Undirected):")
        print(f"  Connected components: {stats['connected_components']}")
        print(f"  Largest CC size: {stats.get('largest_cc_size', 0):,} ({stats.get('largest_cc_ratio', 0):.2%})")
    
    return stats

def calculate_embeddedness(G):
    """
    Calculate embeddedness (number of shared neighbors) for each edge in the graph
    
    Parameters:
    G: NetworkX graph
    
    Returns:
    edge_embeddedness: Mapping from edges to their embeddedness values
    """
    # Create undirected graph to calculate shared neighbors
    G_undirected = G.to_undirected()
    
    # Calculate embeddedness for each edge
    edge_embeddedness = {}
    
    for u, v in G.edges():
        # Get shared neighbors
        shared_neighbors = set(G_undirected.neighbors(u)) & set(G_undirected.neighbors(v))
        edge_embeddedness[(u, v)] = len(shared_neighbors)
    
    return edge_embeddedness

    
#### PLOTTING FUNCTIONS ####


def plot_roc_curve(y_true, y_prob, save_path=None, show=True):
    """
    Plot ROC curve for edge sign prediction
    FIXED: Handle cases where labels might include 0 or other unexpected values
    
    Parameters:
    y_true: True edge signs
    y_prob: Predicted probabilities for positive class
    save_path: Path to save the image (optional)
    
    Returns:
    auc_score: Area under the ROC curve
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    # Filter out any zero or unexpected labels to ensure binary classification
    valid_indices = np.isin(y_true, [-1, 1])
    
    if not np.any(valid_indices):
        print("Warning: No valid binary labels found for ROC curve")
        return 0.0
    
    # Filter to only valid binary labels
    y_true_filtered = y_true[valid_indices]
    y_prob_filtered = y_prob[valid_indices]
    
    print(f"ROC curve: using {len(y_true_filtered)} valid samples from {len(y_true)} total")
    
    # Convert labels to binary for ROC calculation
    y_binary = (y_true_filtered + 1) / 2  # Convert from {-1, 1} to {0, 1}
    
    # Check if we have both classes
    if len(np.unique(y_binary)) < 2:
        print("Warning: Only one class present, cannot compute ROC curve")
        return 0.0
    
    # Calculate ROC curve points
    fpr, tpr, _ = roc_curve(y_binary, y_prob_filtered)
    auc_score = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    
    if show:
        plt.show()
    
    return auc_score

def plot_precision_recall(y_true, y_prob, save_path=None, show=True):
    """
    Plot Precision-Recall curve for edge sign prediction
    FIXED: Handle cases where labels might include 0 or other unexpected values
    
    Parameters:
    y_true: True edge signs
    y_prob: Predicted probabilities for positive class
    save_path: Path to save the image (optional)
    show: Whether to display the plot (default: True)
    
    Returns:
    pr_auc: Area under the Precision-Recall curve
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    # Filter out any zero or unexpected labels to ensure binary classification
    valid_indices = np.isin(y_true, [-1, 1])
    
    if not np.any(valid_indices):
        print("Warning: No valid binary labels found for PR curve")
        return 0.0
    
    # Filter to only valid binary labels
    y_true_filtered = y_true[valid_indices]
    y_prob_filtered = y_prob[valid_indices]
    
    print(f"PR curve: using {len(y_true_filtered)} valid samples from {len(y_true)} total")
    
    # Convert labels to binary for PR calculation
    y_binary = (y_true_filtered + 1) / 2  # Convert from {-1, 1} to {0, 1}
    
    # Check if we have both classes
    if len(np.unique(y_binary)) < 2:
        print("Warning: Only one class present, cannot compute PR curve")
        return 0.0
    
    precision, recall, _ = precision_recall_curve(y_binary, y_prob_filtered)
    pr_auc = average_precision_score(y_binary, y_prob_filtered)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='purple', lw=2, label=f'PR curve (AP = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    
    if show:
        plt.show()
    
    return pr_auc

def analyze_false_positives(G, edges, y_true, y_pred):
    """
    Analyze characteristics of false positive predictions
    
    Parameters:
    G: NetworkX graph
    edges: List of edges
    y_true: True edge signs
    y_pred: Predicted edge signs
    
    Returns:
    analysis: Dictionary containing analysis of false positives
    """
    # Identify false positives (predicted positive but actually negative)
    fp_indices = [i for i, (true, pred) in enumerate(zip(y_true, y_pred)) 
                 if true == -1 and pred == 1]
    
    if not fp_indices:
        return {"num_false_positives": 0, "message": "No false positives found"}
    
    fp_edges = [edges[i] for i in fp_indices]
    
    # Calculate embeddedness for false positive edges
    G_undirected = G.to_undirected()
    
    fp_embeddedness = []
    for u, v in fp_edges:
        common_neighbors = set(G_undirected.neighbors(u)) & set(G_undirected.neighbors(v))
        fp_embeddedness.append(len(common_neighbors))
    
    # Calculate degree statistics for false positive edges
    fp_out_degree_u = [G.out_degree(u) for u, _ in fp_edges]
    fp_in_degree_u = [G.in_degree(u) for u, _ in fp_edges]
    fp_out_degree_v = [G.out_degree(v) for _, v in fp_edges]
    fp_in_degree_v = [G.in_degree(v) for _, v in fp_edges]
    
    # Prepare analysis results
    analysis = {
        'num_false_positives': len(fp_edges),
        'false_positive_rate': len(fp_edges) / len(edges),
        'avg_embeddedness': np.mean(fp_embeddedness),
        'median_embeddedness': np.median(fp_embeddedness),
        'min_embeddedness': min(fp_embeddedness),
        'max_embeddedness': max(fp_embeddedness),
        'avg_out_degree_u': np.mean(fp_out_degree_u),
        'avg_in_degree_u': np.mean(fp_in_degree_u),
        'avg_out_degree_v': np.mean(fp_out_degree_v),
        'avg_in_degree_v': np.mean(fp_in_degree_v)
    }
    
    # Generate visualization of embeddedness distribution
    plt.figure(figsize=(8, 6))
    plt.hist(fp_embeddedness, bins=10, alpha=0.7)
    plt.axvline(x=np.mean(fp_embeddedness), color='r', linestyle='--', 
                label=f'Mean: {np.mean(fp_embeddedness):.2f}')
    plt.xlabel('Embeddedness (Number of Common Neighbors)')
    plt.ylabel('Count')
    plt.title('Embeddedness Distribution of False Positive Edges')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    # plt.show()
    
    return analysis


def plot_confusion_matrix(y_true, y_pred, save_path=None, show=True):
    """
    Plot confusion matrix visualization
    FIXED: Handle cases where labels might include 0 or other unexpected values
    
    Parameters:
    y_true: True edge signs
    y_pred: Predicted edge signs
    save_path: Path to save the image (optional)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Filter out any zero or unexpected labels to ensure binary classification
    valid_indices = np.isin(y_true, [-1, 1]) & np.isin(y_pred, [-1, 1])
    
    if not np.any(valid_indices):
        print("Warning: No valid binary labels found for confusion matrix")
        return
    
    # Filter to only valid binary labels
    y_true_filtered = y_true[valid_indices]
    y_pred_filtered = y_pred[valid_indices]
    
    print(f"Confusion matrix: using {len(y_true_filtered)} valid samples from {len(y_true)} total")
    
    # Print class distribution in the evaluation set for verification
    unique, counts = np.unique(y_true, return_counts=True)
    class_dist = dict(zip(unique, counts))
    total = len(y_true)
    print("Evaluation set class distribution:")
    for cls, count in class_dist.items():
        print(f"  Class {cls}: {count} edges ({count/total*100:.1f}%)")
    
    # Calculate confusion matrix with filtered data
    cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=[-1, 1])
    
    # Handle different confusion matrix shapes
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    elif cm.shape == (1, 1) and len(np.unique(y_true_filtered)) == 1:
        # Only one class present
        if np.unique(y_true_filtered)[0] == 1:
            tp = cm.ravel()[0]
            fn = (y_pred_filtered == -1).sum()
            tn = fp = 0
        else:
            tn = cm.ravel()[0]
            fp = (y_pred_filtered == 1).sum()
            tp = fn = 0
    else:
        print(f"Unexpected confusion matrix shape: {cm.shape}")
        return
    
    # Calculate metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
    specificity = tn / (fp + tn) if (fp + tn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # Get positive and negative counts for debugging (using filtered data)
    pos_true = np.sum(y_true_filtered == 1)
    neg_true = np.sum(y_true_filtered == -1)
    pos_ratio = pos_true / len(y_true_filtered) if len(y_true_filtered) > 0 else 0
    
    # Print warning if distribution differs from expected
    if pos_ratio < 0.7:  # If less than 70% positive (expected 89%)
        print(f"\nWARNING: Evaluation set has {pos_ratio:.1%} positive edges")
        print(f"This differs from the expected 89% in the original dataset")
        print(f"Positive: {pos_true}, Negative: {neg_true}, Total: {len(y_true_filtered)}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Negative', 'Positive'],
               yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted Sign')
    plt.ylabel('True Sign')
    plt.title('Confusion Matrix')
    
    # Add dataset statistics and metrics for reference
    plt.figtext(0.5, 0.01, 
               f"Evaluation set: {pos_ratio:.1%} positive edges (Dataset: 89% positive)\n"
               f"Sensitivity (Recall): {sensitivity:.3f}    "
               f"Specificity: {specificity:.3f}    "
               f"False Positive Rate: {fpr:.3f}",
               ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2})
    
    if save_path:
        plt.savefig(save_path)
    
    if show:
        plt.show()

def plot_accuracy_vs_threshold(y_true, y_prob, thresholds=None, save_path=None, show=True):
    """
    Plot accuracy as a function of the decision threshold.
    FIXED: Handle cases where labels might include 0 or other unexpected values
    
    Parameters:
    y_true: True edge signs (numpy array or list)
    y_prob: Predicted probabilities for positive class (numpy array or list)
    thresholds: List or numpy array of thresholds to evaluate (default: np.linspace(0, 1, 101))
    save_path: Path to save the image (optional)
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    # Filter out any zero or unexpected labels to ensure binary classification
    valid_indices = np.isin(y_true, [-1, 1])
    
    if not np.any(valid_indices):
        print("Warning: No valid binary labels found for accuracy vs threshold plot")
        return
    
    # Filter to only valid binary labels
    y_true_filtered = y_true[valid_indices]
    y_prob_filtered = y_prob[valid_indices]
    
    print(f"Accuracy vs threshold: using {len(y_true_filtered)} valid samples from {len(y_true)} total")
    
    if thresholds is None:
        thresholds = np.linspace(0, 1, 101)
    accuracies = []
    
    for thresh in thresholds:
        # Convert probabilities to predicted labels using threshold
        y_pred = np.where(y_prob_filtered >= thresh, 1, -1)
        acc = (y_pred == y_true_filtered).mean()
        accuracies.append(acc)
    
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, accuracies, marker='o', lw=2)
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Threshold')
    plt.grid(alpha=0.3)
    plt.ylim(0, 1)
    # Mark the best threshold
    best_idx = np.argmax(accuracies)
    plt.axvline(x=thresholds[best_idx], color='r', linestyle='--', label=f'Best: {thresholds[best_idx]:.2f} (Acc={accuracies[best_idx]:.3f})')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    
    if show:
        plt.show()

def plot_f1_vs_threshold(y_true, y_prob, thresholds=None, save_path=None, show=True):
    """
    Plot F1 score as a function of the decision threshold.
    FIXED: Handle cases where labels might include 0 or other unexpected values
    
    Parameters:
    y_true: True edge signs (numpy array or list)
    y_prob: Predicted probabilities for positive class (numpy array or list)
    thresholds: List or numpy array of thresholds to evaluate (default: np.linspace(0, 1, 101))
    save_path: Path to save the image (optional)
    show: Whether to display the plot (default: True)
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    # Filter out any zero or unexpected labels to ensure binary classification
    valid_indices = np.isin(y_true, [-1, 1])
    
    if not np.any(valid_indices):
        print("Warning: No valid binary labels found for F1 vs threshold plot")
        return
    
    # Filter to only valid binary labels
    y_true_filtered = y_true[valid_indices]
    y_prob_filtered = y_prob[valid_indices]
    
    print(f"F1 vs threshold: using {len(y_true_filtered)} valid samples from {len(y_true)} total")
    
    if thresholds is None:
        thresholds = np.linspace(0, 1, 101)
    f1_scores = []
    for thresh in thresholds:
        y_pred = np.where(y_prob_filtered >= thresh, 1, -1)
        tp = np.sum((y_pred == 1) & (y_true_filtered == 1))
        fp = np.sum((y_pred == 1) & (y_true_filtered == -1))
        fn = np.sum((y_pred == -1) & (y_true_filtered == 1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, f1_scores, marker='o', lw=2)
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs. Threshold')
    plt.grid(alpha=0.3)
    plt.ylim(0, 1)
    best_idx = np.argmax(f1_scores)
    plt.axvline(x=thresholds[best_idx], color='r', linestyle='--', label=f'Best: {thresholds[best_idx]:.2f} (F1={f1_scores[best_idx]:.3f})')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    
    if show:
        plt.show()

def plot_calibration_curve(y_true, y_prob, n_bins=10, save_path=None, show=True):
    """
    Plot calibration curve (reliability diagram) for predicted probabilities.
    FIXED: Handle cases where labels might include 0 or other unexpected values
    
    Parameters:
    y_true: True edge signs (numpy array or list)
    y_prob: Predicted probabilities for positive class (numpy array or list)
    n_bins: Number of bins to use for calibration curve (default: 10)
    save_path: Path to save the image (optional)
    show: Whether to display the plot (default: True)
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    # Filter out any zero or unexpected labels to ensure binary classification
    valid_indices = np.isin(y_true, [-1, 1])
    
    if not np.any(valid_indices):
        print("Warning: No valid binary labels found for calibration curve")
        return
    
    # Filter to only valid binary labels
    y_true_filtered = y_true[valid_indices]
    y_prob_filtered = y_prob[valid_indices]
    
    print(f"Calibration curve: using {len(y_true_filtered)} valid samples from {len(y_true)} total")
    
    # Convert labels to binary for calibration
    y_binary = (y_true_filtered + 1) / 2  # {-1,1} -> {0,1}
    
    # Check if we have both classes
    if len(np.unique(y_binary)) < 2:
        print("Warning: Only one class present, cannot compute calibration curve")
        return
    
    prob_true, prob_pred = calibration_curve(y_binary, y_prob_filtered, n_bins=n_bins, strategy='uniform')
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve (Reliability Diagram)')
    plt.legend(loc='upper left')
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path)
    
    if show:
        plt.show()


def plot_feature_distributions_from_graph(G, save_path=None, k=4, show=False):
    """
    Extracts features from a NetworkX graph and plots the distribution of each feature in subplots.
    All plots are saved in a single image.
    Args:
        G: NetworkX graph.
        save_path: Path to save the image (optional).
        k: Maximum cycle length for feature extraction (default: 4).
        show: Whether to display the plot (default: False).
    """
    X, y, edge_list = feature_matrix_from_graph(G, k=k)
    n_features = X.shape[1]
    n_cols = min(4, n_features)
    n_rows = int(np.ceil(n_features / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    for i in range(n_features):
        ax = axes[i]
        ax.hist(X[:, i], bins=30, alpha=0.7, color='tab:blue')
        ax.set_title(f'Feature {i+1}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
    # Hide unused subplots
    for j in range(n_features, len(axes)):
        axes[j].axis('off')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    
    if show:
        plt.show()
    plt.close(fig)


def visualize_weight_distribution(G, save_path=None, graph_name="Graph"):
    """
    Visualize comprehensive distribution of edge weights
    
    Parameters:
    G: NetworkX graph
    save_path: Path to save the image (optional)
    graph_name: Name for the graph being visualized
    
    Returns:
    metrics: Dictionary containing calculated metrics
    """
    weights = [data.get('weight', 1) for _, _, data in G.edges(data=True)]
    
    if not weights:
        print("No edges found in graph")
        return
    
    # Create subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Edge Weight Analysis - {graph_name}', fontsize=16, fontweight='bold')
    
    # 1. Histogram of all weights
    ax1 = axes[0, 0]
    n, bins, patches = ax1.hist(weights, bins=30, alpha=0.7, edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero line')
    ax1.set_title('Weight Distribution')
    ax1.set_xlabel('Weight')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Color bars based on sign
    for i, p in enumerate(patches):
        if bins[i] < 0:
            p.set_facecolor('lightcoral')
        elif bins[i] > 0:
            p.set_facecolor('lightblue')
        else:
            p.set_facecolor('gray')
    
    # 2. Separate positive and negative weights
    ax2 = axes[0, 1]
    pos_weights = [w for w in weights if w > 0]
    neg_weights = [w for w in weights if w < 0]
    
    if pos_weights:
        ax2.hist(pos_weights, bins=20, alpha=0.7, label=f'Positive ({len(pos_weights)})', 
                color='lightblue', edgecolor='black')
    if neg_weights:
        ax2.hist(neg_weights, bins=20, alpha=0.7, label=f'Negative ({len(neg_weights)})', 
                color='lightcoral', edgecolor='black')
    
    ax2.set_title('Positive vs Negative Weights')
    ax2.set_xlabel('Weight')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Cumulative distribution
    ax3 = axes[1, 0]
    sorted_weights = np.sort(weights)
    cumulative = np.arange(1, len(sorted_weights) + 1) / len(sorted_weights)
    ax3.plot(sorted_weights, cumulative, linewidth=2, color='darkblue')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero line')
    ax3.set_title('Cumulative Distribution')
    ax3.set_xlabel('Weight')
    ax3.set_ylabel('Cumulative Probability')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Box plot
    ax4 = axes[1, 1]
    box_data = []
    labels = []
    
    if pos_weights:
        box_data.append(pos_weights)
        labels.append(f'Positive\n(n={len(pos_weights)})')
    if neg_weights:
        box_data.append(neg_weights)
        labels.append(f'Negative\n(n={len(neg_weights)})')
    
    if box_data:
        bp = ax4.boxplot(box_data, labels=labels, patch_artist=True)
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
            patch.set_facecolor(color)
    
    ax4.set_title('Weight Distribution Summary')
    ax4.set_ylabel('Weight')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    # Calculate comprehensive metrics
    metrics = {
        'total_edges': len(weights),
        'positive_edges': len(pos_weights),
        'negative_edges': len(neg_weights),
        'positive_ratio': len(pos_weights) / len(weights) if weights else 0,
        'negative_ratio': len(neg_weights) / len(weights) if weights else 0,
        'weight_range': [float(min(weights)), float(max(weights))] if weights else [0, 0],
        'weight_mean': float(np.mean(weights)) if weights else 0,
        'weight_median': float(np.median(weights)) if weights else 0,
        'weight_std': float(np.std(weights)) if weights else 0
    }
    
    if pos_weights:
        metrics['positive_weights_mean'] = float(np.mean(pos_weights))
        metrics['positive_weights_std'] = float(np.std(pos_weights))
    if neg_weights:
        metrics['negative_weights_mean'] = float(np.mean(neg_weights))
        metrics['negative_weights_std'] = float(np.std(neg_weights))
    
    # Print detailed statistics
    print(f"\nWeight Distribution Statistics for {graph_name}:")
    print(f"  Total edges: {len(weights):,}")
    print(f"  Positive edges: {len(pos_weights):,} ({len(pos_weights)/len(weights):.2%})")
    print(f"  Negative edges: {len(neg_weights):,} ({len(neg_weights)/len(weights):.2%})")
    print(f"  Weight range: [{min(weights):.3f}, {max(weights):.3f}]")
    print(f"  Mean: {np.mean(weights):.3f}, Median: {np.median(weights):.3f}")
    print(f"  Std dev: {np.std(weights):.3f}")
    
    if pos_weights:
        print(f"  Positive weights - Mean: {np.mean(pos_weights):.3f}, Std: {np.std(pos_weights):.3f}")
    if neg_weights:
        print(f"  Negative weights - Mean: {np.mean(neg_weights):.3f}, Std: {np.std(neg_weights):.3f}")
    
    return metrics
def remove_edges(X, y, edges, sampled_edges):
    """
    Remove specified edges from the feature matrix, label vector, and edge list.
    
    Parameters:
    X: Feature matrix (numpy array or similar structure)
    y: Label vector (numpy array or similar structure)
    edges: List of edges corresponding to rows in X
    sampled_edges: List of edges to be removed
    
    Returns:
    X_sup: Updated feature matrix with specified edges removed
    y_sup: Updated label vector with specified edges removed
    edges_sup: Updated list of edges with specified edges removed
    """
    indices_to_keep = [i for i, edge in enumerate(edges) if edge not in sampled_edges]
    X_sup = X[indices_to_keep]
    y_sup = y[indices_to_keep]
    edges_sup = [edge for i, edge in enumerate(edges) if edge not in sampled_edges]
    
    return X_sup, y_sup, edges_sup

def cml_evaluation(model, X, y, edges, removed_edges, threshold=0.5):
    """
    Predict and evaluate the sign of sampled edges.
    
    Parameters:
    model: Trained edge sign prediction model
    X: Feature matrix
    y: Label vector
    edges: List of all edges
    removed_edges: List of edges to predict
    threshold: Probability threshold for classifying edges as positive
    
    Returns:
    results: List of dictionaries containing prediction details for each sampled edge
    """
    #TODO: delete this function
    results = []
    for edge in removed_edges:
        edge_index = edges.index(edge)
        suppressed_edge_features = X[edge_index].reshape(1, -1)
        predictions, probabilities = predict_edge_signs(model, suppressed_edge_features, threshold=threshold)
        true_label = y[edge_index]  # Get the true label for the edge
        
        # Store prediction results
        results.append({
            "edge": edge,
            "true_label": true_label,
            "predicted_sign": predictions[0],
            "probability_positive": probabilities[0]
        })
        
        # Print prediction results
        print(f"Edge: {edge}, True label: {true_label}, Predicted sign: {predictions[0]}, "
              f"Probability of positive sign: {probabilities[0]:.4f} (Threshold: {threshold})")
    
    return results



def cross_validate(X, y, model_func, n_splits=5, class_weight=None):
    """
    Perform k-fold cross-validation
    
    Parameters:
    X: Feature matrix
    y: Labels (edge signs)
    model_func: Function that trains the model
    n_splits: Number of folds
    class_weight: Optional class weights for handling imbalance
    
    Returns:
    cv_results: Dictionary with cross-validation results
    """
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    accuracies = []
    fprs = []
    aucs = []
    f1_scores = []
    
    fold = 1
    for train_idx, test_idx in kf.split(X):
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train model
        model = model_func(X_train, y_train, class_weight)
        
        # Make predictions
        y_pred, y_prob = predict_edge_signs(model, X_test)
        
        # Evaluate
        metrics = evaluate_sign_predictor(y_test, y_pred, y_prob)
        
        # Store results
        accuracies.append(metrics['accuracy'])
        fprs.append(metrics['false_positive_rate'])
        aucs.append(metrics.get('auc', 0))
        f1_scores.append(metrics['f1_score'])
        
        print(f"Fold {fold}/{n_splits} - Accuracy: {metrics['accuracy']:.4f}, FPR: {metrics['false_positive_rate']:.4f}, AUC: {metrics.get('auc', 0):.4f}")
        fold += 1
    
    # Calculate average metrics
    cv_results = {
        'accuracy': np.mean(accuracies),
        'accuracy_std': np.std(accuracies),
        'fpr': np.mean(fprs),
        'fpr_std': np.std(fprs),
        'auc': np.mean(aucs),
        'auc_std': np.std(aucs),
        'f1_score': np.mean(f1_scores),
        'f1_score_std': np.std(f1_scores)
    }
    
    return cv_results

def save_and_plot_results(all_y_true, all_y_pred, all_y_prob, evaluation_name, cycle_length):
    """
    Convert results to arrays, evaluate, save, and plot.
    """
    #TODO: delete this function
    # Convert to arrays
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_prob = np.array(all_y_prob)
    
    # Save results
    results_dir = os.path.join('..', 'results', f'{evaluation_name}_evaluation')
    os.makedirs(results_dir, exist_ok=True)

    # Generate plots
    plot_roc_curve(all_y_true, all_y_prob, 
                   save_path=os.path.join(results_dir, f'{evaluation_name}_roc_k{cycle_length}.png'))
    plot_confusion_matrix(all_y_true, all_y_pred,
                         save_path=os.path.join(results_dir, f'{evaluation_name}_cm_k{cycle_length}.png'))


def save_run_results(true_labels, predicted_labels, predicted_probabilities, run_name="inductive_run", config=None):
    """
    Save evaluation results (labels, probabilities) to a new timestamped folder in the results directory in human-readable format.
    Parameters:
        true_labels: List of true edge labels
        predicted_labels: List of predicted edge labels
        predicted_probabilities: List of predicted probabilities
        run_name: Prefix for the results directory name (default: "inductive_run")
        config: Optional configuration dictionary
    """
    
    results_dir = os.path.join('..', 'results', run_name, 'run_results')
    os.makedirs(results_dir, exist_ok=True)

    # Save results as CSV files (human-readable)
    with open(os.path.join(results_dir, 'true_labels.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['true_label'])
        for label in true_labels:
            writer.writerow([label])
    with open(os.path.join(results_dir, 'predicted_labels.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['predicted_label'])
        for label in predicted_labels:
            writer.writerow([label])
    with open(os.path.join(results_dir, 'predicted_probabilities.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['predicted_probability'])
        for prob in predicted_probabilities:
            writer.writerow([prob])

    # Save configuration as JSON if provided
    if config is not None:
        with open(os.path.join(results_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

    logging.info(f"Saved all results to {results_dir} (CSV/JSON format)")

def load_run_results(results_dir):
    """
    Load evaluation results (labels, probabilities, config) from a results directory.
    Parameters:
        results_dir: Path to the results directory containing CSV/JSON files
    Returns:
        true_labels: Numpy array of true edge labels
        predicted_labels: Numpy array of predicted edge labels
        predicted_probabilities: Numpy array of predicted probabilities
        config: Configuration dictionary (or None if not found)
    """

    def load_csv_column(filepath):
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            return [float(row[0]) for row in reader]

    true_labels = np.array(load_csv_column(os.path.join(results_dir, 'true_labels.csv')))
    predicted_labels = np.array(load_csv_column(os.path.join(results_dir, 'predicted_labels.csv')))
    predicted_probabilities = np.array(load_csv_column(os.path.join(results_dir, 'predicted_probabilities.csv')), dtype=float)

    config_path = os.path.join(results_dir, 'config.json')
    config = None
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)

    return true_labels, predicted_labels, predicted_probabilities, config

def calculate_comparative_evaluation_metrics(y_true, y_pred, y_prob, thresholds=None, default_threshold=None, random_seed=42):
    """
    Calculate evaluation metrics comparing actual predictions against random and all-positive baselines.
    
    Parameters:
    y_true: True edge signs
    y_pred: Predicted edge signs
    y_prob: Predicted probabilities for positive class
    thresholds: Optional thresholds to evaluate
    default_threshold: Default threshold for evaluation
    random_seed: Random seed for reproducible random predictions
    
    Returns:
    Dictionary containing metrics for actual, random, and all-positive predictions
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    
    # 1. Calculate metrics with actual predictions (full optimization)
    actual_metrics = calculate_evaluation_metrics(y_true, y_pred, y_prob, thresholds, default_threshold)
    
    # 2. Generate random predictions and calculate simple metrics
    random_pred_prob = np.random.random(len(y_true))
    random_pred_binary = np.where(random_pred_prob >= 0.5, 1, -1)
    
    # Calculate simple metrics for random baseline without threshold optimization
    random_metrics = calculate_evaluation_metrics(y_true, random_pred_binary, random_pred_prob, thresholds, default_threshold)
    
    # 3. Generate all-positive predictions and calculate simple metrics
    all_positive_pred = np.ones(len(y_true), dtype=int)  # All predictions are positive (1)
    all_positive_prob = np.ones(len(y_true))  # All probabilities are 1.0
    
    all_positive_metrics = calculate_evaluation_metrics(y_true, all_positive_pred, all_positive_prob, thresholds, default_threshold)
    
    # Aggregate results
    comparative_results = {
        'actual': actual_metrics,
        'random_baseline': random_metrics,
        'all_positive_baseline': all_positive_metrics,
        'comparison': {
            'actual_vs_random_f1_improvement': actual_metrics['best_f1'] - random_metrics['best_f1'],
            'actual_vs_random_accuracy_improvement': actual_metrics['best_accuracy'] - random_metrics['best_accuracy'],
            'actual_vs_random_roc_auc_improvement': actual_metrics['roc_auc'] - random_metrics['roc_auc'],
            'actual_vs_all_positive_f1_improvement': actual_metrics['best_f1'] - all_positive_metrics['best_f1'],
            'actual_vs_all_positive_accuracy_improvement': actual_metrics['best_accuracy'] - all_positive_metrics['best_accuracy'],
            'actual_vs_all_positive_roc_auc_improvement': actual_metrics['roc_auc'] - all_positive_metrics['roc_auc'],
        }
    }
    
    return comparative_results

def calculate_comparative_test_metrics(y_true, y_pred, y_prob=None, random_seed=42):
    """
    Calculate test metrics comparing actual predictions against random and all-positive baselines.
    
    This function is similar to calculate_comparative_evaluation_metrics but uses calculate_test_metrics
    instead of calculate_evaluation_metrics, providing simpler metrics without threshold optimization.
    
    Parameters:
    y_true: True edge signs
    y_pred: Predicted edge signs
    y_prob: Predicted probabilities for positive class (optional)
    random_seed: Random seed for reproducible random predictions
    
    Returns:
    Dictionary containing metrics for actual, random, and all-positive predictions
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if y_prob is not None:
        y_prob = np.array(y_prob)
    
    # 1. Calculate metrics with actual predictions
    actual_metrics = calculate_test_metrics(y_true, y_pred, y_prob)
    
    # 2. Generate random predictions and calculate metrics
    random_pred_prob = np.random.random(len(y_true)) if y_prob is not None else None
    random_pred_binary = np.where(np.random.random(len(y_true)) >= 0.5, 1, -1)
    
    # Calculate metrics for random baseline
    random_metrics = calculate_test_metrics(y_true, random_pred_binary, random_pred_prob)
    
    # 3. Generate all-positive predictions and calculate metrics
    all_positive_pred = np.ones(len(y_true), dtype=int)  # All predictions are positive (1)
    all_positive_prob = np.ones(len(y_true)) if y_prob is not None else None  # All probabilities are 1.0
    
    all_positive_metrics = calculate_test_metrics(y_true, all_positive_pred, all_positive_prob)
    
    # Aggregate results
    comparative_results = {
        'actual': actual_metrics,
        'random_baseline': random_metrics,
        'all_positive_baseline': all_positive_metrics,
        'comparison': {
            'actual_vs_random_accuracy_improvement': actual_metrics['accuracy'] - random_metrics['accuracy'],
            'actual_vs_random_f1_improvement': actual_metrics['f1_score'] - random_metrics['f1_score'],
            'actual_vs_random_precision_improvement': actual_metrics['precision'] - random_metrics['precision'],
            'actual_vs_random_recall_improvement': actual_metrics['recall'] - random_metrics['recall'],
            'actual_vs_all_positive_accuracy_improvement': actual_metrics['accuracy'] - all_positive_metrics['accuracy'],
            'actual_vs_all_positive_f1_improvement': actual_metrics['f1_score'] - all_positive_metrics['f1_score'],
            'actual_vs_all_positive_precision_improvement': actual_metrics['precision'] - all_positive_metrics['precision'],
            'actual_vs_all_positive_recall_improvement': actual_metrics['recall'] - all_positive_metrics['recall'],
        }
    }
    
    # Add ROC AUC comparisons if probabilities were provided
    if y_prob is not None:
        comparative_results['comparison'].update({
            'actual_vs_random_roc_auc_improvement': actual_metrics.get('roc_auc', 0) - random_metrics.get('roc_auc', 0),
            'actual_vs_random_avg_precision_improvement': actual_metrics.get('average_precision', 0) - random_metrics.get('average_precision', 0),
            'actual_vs_all_positive_roc_auc_improvement': actual_metrics.get('roc_auc', 0) - all_positive_metrics.get('roc_auc', 0),
            'actual_vs_all_positive_avg_precision_improvement': actual_metrics.get('average_precision', 0) - all_positive_metrics.get('average_precision', 0),
        })
    
    return comparative_results

def visualize_degree_distribution(G, save_path=None, graph_name="Graph"):
    """
    Visualize degree distributions with comprehensive analysis
    
    Parameters:
    G: NetworkX graph
    save_path: Path to save the image (optional)
    graph_name: Name for the graph being visualized
    
    Returns:
    metrics: Dictionary containing calculated metrics
    """
    metrics = {}
    
    if isinstance(G, nx.DiGraph):
        # Directed graph
        in_degrees = [d for n, d in G.in_degree()]
        out_degrees = [d for n, d in G.out_degree()]
        total_degrees = [G.in_degree(n) + G.out_degree(n) for n in G.nodes()]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Degree Distribution Analysis - {graph_name}', fontsize=16, fontweight='bold')
        
        # In-degree distribution
        ax1 = axes[0, 0]
        ax1.hist(in_degrees, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
        ax1.set_title('In-Degree Distribution')
        ax1.set_xlabel('In-Degree')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Out-degree distribution
        ax2 = axes[0, 1]
        ax2.hist(out_degrees, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.set_title('Out-Degree Distribution')
        ax2.set_xlabel('Out-Degree')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # Total degree distribution
        ax3 = axes[1, 0]
        ax3.hist(total_degrees, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.set_title('Total Degree Distribution')
        ax3.set_xlabel('Total Degree (In + Out)')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        # Degree correlation
        ax4 = axes[1, 1]
        ax4.scatter(in_degrees, out_degrees, alpha=0.6, s=20)
        ax4.set_title('In-Degree vs Out-Degree')
        ax4.set_xlabel('In-Degree')
        ax4.set_ylabel('Out-Degree')
        ax4.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        correlation = np.corrcoef(in_degrees, out_degrees)[0, 1]
        ax4.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=ax4.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
        
        # Calculate metrics for directed graph
        metrics.update({
            'graph_type': 'directed',
            'in_degree_mean': float(np.mean(in_degrees)),
            'in_degree_std': float(np.std(in_degrees)),
            'in_degree_max': int(max(in_degrees)),
            'in_degree_min': int(min(in_degrees)),
            'out_degree_mean': float(np.mean(out_degrees)),
            'out_degree_std': float(np.std(out_degrees)),
            'out_degree_max': int(max(out_degrees)),
            'out_degree_min': int(min(out_degrees)),
            'total_degree_mean': float(np.mean(total_degrees)),
            'total_degree_std': float(np.std(total_degrees)),
            'total_degree_max': int(max(total_degrees)),
            'in_out_degree_correlation': float(correlation)
        })
        
        print(f"\nDegree Distribution Statistics for {graph_name} (Directed):")
        print(f"  In-degree - Mean: {np.mean(in_degrees):.2f}, Std: {np.std(in_degrees):.2f}, Max: {max(in_degrees)}")
        print(f"  Out-degree - Mean: {np.mean(out_degrees):.2f}, Std: {np.std(out_degrees):.2f}, Max: {max(out_degrees)}")
        print(f"  Total degree - Mean: {np.mean(total_degrees):.2f}, Std: {np.std(total_degrees):.2f}, Max: {max(total_degrees)}")
        print(f"  In-Out degree correlation: {correlation:.3f}")
        
    else:
        # Undirected graph
        degrees = [d for n, d in G.degree()]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Degree Distribution Analysis - {graph_name}', fontsize=16, fontweight='bold')
        
        # Degree distribution
        ax1 = axes[0]
        ax1.hist(degrees, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
        ax1.set_title('Degree Distribution')
        ax1.set_xlabel('Degree')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Log-log plot for power law analysis
        ax2 = axes[1]
        degree_counts = Counter(degrees)
        degrees_sorted = sorted(degree_counts.keys())
        counts = [degree_counts[d] for d in degrees_sorted]
        
        ax2.loglog(degrees_sorted, counts, 'bo-', alpha=0.7)
        ax2.set_title('Degree Distribution (Log-Log)')
        ax2.set_xlabel('Degree (log)')
        ax2.set_ylabel('Frequency (log)')
        ax2.grid(True, alpha=0.3)
        
        # Calculate metrics for undirected graph
        metrics.update({
            'graph_type': 'undirected',
            'degree_mean': float(np.mean(degrees)),
            'degree_std': float(np.std(degrees)),
            'degree_max': int(max(degrees)),
            'degree_min': int(min(degrees)),
            'degree_counts': {int(k): int(v) for k, v in degree_counts.items()}
        })
        
        print(f"\nDegree Distribution Statistics for {graph_name} (Undirected):")
        print(f"  Mean degree: {np.mean(degrees):.2f}")
        print(f"  Std degree: {np.std(degrees):.2f}")
        print(f"  Max degree: {max(degrees)}")
        print(f"  Min degree: {min(degrees)}")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return metrics
def visualize_embeddedness(G, save_path=None, graph_name="Graph"):
    """
    Visualize distribution of edge embeddedness with comprehensive analysis
    
    Parameters:
    G: NetworkX graph
    save_path: Path to save the image (optional)
    graph_name: Name for the graph being visualized
    
    Returns:
    edge_embeddedness: Dictionary mapping edges to embeddedness values
    metrics: Dictionary containing calculated metrics
    """
    print(f"\nCalculating embeddedness for {graph_name}...")
    start_time = time.time()
    
    edge_embeddedness = calculate_embeddedness(G)
    embeddedness_values = list(edge_embeddedness.values())
    
    calc_time = time.time() - start_time
    print(f"Embeddedness calculation completed in {calc_time:.2f} seconds")
    
    if not embeddedness_values:
        print("No edges found for embeddedness analysis")
        return
    
    # Create comprehensive subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Edge Embeddedness Analysis - {graph_name}', fontsize=16, fontweight='bold')
    
    # 1. Histogram of embeddedness values
    ax1 = axes[0, 0]
    n, bins, patches = ax1.hist(embeddedness_values, bins=min(30, max(embeddedness_values)+1), 
                               alpha=0.7, edgecolor='black', color='skyblue')
    ax1.set_title('Embeddedness Distribution')
    ax1.set_xlabel('Embeddedness (Number of Common Neighbors)')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # 2. Cumulative distribution
    ax2 = axes[0, 1]
    max_embed = max(embeddedness_values)
    cum_dist = []
    x_values = list(range(max_embed + 1))
    
    for i in x_values:
        cum_dist.append(sum(1 for x in embeddedness_values if x <= i) / len(embeddedness_values))
    
    ax2.plot(x_values, cum_dist, marker='o', linestyle='-', linewidth=2, markersize=4)
    ax2.set_title('Cumulative Distribution of Embeddedness')
    ax2.set_xlabel('Embeddedness Threshold')
    ax2.set_ylabel('Proportion of Edges')
    ax2.grid(True, alpha=0.3)
    
    # Add key percentiles
    percentiles = [0.5, 0.8, 0.9, 0.95]
    for p in percentiles:
        threshold = np.percentile(embeddedness_values, p * 100)
        ax2.axhline(y=p, color='red', linestyle='--', alpha=0.5)
        ax2.text(max_embed * 0.7, p + 0.02, f'{p:.0%}: {threshold:.1f}', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    # 3. Box plot and violin plot
    ax3 = axes[1, 0]
    bp = ax3.boxplot(embeddedness_values, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    ax3.set_title('Embeddedness Box Plot')
    ax3.set_ylabel('Embeddedness')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticklabels(['All Edges'])
    
    # 4. Embeddedness vs edge weight (if weights available)
    ax4 = axes[1, 1]
    weights = [data.get('weight', 1) for _, _, data in G.edges(data=True)]
    
    if len(set(weights)) > 1:  # If there are different weights
        # Create scatter plot
        edge_list = list(G.edges())
        edge_weights = [G[u][v].get('weight', 1) for u, v in edge_list]
        edge_embeds = [edge_embeddedness.get((u, v), 0) for u, v in edge_list]
        
        scatter = ax4.scatter(edge_embeds, edge_weights, alpha=0.6, s=20, c=edge_weights, 
                            cmap='RdYlBu', edgecolors='black', linewidth=0.5)
        ax4.set_title('Embeddedness vs Edge Weight')
        ax4.set_xlabel('Embeddedness')
        ax4.set_ylabel('Edge Weight')
        plt.colorbar(scatter, ax=ax4, label='Weight')
        
        # Calculate correlation
        correlation = np.corrcoef(edge_embeds, edge_weights)[0, 1]
        ax4.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=ax4.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    else:
        # Show embeddedness frequency for uniform weights
        embed_counts = Counter(embeddedness_values)
        embeds = sorted(embed_counts.keys())
        counts = [embed_counts[e] for e in embeds]
        
        ax4.bar(embeds, counts, alpha=0.7, color='lightgreen', edgecolor='black')
        ax4.set_title('Embeddedness Frequency')
        ax4.set_xlabel('Embeddedness')
        ax4.set_ylabel('Number of Edges')
    
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    # plt.show()
    
    # Calculate comprehensive metrics
    embed_counts = Counter(embeddedness_values)
    zero_embed = embed_counts.get(0, 0)
    
    metrics = {
        'total_edges_analyzed': len(embeddedness_values),
        'embeddedness_range': [int(min(embeddedness_values)), int(max(embeddedness_values))],
        'embeddedness_mean': float(np.mean(embeddedness_values)),
        'embeddedness_median': float(np.median(embeddedness_values)),
        'embeddedness_std': float(np.std(embeddedness_values)),
        'zero_embeddedness_count': int(zero_embed),
        'zero_embeddedness_percentage': float(zero_embed / len(embeddedness_values) * 100),
        'embeddedness_distribution': {int(k): int(v) for k, v in embed_counts.items()},
        'calculation_time_seconds': float(calc_time)
    }
    
    # Add percentile information
    for p in [25, 50, 75, 90, 95]:
        metrics[f'percentile_{p}'] = float(np.percentile(embeddedness_values, p))
    
    # Print detailed statistics
    print(f"\nEmbeddedness Statistics for {graph_name}:")
    print(f"  Total edges analyzed: {len(embeddedness_values):,}")
    print(f"  Embeddedness range: [{min(embeddedness_values)}, {max(embeddedness_values)}]")
    print(f"  Mean embeddedness: {np.mean(embeddedness_values):.3f}")
    print(f"  Median embeddedness: {np.median(embeddedness_values):.3f}")
    print(f"  Std deviation: {np.std(embeddedness_values):.3f}")
    
    # Show distribution of embeddedness levels
    print(f"\nEmbeddedness Level Distribution:")
    for embed in sorted(embed_counts.keys())[:10]:  # Show first 10 levels
        count = embed_counts[embed]
        percentage = count / len(embeddedness_values) * 100
        print(f"    {embed:2d} common neighbors: {count:5,} edges ({percentage:5.1f}%)")
    
    if len(embed_counts) > 10:
        remaining = sum(embed_counts[e] for e in sorted(embed_counts.keys())[10:])
        remaining_pct = remaining / len(embeddedness_values) * 100
        print(f"    >9 common neighbors: {remaining:5,} edges ({remaining_pct:5.1f}%)")
    
    # Calculate zero embeddedness
    zero_pct = zero_embed / len(embeddedness_values) * 100
    print(f"\nEdges with zero embeddedness: {zero_embed:,} ({zero_pct:.1f}%)")
    
    return edge_embeddedness, metrics
def analyze_temporal_patterns(G, df, save_path=None):
    """
    Analyze temporal patterns in the Bitcoin trust network
    
    Parameters:
    G: NetworkX graph
    df: Original dataframe with timestamp information
    save_path: Path to save the image (optional)
    
    Returns:
    metrics: Dictionary containing calculated metrics
    """
    print(f"\nAnalyzing temporal patterns...")
    
    # Convert timestamps to datetime
    df_temp = df.copy()
    df_temp['datetime'] = pd.to_datetime(df_temp['time'], unit='s')
    df_temp['year'] = df_temp['datetime'].dt.year
    df_temp['month'] = df_temp['datetime'].dt.to_period('M')
    df_temp['day'] = df_temp['datetime'].dt.date
    
    # Create subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Temporal Analysis of Bitcoin Trust Network', fontsize=16, fontweight='bold')
    
    # 1. Edges over time (daily)
    ax1 = axes[0, 0]
    daily_counts = df_temp.groupby('day').size()
    ax1.plot(daily_counts.index, daily_counts.values, alpha=0.7, linewidth=1)
    ax1.set_title('Daily Edge Creation')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Number of Edges')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 2. Monthly aggregation with sign analysis
    ax2 = axes[0, 1]
    df_temp['is_positive'] = df_temp['rating'] > 0
    monthly_pos = df_temp[df_temp['is_positive']].groupby('month').size()
    monthly_neg = df_temp[~df_temp['is_positive']].groupby('month').size()
    
    # Ensure both series have same index
    all_months = sorted(set(monthly_pos.index) | set(monthly_neg.index))
    monthly_pos = monthly_pos.reindex(all_months, fill_value=0)
    monthly_neg = monthly_neg.reindex(all_months, fill_value=0)
    
    ax2.bar(range(len(all_months)), monthly_pos.values, alpha=0.7, 
           label='Positive', color='lightblue')
    ax2.bar(range(len(all_months)), monthly_neg.values, alpha=0.7, 
           bottom=monthly_pos.values, label='Negative', color='lightcoral')
    ax2.set_title('Monthly Edge Creation by Sign')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Number of Edges')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Set x-axis labels (show every 6th month)
    tick_positions = range(0, len(all_months), max(1, len(all_months)//6))
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels([str(all_months[i]) for i in tick_positions], rotation=45)
    
    # 3. Weight distribution over time
    ax3 = axes[1, 0]
    yearly_weight_means = df_temp.groupby('year')['rating'].mean()
    yearly_weight_stds = df_temp.groupby('year')['rating'].std()
    
    ax3.errorbar(yearly_weight_means.index, yearly_weight_means.values, 
                yerr=yearly_weight_stds.values, marker='o', linestyle='-', 
                capsize=5, capthick=2)
    ax3.set_title('Average Edge Weight by Year')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Average Weight')
    ax3.grid(True, alpha=0.3)
    
    # 4. Cumulative network growth
    ax4 = axes[1, 1]
    df_temp_sorted = df_temp.sort_values('datetime')
    cumulative_edges = range(1, len(df_temp_sorted) + 1)
    
    # Sample data for visualization (every 100th point for large datasets)
    sample_indices = range(0, len(df_temp_sorted), max(1, len(df_temp_sorted)//1000))
    sample_dates = df_temp_sorted.iloc[sample_indices]['datetime']
    sample_cumulative = [cumulative_edges[i] for i in sample_indices]
    
    ax4.plot(sample_dates, sample_cumulative, linewidth=2, color='darkgreen')
    ax4.set_title('Cumulative Network Growth')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Cumulative Number of Edges')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    # plt.show()
    
    # Calculate comprehensive metrics
    time_span_days = (df_temp['datetime'].max() - df_temp['datetime'].min()).days
    daily_counts = df_temp.groupby('day').size()
    top_days = daily_counts.nlargest(5)
    
    metrics = {
        'date_range': {
            'start': str(df_temp['datetime'].min()),
            'end': str(df_temp['datetime'].max())
        },
        'time_span_days': int(time_span_days),
        'total_edges': len(df_temp),
        'average_edges_per_day': float(len(df_temp) / time_span_days) if time_span_days > 0 else 0,
        'daily_edge_statistics': {
            'mean': float(daily_counts.mean()),
            'std': float(daily_counts.std()),
            'min': int(daily_counts.min()),
            'max': int(daily_counts.max()),
            'median': float(daily_counts.median())
        },
        'most_active_days': {
            str(date): int(count) for date, count in top_days.items()
        },
        'yearly_statistics': {
            int(year): {
                'edge_count': int(count),
                'avg_weight': float(df_temp[df_temp['year'] == year]['rating'].mean()),
                'positive_edges': int(df_temp[(df_temp['year'] == year) & (df_temp['rating'] > 0)].shape[0]),
                'negative_edges': int(df_temp[(df_temp['year'] == year) & (df_temp['rating'] <= 0)].shape[0])
            }
            for year, count in df_temp.groupby('year').size().items()
        },
        'monthly_statistics': {
            'total_months': len(all_months),
            'avg_edges_per_month': float(len(df_temp) / len(all_months)) if all_months else 0,
            'positive_negative_ratio': float(monthly_pos.sum() / monthly_neg.sum()) if monthly_neg.sum() > 0 else float('inf')
        }
    }
    
    # Print temporal statistics
    print(f"\nTemporal Statistics:")
    print(f"  Date range: {df_temp['datetime'].min()} to {df_temp['datetime'].max()}")
    print(f"  Total time span: {time_span_days} days")
    print(f"  Average edges per day: {len(df_temp) / time_span_days:.2f}")
    
    # Show most active periods
    print(f"\nMost active days:")
    for date, count in top_days.items():
        print(f"    {date}: {count} edges")
    
    return metrics

def visualize_connectivity_analysis(graphs_dict, save_path=None):
    """
    Compare connectivity properties across different preprocessing steps
    
    Parameters:
    graphs_dict: Dictionary with {name: graph} pairs
    save_path: Path to save the image (optional)
    
    Returns:
    metrics: Dictionary containing calculated metrics
    """
    print(f"\nAnalyzing connectivity across preprocessing steps...")
    
    # Calculate connectivity metrics for each graph
    metrics = {}
    for name, G in graphs_dict.items():
        metrics[name] = {}
        
        if isinstance(G, nx.DiGraph):
            metrics[name]['weakly_connected'] = nx.number_weakly_connected_components(G)
            metrics[name]['strongly_connected'] = nx.number_strongly_connected_components(G)
            
            if metrics[name]['weakly_connected'] > 0:
                largest_wcc = max(nx.weakly_connected_components(G), key=len)
                metrics[name]['largest_wcc_size'] = len(largest_wcc)
                metrics[name]['largest_wcc_ratio'] = len(largest_wcc) / G.number_of_nodes()
            else:
                metrics[name]['largest_wcc_size'] = 0
                metrics[name]['largest_wcc_ratio'] = 0
                
            if metrics[name]['strongly_connected'] > 0:
                largest_scc = max(nx.strongly_connected_components(G), key=len)
                metrics[name]['largest_scc_size'] = len(largest_scc)
                metrics[name]['largest_scc_ratio'] = len(largest_scc) / G.number_of_nodes()
            else:
                metrics[name]['largest_scc_size'] = 0
                metrics[name]['largest_scc_ratio'] = 0
        else:
            metrics[name]['connected_components'] = nx.number_connected_components(G)
            if metrics[name]['connected_components'] > 0:
                largest_cc = max(nx.connected_components(G), key=len)
                metrics[name]['largest_cc_size'] = len(largest_cc)
                metrics[name]['largest_cc_ratio'] = len(largest_cc) / G.number_of_nodes()
            else:
                metrics[name]['largest_cc_size'] = 0
                metrics[name]['largest_cc_ratio'] = 0
        
        metrics[name]['nodes'] = G.number_of_nodes()
        metrics[name]['edges'] = G.number_of_edges()
        metrics[name]['density'] = nx.density(G)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Connectivity Analysis Across Preprocessing Steps', fontsize=16, fontweight='bold')
    
    graph_names = list(graphs_dict.keys())
    
    # 1. Nodes and edges comparison
    ax1 = axes[0, 0]
    nodes = [metrics[name]['nodes'] for name in graph_names]
    edges = [metrics[name]['edges'] for name in graph_names]
    
    x = np.arange(len(graph_names))
    width = 0.35
    
    ax1.bar(x - width/2, nodes, width, label='Nodes', alpha=0.8, color='lightblue')
    ax1.bar(x + width/2, edges, width, label='Edges', alpha=0.8, color='lightcoral')
    ax1.set_title('Nodes and Edges Comparison')
    ax1.set_xlabel('Preprocessing Step')
    ax1.set_ylabel('Count')
    ax1.set_xticks(x)
    ax1.set_xticklabels(graph_names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Density comparison
    ax2 = axes[0, 1]
    densities = [metrics[name]['density'] for name in graph_names]
    bars = ax2.bar(graph_names, densities, alpha=0.8, color='lightgreen')
    ax2.set_title('Graph Density Comparison')
    ax2.set_xlabel('Preprocessing Step')
    ax2.set_ylabel('Density')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, density in zip(bars, densities):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{density:.6f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Connected components
    ax3 = axes[1, 0]
    # Check if graphs are directed or undirected
    sample_graph = list(graphs_dict.values())[0]
    if isinstance(sample_graph, nx.DiGraph):
        wcc_counts = [metrics[name]['weakly_connected'] for name in graph_names]
        scc_counts = [metrics[name]['strongly_connected'] for name in graph_names]
        
        x = np.arange(len(graph_names))
        ax3.bar(x - width/2, wcc_counts, width, label='Weakly Connected', alpha=0.8, color='orange')
        ax3.bar(x + width/2, scc_counts, width, label='Strongly Connected', alpha=0.8, color='purple')
        ax3.set_title('Connected Components (Directed)')
    else:
        cc_counts = [metrics[name]['connected_components'] for name in graph_names]
        ax3.bar(graph_names, cc_counts, alpha=0.8, color='skyblue')
        ax3.set_title('Connected Components (Undirected)')
    
    ax3.set_xlabel('Preprocessing Step')
    ax3.set_ylabel('Number of Components')
    ax3.set_xticks(x)
    ax3.set_xticklabels(graph_names, rotation=45)
    if isinstance(sample_graph, nx.DiGraph):
        ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Largest component size ratio
    ax4 = axes[1, 1]
    if isinstance(sample_graph, nx.DiGraph):
        largest_wcc_ratios = [metrics[name]['largest_wcc_ratio'] for name in graph_names]
        largest_scc_ratios = [metrics[name]['largest_scc_ratio'] for name in graph_names]
        
        x = np.arange(len(graph_names))
        ax4.bar(x - width/2, largest_wcc_ratios, width, label='Largest WCC', alpha=0.8, color='orange')
        ax4.bar(x + width/2, largest_scc_ratios, width, label='Largest SCC', alpha=0.8, color='purple')
        ax4.set_title('Largest Component Size Ratio')
        ax4.legend()
    else:
        largest_cc_ratios = [metrics[name]['largest_cc_ratio'] for name in graph_names]
        ax4.bar(graph_names, largest_cc_ratios, alpha=0.8, color='skyblue')
        ax4.set_title('Largest Component Size Ratio')
    
    ax4.set_xlabel('Preprocessing Step')
    ax4.set_ylabel('Ratio of Nodes in Largest Component')
    ax4.set_xticks(x)
    ax4.set_xticklabels(graph_names, rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    # plt.show()
    
    # Print connectivity summary
    print(f"\nConnectivity Summary:")
    for name in graph_names:
        print(f"\n{name}:")
        print(f"  Nodes: {metrics[name]['nodes']:,}")
        print(f"  Edges: {metrics[name]['edges']:,}")
        print(f"  Density: {metrics[name]['density']:.6f}")
        if isinstance(sample_graph, nx.DiGraph):
            print(f"  Weakly connected components: {metrics[name]['weakly_connected']}")
            print(f"  Strongly connected components: {metrics[name]['strongly_connected']}")
            print(f"  Largest WCC ratio: {metrics[name]['largest_wcc_ratio']:.3f}")
            print(f"  Largest SCC ratio: {metrics[name]['largest_scc_ratio']:.3f}")
        else:
            print(f"  Connected components: {metrics[name]['connected_components']}")
            print(f"  Largest CC ratio: {metrics[name]['largest_cc_ratio']:.3f}")
    
    return metrics

def visualize_preprocessing_pipeline(save_path=None):
    """
    Visualize the complete preprocessing pipeline with before/after comparisons
    
    Parameters:
    save_path: Path to save the main summary image (optional)
    
    Returns:
    graphs: Dictionary of all preprocessing step graphs
    stats_summary: Dictionary of statistics for each step  
    """
    print(f"\n{'='*80}")
    print("COMPREHENSIVE PREPROCESSING PIPELINE ANALYSIS")
    print(f"{'='*80}")
    
    # Load original data
    data_path = os.path.join('..', 'data', 'soc-sign-bitcoinotc.csv')
    if not os.path.exists(data_path):
        # Try alternative path
        data_path = os.path.join('data', 'soc-sign-bitcoinotc.csv')
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return {}, {}
    
    G_original, df = load_bitcoin_data(data_path)
    
    # Store all preprocessing steps
    graphs = {
        '1. Original': G_original
    }
    
    # Step 2: Filter neutral edges
    print("\nStep 2: Filtering neutral edges...")
    G_filtered = filter_neutral_edges(G_original.copy())
    graphs['2. Filtered'] = G_filtered
    
    # Step 3: Map to signed graph
    print("Step 3: Converting to signed graph...")
    G_signed = map_to_unweighted_graph(G_filtered.copy())
    graphs['3. Signed'] = G_signed
    
    # Step 4: Ensure connectivity
    print("Step 4: Ensuring connectivity...")
    G_connected = ensure_connectivity(G_signed.copy())
    graphs['4. Connected'] = G_connected
    
    # Step 5: Reindex nodes
    print("Step 5: Reindexing nodes...")
    G_final = reindex_nodes(G_connected.copy())
    graphs['5. Final'] = G_final
    
    # Create results directory
    results_dir = os.path.join('..', 'results', 'preprocessing_analysis')
    os.makedirs(results_dir, exist_ok=True)
    
    # Analyze each step
    stats_summary = {}
    for step_name, graph in graphs.items():
        print(f"\n{'-'*60}")
        stats = analyze_network(graph, step_name)
        stats_summary[step_name] = stats
        
        # Create individual visualizations for key steps
        if step_name in ['1. Original', '3. Signed', '5. Final']:
            step_dir = os.path.join(results_dir, step_name.replace('. ', '_').lower())
            os.makedirs(step_dir, exist_ok=True)
            
            # Weight distribution
            visualize_weight_distribution(graph, 
                                        os.path.join(step_dir, 'weight_distribution.png'),
                                        step_name)
            
            # Degree distribution
            visualize_degree_distribution(graph, 
                                        os.path.join(step_dir, 'degree_distribution.png'),
                                        step_name)
            
            # Embeddedness (for final graph only to save time)
            if step_name == '5. Final':
                visualize_embeddedness(graph, 
                                     os.path.join(step_dir, 'embeddedness_distribution.png'),
                                     step_name)
    
    # Connectivity analysis across all steps
    connectivity_metrics = visualize_connectivity_analysis(graphs, 
                                  os.path.join(results_dir, 'connectivity_comparison.png'))
    
    # Temporal analysis on original data
    temporal_metrics = analyze_temporal_patterns(G_original, df, 
                            os.path.join(results_dir, 'temporal_analysis.png'))
    
    # Create summary comparison table
    create_preprocessing_summary_table(stats_summary, 
                                     os.path.join(results_dir, 'preprocessing_summary.png'))
    
    # Compile comprehensive metrics
    pipeline_metrics = {
        'preprocessing_steps': stats_summary,
        'connectivity_analysis': connectivity_metrics,
        'temporal_analysis': temporal_metrics,
        'pipeline_summary': {
            'original_nodes': stats_summary['1. Original']['num_nodes'],
            'original_edges': stats_summary['1. Original']['num_edges'],
            'final_nodes': stats_summary['5. Final']['num_nodes'],
            'final_edges': stats_summary['5. Final']['num_edges'],
            'node_retention_ratio': stats_summary['5. Final']['num_nodes'] / stats_summary['1. Original']['num_nodes'],
            'edge_retention_ratio': stats_summary['5. Final']['num_edges'] / stats_summary['1. Original']['num_edges'],
            'final_positive_ratio': stats_summary['5. Final'].get('positive_ratio', 0),
            'final_density': stats_summary['5. Final']['graph_density']
        }
    }
    
    print(f"\n{'='*80}")
    print("PREPROCESSING PIPELINE ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"All results saved to: {results_dir}")
    
    return graphs, stats_summary

def create_preprocessing_summary_table(stats_summary, save_path=None):
    """
    Create a summary table comparing all preprocessing steps
    
    Parameters:
    stats_summary: Dictionary of statistics for each preprocessing step
    save_path: Path to save the summary table image
    """
    # Prepare data for table
    steps = list(stats_summary.keys())
    
    # Select key metrics to display
    metrics = [
        ('Nodes', 'num_nodes'),
        ('Edges', 'num_edges'),
        ('Density', 'graph_density'),
        ('Positive Ratio', 'positive_ratio'),
        ('Avg In-Degree', 'avg_in_degree'),
        ('Avg Out-Degree', 'avg_out_degree'),
        ('WCC Count', 'weakly_connected_components'),
        ('Largest WCC %', 'largest_wcc_ratio')
    ]
    
    # Create table data
    table_data = []
    for metric_name, metric_key in metrics:
        row = [metric_name]
        for step in steps:
            value = stats_summary[step].get(metric_key, 'N/A')
            if isinstance(value, float):
                if metric_key in ['graph_density', 'positive_ratio', 'largest_wcc_ratio']:
                    row.append(f"{value:.3f}")
                else:
                    row.append(f"{value:.2f}")
            elif isinstance(value, int):
                row.append(f"{value:,}")
            else:
                row.append(str(value))
        table_data.append(row)
    
    # Create figure for table
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    headers = ['Metric'] + steps
    table = ax.table(cellText=table_data, colLabels=headers, 
                    cellLoc='center', loc='center',
                    colWidths=[0.15] + [0.17] * len(steps))
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Color header row
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color metric column
    for i in range(1, len(table_data) + 1):
        table[(i, 0)].set_facecolor('#f1f1f2')
        table[(i, 0)].set_text_props(weight='bold')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(1, len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f9f9f9')
    
    plt.title('Preprocessing Pipeline Summary', fontsize=16, fontweight='bold', pad=20)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Summary table saved to {save_path}")
    
    # plt.show()
