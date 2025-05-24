import csv
import datetime
import json
import logging
import os
import pickle
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, precision_score, recall_score, f1_score
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set project root as working directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(PROJECT_ROOT)

from src.feature_extraction import feature_matrix_from_graph
from src.models import predict_edge_signs

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
    plt.show()
    
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
            tn = fp = fn = 0
        else:
            tn = cm.ravel()[0]
            tp = fp = fn = 0
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
    from sklearn.calibration import calibration_curve
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


#### EVERYTHING BELOW THIS LINE IS DEPRECATED ####



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
    from sklearn.model_selection import KFold
    
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
