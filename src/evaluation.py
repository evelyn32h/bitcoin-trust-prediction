from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.models import predict_edge_signs

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
        # Convert labels to binary for AUC calculation
        y_binary = (y_true + 1) / 2  # Convert from {-1, 1} to {0, 1}
        auc_score = roc_auc_score(y_binary, y_prob)
        metrics['auc'] = auc_score
    
    return metrics

def plot_roc_curve(y_true, y_prob, save_path=None):
    """
    Plot ROC curve for edge sign prediction
    
    Parameters:
    y_true: True edge signs
    y_prob: Predicted probabilities for positive class
    save_path: Path to save the image (optional)
    
    Returns:
    auc_score: Area under the ROC curve
    """
    # Convert labels to binary for ROC calculation
    y_binary = (y_true + 1) / 2  # Convert from {-1, 1} to {0, 1}
    
    # Calculate ROC curve points
    fpr, tpr, _ = roc_curve(y_binary, y_prob)
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
        print(f"ROC curve saved to {save_path}")
    
    plt.show()
    
    return auc_score

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
    import networkx as nx
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

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot confusion matrix visualization
    
    Parameters:
    y_true: True edge signs
    y_pred: Predicted edge signs
    save_path: Path to save the image (optional)
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Negative', 'Positive'],
               yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted Sign')
    plt.ylabel('True Sign')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()

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
