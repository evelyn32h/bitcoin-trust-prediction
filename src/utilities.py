import random
import numpy as np
import networkx as nx
import logging
logger = logging.getLogger(__name__)

def print_feature_statistics(features):
    """
    Print statistics (min, max, mean, and count of zeros) for each feature across all edges.
    Parameters:
    features: dict mapping edge tuples to lists of feature values (one list per edge)
    """
    feature_matrix = np.array(list(features.values()))
    n_features = feature_matrix.shape[1]
    print(f"Feature statistics for {n_features} features:")
    for i in range(n_features):
        col = feature_matrix[:, i]
        min_val = np.min(col)
        max_val = np.max(col)
        mean_val = np.mean(col)
        zero_count = np.sum(col == 0)
        print(f"Feature {i+1}: min={min_val}, max={max_val}, mean={mean_val}, zeros={zero_count}/{len(col)}")

def sample_edges_with_positive_ratio(G, sample_size, pos_ratio=0.5):
    """
    Sample a subset of edges from a NetworkX graph with a specified ratio of positive to negative edges.
    
    Parameters:
        G (networkx.Graph): Input graph with edge attribute 'weight' indicating sign (+/-).
        sample_size (int): Total number of edges to sample.
        pos_ratio (float): Desired ratio of positive edges (0 < pos_ratio < 1).
    Returns:
        list: Sampled list of (u, v, data) edge tuples.
    """
    edge_list = list(G.edges(data=True))
    # Separate positive and negative edges
    pos_edges = [e for e in edge_list if e[2].get('weight', 1) > 0]
    neg_edges = [e for e in edge_list if e[2].get('weight', 1) < 0]
    
    n_pos = int(round(sample_size * pos_ratio))
    n_neg = sample_size - n_pos
    
    # If not enough edges, take as many as possible
    orig_n_pos, orig_n_neg = n_pos, n_neg
    n_pos = min(n_pos, len(pos_edges))
    n_neg = min(n_neg, len(neg_edges))
    if n_pos < orig_n_pos or n_neg < orig_n_neg:
        logger.warning(f"Requested {orig_n_pos} positive and {orig_n_neg} negative edges, but only {n_pos} positive and {n_neg} negative edges available.")
    
    random.seed(42)
    pos_sample = random.sample(pos_edges, n_pos) if n_pos > 0 else []
    neg_sample = random.sample(neg_edges, n_neg) if n_neg > 0 else []
    
    sample = pos_sample + neg_sample
    random.shuffle(sample)
    
    print(f"Sampled {len(sample)} edges: {len(pos_sample)} (out of {len(pos_edges)}) positive, {len(neg_sample)} (out of {len(neg_edges)}) negative")
    return sample

def print_comparative_evaluation_metrics(comparative_results):
    """
    Neatly print the results from calculate_comparative_evaluation_metrics.
    
    Parameters:
    comparative_results: Dictionary returned by calculate_comparative_evaluation_metrics
    """
    if not comparative_results:
        print("No comparative results to display")
        return
    
    print("\n" + "="*80)
    print("COMPARATIVE EVALUATION METRICS")
    print("="*80)
    
    # Extract main sections
    actual = comparative_results.get('actual', {})
    random = comparative_results.get('random_baseline', {})
    all_positive = comparative_results.get('all_positive_baseline', {})
    comparison = comparative_results.get('comparison', {})
    
    # 1. Actual Model Performance
    print("\n1. ACTUAL MODEL PERFORMANCE")
    print("-" * 40)
    if actual:
        print(f"   Best F1 Score:        {actual.get('best_f1', 0):.4f} (threshold: {actual.get('best_f1_threshold', 0):.3f})")
        print(f"   Best Accuracy:        {actual.get('best_accuracy', 0):.4f} (threshold: {actual.get('best_accuracy_threshold', 0):.3f})")
        print(f"   ROC AUC:              {actual.get('roc_auc', 0):.4f}")
        print(f"   Average Precision:    {actual.get('average_precision', 0):.4f}")
        if 'default_threshold' in actual:
            print(f"   Default Threshold:    {actual.get('default_threshold', 0.5):.3f}")
    else:
        print("   No actual metrics available")
    
    # 2. Random Baseline Performance
    print("\n2. RANDOM BASELINE PERFORMANCE")
    print("-" * 40)
    if random:
        print(f"   Best F1 Score:        {random.get('best_f1', 0):.4f} (threshold: {random.get('best_f1_threshold', 0):.3f})")
        print(f"   Best Accuracy:        {random.get('best_accuracy', 0):.4f} (threshold: {random.get('best_accuracy_threshold', 0):.3f})")
        print(f"   ROC AUC:              {random.get('roc_auc', 0):.4f}")
        print(f"   Average Precision:    {random.get('average_precision', 0):.4f}")
    else:
        print("   No random baseline metrics available")
    
    # 3. All-Positive Baseline Performance
    print("\n3. ALL-POSITIVE BASELINE PERFORMANCE")
    print("-" * 40)
    if all_positive:
        # For all-positive baseline, use test metrics format
        print(f"   Best F1 Score:        {all_positive.get('best_f1', 0):.4f} (threshold: {all_positive.get('best_f1_threshold', 0):.3f})")
        print(f"   Best Accuracy:        {all_positive.get('best_accuracy', 0):.4f} (threshold: {all_positive.get('best_accuracy_threshold', 0):.3f})")
        print(f"   ROC AUC:              {all_positive.get('roc_auc', 0):.4f}")
        print(f"   Average Precision:    {all_positive.get('average_precision', 0):.4f}")
    else:
        print("   No all-positive baseline metrics available")
    
    # 4. Performance Comparisons
    print("\n4. PERFORMANCE IMPROVEMENTS")
    print("-" * 40)
    if comparison:
        print("   vs Random Baseline:")
        print(f"     F1 Score improvement:      {comparison.get('actual_vs_random_f1_improvement', 0):+.4f}")
        print(f"     Accuracy improvement:      {comparison.get('actual_vs_random_accuracy_improvement', 0):+.4f}")
        print(f"     ROC AUC improvement:       {comparison.get('actual_vs_random_roc_auc_improvement', 0):+.4f}")
        
        print("\n   vs All-Positive Baseline:")
        print(f"     F1 Score improvement:      {comparison.get('actual_vs_all_positive_f1_improvement', 0):+.4f}")
        print(f"     Accuracy improvement:      {comparison.get('actual_vs_all_positive_accuracy_improvement', 0):+.4f}")
        print(f"     ROC AUC improvement:       {comparison.get('actual_vs_all_positive_roc_auc_improvement', 0):+.4f}")
    else:
        print("   No comparison metrics available")
    
    # 5. Summary Assessment
    print("\n5. PERFORMANCE ASSESSMENT")
    print("-" * 40)
    if actual and random and comparison:
        actual_f1 = actual.get('best_f1', 0)
        random_f1 = random.get('best_f1', 0)
        f1_improvement = comparison.get('actual_vs_random_f1_improvement', 0)
        
        actual_auc = actual.get('roc_auc', 0)
        random_auc = random.get('roc_auc', 0)
        auc_improvement = comparison.get('actual_vs_random_roc_auc_improvement', 0)
        
        # Assessment logic
        if f1_improvement > 0.1 and auc_improvement > 0.1:
            assessment = "[EXCELLENT] - Model significantly outperforms baselines"
        elif f1_improvement > 0.05 and auc_improvement > 0.05:
            assessment = "[GOOD] - Model moderately outperforms baselines"
        elif f1_improvement > 0 and auc_improvement > 0:
            assessment = "[FAIR] - Model slightly outperforms baselines"
        else:
            assessment = "[POOR] - Model does not outperform baselines"
        
        print(f"   Overall Assessment: {assessment}")
        
        # Percentage improvements
        if random_f1 > 0:
            f1_pct = (f1_improvement / random_f1) * 100
            print(f"   F1 Score improvement: {f1_pct:+.1f}% over random")
        
        if random_auc > 0:
            auc_pct = (auc_improvement / random_auc) * 100
            print(f"   ROC AUC improvement: {auc_pct:+.1f}% over random")
    else:
        print("   Cannot assess - insufficient metrics")
    
    print("\n" + "="*80)

def print_comparative_test_metrics(comparative_results):
    """
    Neatly print the results from calculate_comparative_test_metrics.
    
    Parameters:
    comparative_results: Dictionary returned by calculate_comparative_test_metrics
    """
    if not comparative_results:
        print("No comparative test results to display")
        return
    
    print("\n" + "="*80)
    print("COMPARATIVE TEST METRICS")
    print("="*80)
    
    # Extract main sections
    actual = comparative_results.get('actual', {})
    random = comparative_results.get('random_baseline', {})
    all_positive = comparative_results.get('all_positive_baseline', {})
    comparison = comparative_results.get('comparison', {})
    
    # 1. Actual Model Performance
    print("\n1. ACTUAL MODEL PERFORMANCE")
    print("-" * 40)
    if actual:
        print(f"   Accuracy:             {actual.get('accuracy', 0):.4f}")
        print(f"   F1 Score:             {actual.get('f1_score', 0):.4f}")
        print(f"   Precision:            {actual.get('precision', 0):.4f}")
        print(f"   Recall:               {actual.get('recall', 0):.4f}")
        print(f"   Specificity:          {actual.get('specificity', 0):.4f}")
        print(f"   False Positive Rate:  {actual.get('false_positive_rate', 0):.4f}")
        if 'roc_auc' in actual:
            print(f"   ROC AUC:              {actual.get('roc_auc', 0):.4f}")
        if 'average_precision' in actual:
            print(f"   Average Precision:    {actual.get('average_precision', 0):.4f}")
        print(f"   True Positive:        {actual.get('true_positive', 0)}")
        print(f"   False Positive:       {actual.get('false_positive', 0)}")
        print(f"   True Negative:        {actual.get('true_negative', 0)}")
        print(f"   False Negative:       {actual.get('false_negative', 0)}")
    else:
        print("   No actual metrics available")
    
    # 2. Random Baseline Performance
    print("\n2. RANDOM BASELINE PERFORMANCE")
    print("-" * 40)
    if random:
        print(f"   Accuracy:             {random.get('accuracy', 0):.4f}")
        print(f"   F1 Score:             {random.get('f1_score', 0):.4f}")
        print(f"   Precision:            {random.get('precision', 0):.4f}")
        print(f"   Recall:               {random.get('recall', 0):.4f}")
        print(f"   Specificity:          {random.get('specificity', 0):.4f}")
        print(f"   False Positive Rate:  {random.get('false_positive_rate', 0):.4f}")
        if 'roc_auc' in random:
            print(f"   ROC AUC:              {random.get('roc_auc', 0):.4f}")
        if 'average_precision' in random:
            print(f"   Average Precision:    {random.get('average_precision', 0):.4f}")
    else:
        print("   No random baseline metrics available")
    
    # 3. All-Positive Baseline Performance
    print("\n3. ALL-POSITIVE BASELINE PERFORMANCE")
    print("-" * 40)
    if all_positive:
        print(f"   Accuracy:             {all_positive.get('accuracy', 0):.4f}")
        print(f"   F1 Score:             {all_positive.get('f1_score', 0):.4f}")
        print(f"   Precision:            {all_positive.get('precision', 0):.4f}")
        print(f"   Recall:               {all_positive.get('recall', 0):.4f}")
        print(f"   Specificity:          {all_positive.get('specificity', 0):.4f}")
        print(f"   False Positive Rate:  {all_positive.get('false_positive_rate', 0):.4f}")
        if 'roc_auc' in all_positive:
            print(f"   ROC AUC:              {all_positive.get('roc_auc', 0):.4f}")
        if 'average_precision' in all_positive:
            print(f"   Average Precision:    {all_positive.get('average_precision', 0):.4f}")
    else:
        print("   No all-positive baseline metrics available")
    
    # 4. Performance Comparisons
    print("\n4. PERFORMANCE IMPROVEMENTS")
    print("-" * 40)
    if comparison:
        print("   vs Random Baseline:")
        print(f"     Accuracy improvement:      {comparison.get('actual_vs_random_accuracy_improvement', 0):+.4f}")
        print(f"     F1 Score improvement:      {comparison.get('actual_vs_random_f1_improvement', 0):+.4f}")
        print(f"     Precision improvement:     {comparison.get('actual_vs_random_precision_improvement', 0):+.4f}")
        print(f"     Recall improvement:        {comparison.get('actual_vs_random_recall_improvement', 0):+.4f}")
        if 'actual_vs_random_roc_auc_improvement' in comparison:
            print(f"     ROC AUC improvement:       {comparison.get('actual_vs_random_roc_auc_improvement', 0):+.4f}")
        if 'actual_vs_random_avg_precision_improvement' in comparison:
            print(f"     Avg Precision improvement: {comparison.get('actual_vs_random_avg_precision_improvement', 0):+.4f}")
        
        print("\n   vs All-Positive Baseline:")
        print(f"     Accuracy improvement:      {comparison.get('actual_vs_all_positive_accuracy_improvement', 0):+.4f}")
        print(f"     F1 Score improvement:      {comparison.get('actual_vs_all_positive_f1_improvement', 0):+.4f}")
        print(f"     Precision improvement:     {comparison.get('actual_vs_all_positive_precision_improvement', 0):+.4f}")
        print(f"     Recall improvement:        {comparison.get('actual_vs_all_positive_recall_improvement', 0):+.4f}")
        if 'actual_vs_all_positive_roc_auc_improvement' in comparison:
            print(f"     ROC AUC improvement:       {comparison.get('actual_vs_all_positive_roc_auc_improvement', 0):+.4f}")
        if 'actual_vs_all_positive_avg_precision_improvement' in comparison:
            print(f"     Avg Precision improvement: {comparison.get('actual_vs_all_positive_avg_precision_improvement', 0):+.4f}")
    else:
        print("   No comparison metrics available")
    
    # 5. Summary Assessment
    print("\n5. PERFORMANCE ASSESSMENT")
    print("-" * 40)
    if actual and random and comparison:
        actual_f1 = actual.get('f1_score', 0)
        random_f1 = random.get('f1_score', 0)
        f1_improvement = comparison.get('actual_vs_random_f1_improvement', 0)
        
        actual_acc = actual.get('accuracy', 0)
        random_acc = random.get('accuracy', 0)
        acc_improvement = comparison.get('actual_vs_random_accuracy_improvement', 0)
        
        # Assessment logic
        if f1_improvement > 0.1 and acc_improvement > 0.1:
            assessment = "[EXCELLENT] - Model significantly outperforms baselines"
        elif f1_improvement > 0.05 and acc_improvement > 0.05:
            assessment = "[GOOD] - Model moderately outperforms baselines"
        elif f1_improvement > 0 and acc_improvement > 0:
            assessment = "[FAIR] - Model slightly outperforms baselines"
        else:
            assessment = "[POOR] - Model does not outperform baselines"
        
        print(f"   Overall Assessment: {assessment}")
        
        # Percentage improvements
        if random_f1 > 0:
            f1_pct = (f1_improvement / random_f1) * 100
            print(f"   F1 Score improvement: {f1_pct:+.1f}% over random")
        
        if random_acc > 0:
            acc_pct = (acc_improvement / random_acc) * 100
            print(f"   Accuracy improvement: {acc_pct:+.1f}% over random")
    else:
        print("   Cannot assess - insufficient metrics")
    
    print("\n" + "="*80)

def sample_n_edges(G, sample_size=None, pos_ratio=None, min_embeddedness=None, strict=False):
    """
    Sample edges from a NetworkX graph with optional positive ratio and embeddedness constraints.
    
    Parameters:
        G (networkx.Graph): Input graph with edge attribute 'weight' indicating sign (+/-).
        sample_size (int, optional): Total number of edges to sample. If None, returns all qualifying edges.
        pos_ratio (float, optional): Desired ratio of positive edges (0 < pos_ratio < 1).
        min_embeddedness (int, optional): Minimum number of common neighbors required.
        strict (bool): If True, only return edges meeting all requirements even if < sample_size.
    
    Returns:
        list: Sampled list of (u, v, data) edge tuples.
    """
    random.seed(42)
    edge_list = list(G.edges(data=True))
    
    # Filter by embeddedness if specified
    if min_embeddedness is not None:
        G_undirected = G.to_undirected() if G.is_directed() else G
        qualifying_edges = [
            (u, v, data) for u, v, data in edge_list
            if len(list(nx.common_neighbors(G_undirected, u, v))) >= min_embeddedness
        ]
        logger.info(f"Embeddedness filtering: {len(edge_list)} → {len(qualifying_edges)} edges")
        
        if not qualifying_edges:
            if strict:
                logger.warning("No edges meet embeddedness requirements")
                return []
            else:
                logger.warning("No edges meet embeddedness requirements, using all edges")
                qualifying_edges = edge_list
    else:
        qualifying_edges = edge_list
    
    # Determine sample size
    target_size = sample_size or len(qualifying_edges)
    
    # Helper function to get edge weight
    def is_positive(edge):
        return edge[2].get('weight', 1) > 0
    
    # Sample with positive ratio constraint
    if pos_ratio is not None:
        pos_edges = [e for e in qualifying_edges if is_positive(e)]
        neg_edges = [e for e in qualifying_edges if not is_positive(e)]
        
        n_pos_target = int(round(target_size * pos_ratio))
        n_neg_target = target_size - n_pos_target
        
        n_pos_actual = min(n_pos_target, len(pos_edges))
        n_neg_actual = min(n_neg_target, len(neg_edges))
        
        # Handle shortfall
        shortfall = target_size - (n_pos_actual + n_neg_actual)
        if shortfall > 0 and not strict:
            # Fill shortfall by adjusting ratios within qualifying edges
            if len(pos_edges) > n_pos_actual:
                add_pos = min(shortfall, len(pos_edges) - n_pos_actual)
                n_pos_actual += add_pos
                shortfall -= add_pos
            if shortfall > 0 and len(neg_edges) > n_neg_actual:
                add_neg = min(shortfall, len(neg_edges) - n_neg_actual)
                n_neg_actual += add_neg
                shortfall -= add_neg
            
            # If still short, use non-qualifying edges
            if shortfall > 0:
                non_qualifying = [e for e in edge_list if e not in qualifying_edges]
                if non_qualifying:
                    additional = random.sample(non_qualifying, min(shortfall, len(non_qualifying)))
                    pos_edges.extend([e for e in additional if is_positive(e)])
                    neg_edges.extend([e for e in additional if not is_positive(e)])
                    n_pos_actual = min(n_pos_target + shortfall//2, len(pos_edges))
                    n_neg_actual = min(target_size - n_pos_actual, len(neg_edges))
        
        # Warn about unmet requirements
        if n_pos_actual < n_pos_target:
            logger.warning(f"Only {n_pos_actual}/{n_pos_target} positive edges available")
        if n_neg_actual < n_neg_target:
            logger.warning(f"Only {n_neg_actual}/{n_neg_target} negative edges available")
        
        # Sample and combine
        pos_sample = random.sample(pos_edges, n_pos_actual) if n_pos_actual > 0 else []
        neg_sample = random.sample(neg_edges, n_neg_actual) if n_neg_actual > 0 else []
        sample = pos_sample + neg_sample
        random.shuffle(sample)
        
    else:
        # Simple sampling without ratio constraint
        available_edges = qualifying_edges
        
        # Fill with non-qualifying edges if needed and not strict
        if not strict and target_size > len(available_edges):
            non_qualifying = [e for e in edge_list if e not in qualifying_edges]
            needed = target_size - len(available_edges)
            if non_qualifying:
                logger.warning(f"Adding {min(needed, len(non_qualifying))} non-qualifying edges")
                available_edges.extend(non_qualifying[:needed])
        
        actual_size = min(target_size, len(available_edges))
        if strict and actual_size < target_size:
            logger.warning(f"Strict mode: Only {actual_size}/{target_size} edges qualify")
        
        sample = random.sample(available_edges, actual_size)
    
    # Single comprehensive report
    pos_count = sum(1 for e in sample if is_positive(e))
    embeddedness_count = sum(1 for e in sample if e in qualifying_edges) if min_embeddedness is not None else len(sample)
    
    print(f"Sampled {len(sample)} edges: {pos_count} positive, {embeddedness_count} meeting embeddedness ≥ {min_embeddedness if min_embeddedness is not None else 0}")
    
    return sample