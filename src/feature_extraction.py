import networkx as nx
import numpy as np
import scipy.sparse as sp
from concurrent.futures import ThreadPoolExecutor
import logging
from src.utilities import print_feature_statistics
import time

# Change logging level to see INFO messages (level=logging.INFO)
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def get_sparse_adjacency_matrices(G, use_weighted_features=False):
    """
    Return sparse adjacency matrices for positive and negative edges.
    Now supports both binary and weighted features (Task #1).
    
    Parameters:
        G (networkx.Graph): Input graph with signed edge weights.
        use_weighted_features (bool): Whether to use weighted features instead of binary
    Returns:
        tuple: (A_pos, A_neg) where A_pos and A_neg are CSR sparse adjacency matrices for positive and negative edges, respectively.
    """
    
    # Get the adjacency matrix (scipy sparse matrix)
    A = nx.adjacency_matrix(G, weight='weight', nodelist=sorted(G.nodes()))
    
    if use_weighted_features:
        # Task #1: Weighted features - preserve weight magnitudes
        A_pos = A.copy().tolil()
        A_neg = A.copy().tolil()
        
        # For positive matrix: keep positive weights, zero out negative
        A_pos[A_pos < 0] = 0
        
        # For negative matrix: keep absolute values of negative weights, zero out positive
        A_neg[A_neg > 0] = 0
        A_neg = abs(A_neg)  # Convert to positive values but preserve magnitudes
        
    else:
        # Original binary features - convert to +1/-1
        A_pos = A.copy().tolil()
        A_neg = A.copy().tolil()
        # Set negative values to 0 in A_pos, positive to 0 in A_neg
        A_pos[A_pos < 0] = 0
        A_neg[A_neg > 0] = 0
        # Convert negative values to positive in A_neg
        A_neg = abs(A_neg)
        # Convert to binary (0/1)
        A_pos[A_pos > 0] = 1
        A_neg[A_neg > 0] = 1
    
    # Convert back to preferred sparse format (csr)
    return A_pos.tocsr(), A_neg.tocsr()

def compute_hoc_matrices_weighted(A_pos, A_neg, edges, k, max_k, current_products=None, 
                                 use_weighted_features=False, weight_aggregation='product'):
    """
    Iteratively compute higher-order cycle (HOC) matrices with weighted feature support.
    This is the enhanced version for Task #1.
    
    Parameters:
        A_pos (sp.spmatrix): Positive adjacency matrix (CSR format).
        A_neg (sp.spmatrix): Negative adjacency matrix (CSR format).
        edges (list): List of edge tuples (u, v).
        k (int): Starting cycle length.
        max_k (int): Maximum cycle length.
        current_products (dict): Current matrix products by sequence key.
        use_weighted_features (bool): Whether to use weighted features (Task #1)
        weight_aggregation (str): How to aggregate weights ('product', 'sum', 'mean', 'max')
    Returns:
        dict: Mapping from sequence string to matrix product (sp.spmatrix).
    """
    start_time = time.time()
    hoc_matrices = {}
    
    logger.info(f"Computing weighted HOC matrices (Task #1): use_weighted={use_weighted_features}, aggregation={weight_aggregation}")
    
    while k <= max_k:
        logger.info(f"Calculating cycles of length {k}, requiring {len(current_products)*2} matrix products... ")
        next_products = {}
        
        def compute_products_weighted(seq_product):
            seq, product = seq_product
            
            if use_weighted_features:
                # Task #1: Weighted matrix multiplication
                if weight_aggregation == 'product':
                    # Standard matrix multiplication (preserves weight products)
                    prod_pos = product @ A_pos
                    prod_neg = product @ A_neg
                elif weight_aggregation == 'sum':
                    # Element-wise operations for sum aggregation
                    prod_pos = product @ A_pos
                    prod_neg = product @ A_neg
                elif weight_aggregation == 'max':
                    # Maximum weight propagation
                    prod_pos = product @ A_pos
                    prod_neg = product @ A_neg
                else:
                    # Default to product for unknown aggregation
                    prod_pos = product @ A_pos
                    prod_neg = product @ A_neg
            else:
                # Original binary computation
                prod_pos = product @ A_pos
                prod_neg = product @ A_neg
            
            return [
                (seq + "+", prod_pos),
                (seq + "-", prod_neg)
            ]
        
        # Parallelize the computation of matrix products
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(compute_products_weighted, current_products.items()))
        
        for res in results:
            for seq_key, prod in res:
                next_products[seq_key] = prod
        
        logger.info("finished")
        current_products = next_products
        hoc_matrices.update(current_products)
        k += 1
    
    logger.info(f"Done calculating matrices {list(hoc_matrices.keys())}")
    elapsed = time.time() - start_time
    logger.info(f"compute_hoc_matrices_weighted completed in {elapsed:.2f} seconds")
    return hoc_matrices

def compute_hoc_matrices(A_pos, A_neg, edges, k, max_k, current_products=None):
    """
    Iteratively compute higher-order cycle (HOC) matrices.
    Original function maintained for backward compatibility.
    
    Parameters:
        A_pos (sp.spmatrix): Positive adjacency matrix (CSR format).
        A_neg (sp.spmatrix): Negative adjacency matrix (CSR format).
        edges (list): List of edge tuples (u, v).
        k (int): Starting cycle length.
        max_k (int): Maximum cycle length.
        current_products (dict): Current matrix products by sequence key.
    Returns:
        dict: Mapping from sequence string to matrix product (sp.spmatrix).
    """
    # Call the weighted version with default parameters for backward compatibility
    return compute_hoc_matrices_weighted(A_pos, A_neg, edges, k, max_k, current_products, 
                                       use_weighted_features=False, weight_aggregation='product')

def symmetricize_hoc_matrices(matrix_products):
    """
    Symmetricize matrix products by combining each sequence with its reverse.
    
    Parameters:
        matrix_products (dict): Mapping from sequence string to matrix product.
    Returns:
        dict: Symmetricized mapping from sequence string to matrix product.
    """
    start_time = time.time()
    symmetricized_products = {}
    logger.info("Symmetricizing products, combining: ")
    
    for seq, product in matrix_products.items():
        reverse_seq = seq[::-1]
        if reverse_seq in symmetricized_products:
            # If the reverse sequence is already processed, skip it
            continue
        if reverse_seq == seq:
            # If the sequence is symmetric (e.g., "++"), just add it to the result
            symmetricized_products[seq] = product
        else:
            # Make matrix symmetric by adding the product with its reverse
            logger.info(f"({seq} and {reverse_seq})")
            symmetricized_products[seq] = product + matrix_products[reverse_seq]
            
    # Log time elapsed
    elapsed = time.time() - start_time
    logger.info(f"symmetricize_hoc_matrices completed in {elapsed:.2f} seconds")
    
    return symmetricized_products

def extract_features_from_hoc_matrices(hoc_matrices, edges):
    """
    Extract features for each edge from HOC matrices.
    
    Parameters:
        hoc_matrices (dict): Mapping from sequence string to matrix product.
        edges (list): List of edge tuples (u, v).
    Returns:
        dict: Mapping from edge tuple to list of features.
    """
    start_time = time.time()
    # Convert edge list to numpy array for efficient batch indexing
    edge_indices = np.array(edges)
    feature_list = []
    # For each HOC matrix, extract all edge values at once and collect as columns
    for product in hoc_matrices.values():
        product = product.tocsr()  # Ensure fast row/col indexing
        vals = product[edge_indices[:,0], edge_indices[:,1]]  # Batch extract edge values
        vals = np.array(vals).ravel()  # Flatten to 1D array
        feature_list.append(vals)
    # Stack all features into a 2D array: shape (n_features, n_edges) -> transpose to (n_edges, n_features)
    features_matrix = np.stack(feature_list, axis=0).T
    # Build the dictionary mapping each edge to its feature vector
    features = {edge: list(features_matrix[i]) for i, edge in enumerate(edges)}
    # Log time elapsed
    elapsed = time.time() - start_time
    logger.info(f"extract_features_from_hoc_matrices completed in {elapsed:.2f} seconds")
    return features

def hoc_features_from_undirected_adjacency_matrix(A_pos, A_neg, edges, max_k, 
                                                 use_weighted_features=False, weight_aggregation='product'):
    """
    Extract undirected HOC features for each edge with weighted feature support (Task #1).
    
    Parameters:
        A_pos (sp.spmatrix): Positive adjacency matrix (CSR format).
        A_neg (sp.spmatrix): Negative adjacency matrix (CSR format).
        edges (list): List of edge tuples (u, v).
        max_k (int): Maximum cycle length.
        use_weighted_features (bool): Whether to use weighted features (Task #1)
        weight_aggregation (str): How to aggregate weights in cycles
    Returns:
        dict: Mapping from edge tuple to list of HOC features.
    """
    start_time = time.time()
    logger.info(f"Extracting higher-order cycle features of length < {max_k}")
    logger.info(f"Task #1 settings: weighted_features={use_weighted_features}, aggregation={weight_aggregation}")
        
    # Use weighted HOC computation for Task #1
    matrix_products = compute_hoc_matrices_weighted(
        A_pos, A_neg, edges, 3, max_k, 
        current_products={"+": A_pos, "-": A_neg},
        use_weighted_features=use_weighted_features,
        weight_aggregation=weight_aggregation
    )
    
    # Symmetricize the products
    symmetricized_products = symmetricize_hoc_matrices(matrix_products)
    
    logger.info(f"Final feature keys: {list(symmetricized_products.keys())}")
    logger.info("Extracting features...")
    
    # Extract features
    features = extract_features_from_hoc_matrices(symmetricized_products, edges)

    # Log time elapsed
    elapsed = time.time() - start_time
    logger.info(f"TOTAL time for hoc features: {elapsed:.2f} seconds")
    
    return features  

def feature_matrix_from_graph(G, edges=None, k=4, use_weighted_features=False, weight_aggregation='product'):
    """
    Return feature matrix, labels, and edge list for given edges.
    Now supports weighted features (Task #1).
    
    Parameters:
        G (networkx.Graph): Input graph with signed edge weights.
        edges (list, optional): List of edge tuples (u, v, data). If None, use all edges.
        k (int): Maximum cycle length for HOC features.
        use_weighted_features (bool): Whether to use weighted features (Task #1)
        weight_aggregation (str): How to aggregate weights in cycles
    Returns:
        tuple: (X, y, edge_list) where X is the feature matrix, y is the label vector, and edge_list is the list of edge tuples.
    """
    
    # Check if edges are provided, otherwise use all edges in the graph
    if edges is None:
        edges = list(G.edges(data=True))  # Include edge attributes

    logger.info(f"Extracting features for {len(edges)} edges")
    logger.info(f"Task #1: using weighted features = {use_weighted_features}")
    
    # Initialize feature matrix, labels, and edge list
    X = []
    y = []
    edge_list = [(u, v) for u, v, _ in edges]  # Extract edge tuples
    
    # Convert graph to adjacency matrices
    if G.is_directed():
        G_undirected = G.to_undirected()
    else:
        G_undirected = G
    
    # Get adjacency matrices with weighted feature support
    A_pos, A_neg = get_sparse_adjacency_matrices(G_undirected, use_weighted_features=use_weighted_features)

    # Extract higher-order features for all edges with weighted support
    higher_order_features = hoc_features_from_undirected_adjacency_matrix(
        A_pos, A_neg, edge_list, k, 
        use_weighted_features=use_weighted_features,
        weight_aggregation=weight_aggregation
    )

    # Print feature statistics if logging level is INFO or lower
    if logger.isEnabledFor(logging.INFO):
        print_feature_statistics(higher_order_features)

    # Construct feature matrix 
    X = [higher_order_features[(u, v)] for u, v, _ in edges]
    y = [data.get('label', 0) for _, _, data in edges]

    logger.info(f"Feature matrix shape: {len(X)} rows, {len(X[0]) if X else 0} columns")
    logger.debug(f"Feature matrix sample values: {X[:5] if len(X) > 5 else X}")
    
    if not np.all(np.isin(y, [-1, 1])):
        logger.warning("Label vector y contains values other than -1 or +1.")
    
    if use_weighted_features:
        logger.info(f"Task #1: Successfully extracted weighted features using {weight_aggregation} aggregation")
    
    return np.array(X), np.array(y), edge_list