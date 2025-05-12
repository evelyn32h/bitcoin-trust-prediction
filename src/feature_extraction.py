import networkx as nx
import numpy as np

def get_pos_neg_adjacency_matrix(G):
    """
    Convert a graph to two adjacency matrices: one for positive edges and one for negative edges.
    Positive edges have weights > 0, and negative edges have weights < 0.
    
    Parameters:
    G: NetworkX directed graph
    
    Returns:
    A_pos: Adjacency matrix for positive edges (weights > 0)
    A_neg: Adjacency matrix for negative edges (absolute values of weights < 0)
    """
    print(f"Graph size: {len(G.nodes())} nodes, {len(G.edges())} edges")
    
    # Convert graph to adjacency matrix
    A_pos = nx.to_numpy_array(G, dtype=np.float64, weight='weight', nodelist=sorted(G.nodes()))
    A_neg = np.copy(A_pos)
    
    # Setparate positive and negative edges
    A_pos[A_pos < 0] = 0  # Keep only positive weights
    A_neg[A_neg > 0] = 0  # Keep only negative weights
    A_neg = np.abs(A_neg)  # Convert negative weights to positive for processing
  
    return A_pos, A_neg

def extract_undirected_hoc_features(A_pos, A_neg, edges, max_k):
    """
    Extract higher-order cycle (HOC) features for each edge (u, v) in an undirected graph.
    The function computes matrix products for sequences of positive and negative adjacency matrices
    and symmetrizes the results to capture undirected cycles.
    
    Parameters:
    A_pos: Adjacency matrix for positive edges (binary, 1s and 0s)
    A_neg: Adjacency matrix for negative edges (binary, 1s and 0s)
    edges: List of edges (u, v) for which to extract features
    max_k: Maximum order of cycles to consider
    
    Returns:
    features: Dictionary where keys are edge tuples and values are lists of features for each k
    """
    print(f"Extracting higher-order cycle of length < {max_k}")
        
    matrix_products =  compute_hoc_matrices(A_pos, A_neg, edges, 3, max_k, current_products={"+": A_pos, "-": A_neg})
    
    print("Symmteretricizing products...")
    
    # Symmetricize the products
    symmetricized_products = {}
    for seq, product in matrix_products.items():
        reverse_seq = seq[::-1]
        if reverse_seq in symmetricized_products:
            # Already processed the reverse sequence
            continue
        if reverse_seq == seq:
            # Already symmetric
            symmetricized_products[seq] = product
        else:
            print(f"Summing {seq} and {reverse_seq}")
            symmetricized_products[seq] = product + matrix_products[reverse_seq]

    # Convert the dictionary values to a list
    symmetricized_products = list(symmetricized_products.values())

    print("Extracting features...")
    
    # Extract features
    features = {edge: [] for edge in edges}  # Initialize features for each edge
        
    for u, v in edges:
        for product in symmetricized_products:
            features[(u, v)].append(product[u, v])
            
    return features  


def compute_hoc_matrices(A_pos, A_neg, edges, k, max_k, current_products=None):
    """
    Recursively compute higher-order cycle (HOC) matrices for sequences of positive and negative adjacency matrices.
    Each recursion step computes matrix products for the next order of cycles.
    
    Parameters:
    A_pos: Adjacency matrix for positive edges (binary, 1s and 0s)
    A_neg: Adjacency matrix for negative edges (binary, 1s and 0s)
    edges: List of edges (u, v) for which to extract features
    k: Current order of cycles being processed
    max_k: Maximum order of cycles to consider
    current_products: Dictionary of current matrix products for sequences (default: None)
    
    Returns:
    current_products: Dictionary where keys are sequences of "+" and "-" and values are the corresponding matrix products
    """
    # Base case: stop recursion when k exceeds max_k
    if k > max_k:
        print(f"Done calculating matrices {list(current_products.keys())}")
        return current_products

    print(f"Calculating cycles of length {k}, requiring {len(current_products)*2} matrix products:")
    
    next_products = {}
    for idx, (seq, product) in enumerate(current_products.items()):        
        # Append the new products for the next order
        print(f"({seq} dot +)", end=", ", flush=True)
        next_products[seq + "+"] = np.dot(product, A_pos)
        print(f"({seq} dot -)", end=", ", flush=True)
        next_products[seq + "-"] = np.dot(product, A_neg)
        
    print("finished")
    
    # Recur for the next order (k+1)
    return compute_hoc_matrices(A_pos, A_neg, edges, k + 1, max_k, next_products)


def feature_matrix_from_graph(G, edges=None, k=4):
    """
    Create a feature matrix for all edges or a subset of edges from a directed graph.
    The function extracts higher-order cycle features using adjacency matrices for positive and negative edges.
    
    Parameters:
    G: NetworkX directed graph
    edges: List of edges to extract features for (default: all edges in G)
    k: Maximum order to consider for higher-order features (default: 3)
    
    Returns:
    X: Feature matrix (NumPy array) where each row corresponds to an edge
    y: Label vector (NumPy array) containing edge weights
    edges: List of edges corresponding to rows in X
    """

    if edges is None:
        edges = list(G.edges(data=True))  # Include edge attributes

    print(f"Extracting features for {len(edges)} edges")

    # Initialize feature matrix, labels, and edge list
    X = []
    y = []
    edge_list = [(u, v) for u, v, _ in edges]  # Extract edge tuples
    
    # Convert graph to adjacency matrices
    G_undirected = G.to_undirected()
    A_pos, A_neg = get_pos_neg_adjacency_matrix(G_undirected)

    # Extract higher-order features for all edges
    higher_order_features = extract_undirected_hoc_features(A_pos, A_neg, edge_list, k)

    for idx, (u, v, data) in enumerate(edges):
        # Add edge features
        X.append(higher_order_features[(u, v)])

        # Extract edge label (edge weight)
        y.append(data.get('weight', 0))  # Default weight is 0 if not present

    print(f"Feature matrix shape: {len(X)} rows, {len(X[0]) if X else 0} columns")
    # print(f"Feature matrix sample values: {X[:5] if len(X) > 5 else X}")
    
    return np.array(X), np.array(y), edge_list