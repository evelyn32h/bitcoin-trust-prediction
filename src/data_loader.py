import json
import networkx as nx
import pandas as pd
import numpy as np
import os
import joblib

def save_metrics_to_json(metrics, save_path):
    """
    Save metrics dictionary to JSON file
    
    Parameters:
    metrics: Dictionary containing metrics
    save_path: Path to save the JSON file
    """
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    metrics_serializable = convert_numpy_types(metrics)
    
    with open(save_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    
    print(f"Metrics saved to {save_path}")

def load_bitcoin_data(filepath):
    """
    Load Bitcoin OTC dataset and convert to NetworkX directed weighted graph
    
    Parameters:
    filepath: Path to the CSV data file
    
    Returns:
    G: NetworkX directed graph
    df: DataFrame of the original data
    """
    print(f"Loading data from {filepath}...")
    # 使用逗号作为分隔符
    df = pd.read_csv(filepath, sep=',', header=None, 
                    names=['source', 'target', 'rating', 'time'])
    
    print(f"Data loaded successfully with {len(df)} edges.")
    
    # Create directed weighted graph
    G = nx.DiGraph()
    
    # Add edges and weights
    for _, row in df.iterrows():
        G.add_edge(row['source'], row['target'], 
                   weight=row['rating'], 
                   time=row['time'])
    
    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G, df

def load_undirected_graph_from_csv(filepath):
    """
    Load an undirected NetworkX graph from a CSV file with columns: source, target, rating, time.
    Returns:
        G: NetworkX undirected graph
        df: DataFrame of the original data
    """
    df = pd.read_csv(filepath, sep=',', header=None, names=['source', 'target', 'label', 'weight', 'time'])
    G = nx.from_pandas_edgelist(df, source='source', target='target', edge_attr=['label', 'weight', 'time'], create_using=nx.Graph())
    return G, df

def save_graph_to_csv(G, filepath):
    """
    Save a NetworkX graph to a CSV file with columns: source, target, rating, time.
    The order and field names match the original Bitcoin OTC dataset.
    """
    df = nx.to_pandas_edgelist(G)
    # Ensure 'rating' and 'time' columns exist and are in the correct order
    if 'rating' not in df:
        df['rating'] = None
    if 'time' not in df:
        df['time'] = None
    if 'label' not in df:
        df['label'] = None
        
    df = df[['source', 'target', 'label', 'weight', 'time']]
    df.to_csv(filepath, index=False, header=False)

def save_model(model, scaler, out_dir, fold):
    """
    Saves the model and scaler to out_dir for the given fold.
    """
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, f'model_fold_{fold}.joblib')
    scaler_path = os.path.join(out_dir, f'scaler_fold_{fold}.joblib')
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Saved model and scaler for fold {fold} to {out_dir}")
    
def load_models(training_dir, n_folds):
    """
    Loads models and scalers from training_dir, one per fold.
    Returns a list of (model, scaler) tuples.
    """
    import os
    models_and_scalers = []
    for i in range(n_folds):
        model_path = os.path.join(training_dir, f'model_fold_{i}.joblib')
        scaler_path = os.path.join(training_dir, f'scaler_fold_{i}.joblib')
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        models_and_scalers.append((model, scaler))
    return models_and_scalers
    
def save_prediction_results(true_labels, predicted_labels, predicted_probabilities, out_dir):
    """
    Save prediction results (true labels, predicted labels, predicted probabilities) to a single CSV file in the specified directory.
    Each column represents one parameter.
    """
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame({
        "true_label": true_labels,
        "predicted_label": predicted_labels,
        "predicted_probability": predicted_probabilities
    })
    df.to_csv(os.path.join(out_dir, "prediction_results.csv"), index=False)    

def load_prediction_results(out_dir):
    """
    Load prediction results from a single CSV file in the specified directory.
    Returns true_labels, predicted_labels, predicted_probabilities as numpy arrays.
    """
    file_path = os.path.join(out_dir, "prediction_results.csv")
    df = pd.read_csv(file_path)
    true_labels = df["true_label"].to_numpy()
    predicted_labels = df["predicted_label"].to_numpy()
    predicted_probabilities = df["predicted_probability"].to_numpy()
    return true_labels, predicted_labels, predicted_probabilities

def save_metrics(metrics, out_dir):
    """
    Save a (possibly nested) metrics dictionary as a JSON file at the given path.
    FIXED: Now saves as .json file with JSON content (was .csv with JSON content)
    """
    os.makedirs(out_dir, exist_ok=True)
    file_path = os.path.join(out_dir, "metrics.json")  # Changed from .csv to .json
    with open(file_path, 'w') as f:
        json.dump(metrics, f, indent=2)
        
def load_metrics(metrics_path):
    """
    Load metrics from JSON file.
    """
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    return metrics

def save_config(config_dict, out_dir, filename="config_used.yaml"):
    """
    Save a configuration dictionary as a YAML file in the specified directory.
    """
    import yaml
    os.makedirs(out_dir, exist_ok=True)
    config_path = os.path.join(out_dir, filename)
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f)
    print(f"Saved config to {config_path}")

if __name__ == "__main__":
    # Test data loading
    data_path = os.path.join('..', 'data', 'soc-sign-bitcoinotc.csv')
    G, df = load_bitcoin_data(data_path)
    print("Data sample:")
    print(df.head())