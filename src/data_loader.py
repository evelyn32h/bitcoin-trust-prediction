import networkx as nx
import pandas as pd
import numpy as np
import os

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

if __name__ == "__main__":
    # Test data loading
    data_path = os.path.join('..', 'data', 'soc-sign-bitcoinotc.csv')
    G, df = load_bitcoin_data(data_path)
    print("Data sample:")
    print(df.head())