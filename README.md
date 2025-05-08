# Bitcoin Trust Network Edge Sign Prediction

## Project Overview
This project implements and extends the link prediction algorithm for signed networks proposed by Chiang et al. in their paper "Exploiting Longer Cycles for Link Prediction in Signed Networks". We use the Bitcoin OTC Web of Trust dataset, which represents trust ratings between users.

## Dataset Information
- Source: [Bitcoin OTC Web of Trust](https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html)
- Nodes: 5,881 (representing users)
- Edges: 35,592 (representing trust relationships)
- Edge weight range: -10 to +10 (positive = trust, negative = distrust)
- Positive edge ratio: approximately 89%

## Project Structure
- `data/`: Contains the dataset
- `notebooks/`: Scripts for exploration and visualization
- `src/`: Source code
  - `data_loader.py`: Data loading functions
  - `preprocessing.py`: Data preprocessing 
  - `feature_extraction.py`: Feature extraction
  - `models.py`: Prediction models
  - `evaluation.py`: Evaluation functions
- `results/`: Stores experiment results

## Project Goals
1. Implement Chiang et al.'s edge sign prediction algorithm
2. Improve the original algorithm by incorporating edge weight information
3. Extend the algorithm to predict both sign and weight of edges

## How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Run data exploration
python notebooks/explore_data.py