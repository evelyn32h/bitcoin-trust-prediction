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
  - `soc-sign-bitcoinotc.csv`: Bitcoin OTC trust network data
- `notebooks/`: Scripts for exploration and evaluation
  - `explore_data.py`: Data exploration and statistics (basic network analysis, degree distribution, embeddedness)
  - `cross_validation_eval.py`: Original evaluation with 10-fold cross-validation (contains data leakage)
  - **`strict_evaluation.py`**: **Main evaluation script** - implements Vide's correct method avoiding data leakage
  - `strict_evaluation_test.py`: Quick test version (3 folds, 20 edges) for rapid testing
  - `strict_evaluation_sample.py`: Sampled version (5000 edges) for faster approximate results
  - `compare_methods.py`: Compares original method vs strict evaluation to show data leakage impact
  - `optimization_experiments.py`: Tests preprocessing strategies (embeddedness filtering, balanced dataset)
  - `predict_edge_sign.py`: Demo of single edge prediction (by Vide)
  - `sampled_leave_one_out.py`: Leave-one-out evaluation on sampled edges (placeholder)
  - `test_evaluation.py`: Unit tests for evaluation functions
  - `strict_evaluation_resume.py`: Checkpoint support for resuming long evaluations (incomplete)
- `src/`: Source code
  - `data_loader.py`: Data loading functions
  - `preprocessing.py`: Data preprocessing (filtering, balancing, connectivity)
  - `feature_extraction.py`: Feature extraction using higher-order cycles (optimized by Vide)
  - `models.py`: Prediction models (logistic regression with scaling)
  - `evaluation.py`: Evaluation functions and metrics
  - `utilities.py`: Helper functions for feature statistics
- `results/`: Stores experiment results and visualizations
  - Original method results (cross-validation)
  - Strict evaluation results (avoiding data leakage)
  - Comparison plots and confusion matrices

## Project Goals
1. Implement Chiang et al.'s edge sign prediction algorithm
2. Improve the original algorithm by incorporating edge weight information
3. Extend the algorithm to predict both sign and weight of edges

## Key Results (Part 1)

### Data Leakage Discovery - Our Main Contribution
The original evaluation method had data leakage. When correctly evaluated using our strict method:
- AUC drops from ~0.97 to ~0.63
- Model predicts almost all edges as positive (100% FPR)
- This reveals the true difficulty of edge sign prediction

### Performance Comparison
| Method | k | Accuracy | AUC | False Positive Rate | Note |
|--------|---|----------|-----|---------------------|------|
| Original (with leakage) | 3 | 92.75% | 0.8514 | 65.32% | ❌ Incorrect |
| Original (with leakage) | 4 | 97.10% | 0.9669 | 26.28% | ❌ Incorrect |
| **Strict (no leakage)** | 3 | 90.0% | **0.626** | 100% | ✅ **True performance** |
| **Strict (no leakage)** | 4 | 90.0% | **0.627** | 100% | ✅ **True performance** |

## Notebooks Description

### Core Evaluation Scripts
- **`strict_evaluation.py`** 🌟: **The most important file** - correct evaluation implementation following Vide's method. Extracts features from test set (excluding current edge), completely avoiding data leakage. Takes ~40 minutes for full evaluation.

- **`strict_evaluation_test.py`**: Quick test version created for rapid iteration while developing the strict evaluation. Uses 3 folds and 20 edges per fold, runs in ~14 seconds.

- **`cross_validation_eval.py`**: Original evaluation using standard cross-validation. Contains data leakage as features are extracted from the full graph. Shows inflated performance but kept for comparison.

- **`compare_methods.py`**: Runs both methods side-by-side to demonstrate the impact of data leakage. Clearly shows the performance drop when evaluated correctly.

### Testing and Optimization
- **`strict_evaluation_sample.py`**: Evaluates on a sample of 5000 edges for faster approximate results. Good for initial testing before full runs.

- **`optimization_experiments.py`**: Tests various preprocessing strategies:
  - Filtering by minimum embeddedness
  - Creating balanced datasets
  - Neutral edge removal

### Data Analysis and Utilities
- **`explore_data.py`**: Initial data exploration including:
  - Network statistics (nodes, edges, components)
  - Edge weight distribution
  - Embeddedness analysis
  - Degree distribution

- **`predict_edge_sign.py`**: Vide's implementation showing how to:
  - Suppress edges for testing
  - Scale features properly
  - Make individual predictions

- **`test_evaluation.py`**: Unit tests for evaluation functions using simulated data.

### Incomplete/Placeholder Scripts
- **`sampled_leave_one_out.py`**: Placeholder for leave-one-out evaluation. Marked as "wait for Vide".
- **`strict_evaluation_resume.py`**: Intended for checkpoint support to resume long evaluations. Not fully implemented.

## How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Run data exploration
python notebooks/explore_data.py

# For immediate results - quick test (14 seconds)
python notebooks/strict_evaluation_test.py

# For full evaluation - MAIN RESULT (~40 minutes)
python notebooks/strict_evaluation.py

# Compare methods to see data leakage impact
python notebooks/compare_methods.py

# Run Part 1 evaluation (main experiment)
python notebooks/cross_validation_eval.py

# Run optimization experiments (optional)
python notebooks/optimization_experiments.py