# Bitcoin Trust Network Edge Sign Prediction

## Project Overview
This project implements and extends the link prediction algorithm for signed networks proposed by Chiang et al. in their paper "Exploiting Longer Cycles for Link Prediction in Signed Networks". We use the Bitcoin OTC Web of Trust dataset and conduct comprehensive experiments on higher-order cycle features, embeddedness filtering, weighting strategies, and class-imbalance effects.

## Dataset Information
- Source: [Bitcoin OTC Web of Trust](https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html)
- Nodes: 5,881 (representing users)
- Edges: 35,592 (representing trust relationships)
- Edge weight range: -10 to +10 (positive = trust, negative = distrust)
- Positive edge ratio: approximately 89%

## Repository Structure
```
BITCOIN-TRUST-PREDICTION-MAIN/
├── data/                                   # Raw dataset
│   └── soc-sign-bitcoinotc.csv
├── notebooks/                              # Experiment scripts
│   ├── analyze_all_results.py              # Generates 8 summary plots
│   ├── analyze_feature_distributions.py    # Creates 6 feature analysis plots
│   ├── preprocess.py                       # Preprocessing & BFS-split script
│   ├── run_batch_experiments.py            # Batch runner for main experiments
│   ├── run_experiment.py                   # Single experiment runner (pos/neg ratios)
│   ├── train_model.py                      # Cross-validation training logic
│   └── validate_model.py                   # Hold-out test evaluation (no data leakage)
├── plots/                                  # Generated visualizations
│   ├── complete_results_analysis/          # 8 summary plots
│   └── enhanced_feature_analysis/          # 6 feature distribution plots
├── results/                                # Experiment outputs
│   ├── weighted_optimal/                   # Weighted vs unweighted experiments
│   ├── embed_0_optimal/                    # Embeddedness filtering levels (0/1/2)
│   ├── aggregation_max_optimal/            # Bidirectional aggregation methods
│   ├── cycle_length_3_optimal/             # Cycle length comparisons (3/4/5)
│   ├── pos_ratio_50_50_optimal/            # Class imbalance experiments
│   └── ...                                 # 18 total experiment types
├── src/                                    # Core modules
│   ├── data_loader.py                      # Data loading utilities
│   ├── preprocessing.py                    # Data preprocessing functions
│   ├── feature_extraction.py               # Higher-order cycle feature extraction
│   ├── models.py                           # Logistic regression models
│   ├── evaluation.py                       # Strict evaluation (no data leakage)
│   └── utilities.py                        # Helper functions
├── config.yaml                             # Global configuration
└── requirements.txt                        # Dependencies
```

## Project Goals
1. Implement Chiang et al.'s edge sign prediction algorithm with strict evaluation
2. Conduct comprehensive experiments on cycle features, embeddedness, and weighting
3. Evaluate class imbalance effects and bidirectional edge handling strategies
4. Generate complete analysis with 14 visualization plots

## Key Experiments

### Batch Experiments (run_batch_experiments.py)
- **Weighted vs Unweighted Features**: Compare raw vs weighted cycle features
- **Embeddedness Filtering**: Test 0/1/2 common neighbor thresholds
- **Aggregation Methods**: Compare max/sum/stronger bidirectional edge handling
- **Cycle Lengths**: Evaluate 3/4/5-cycle features
- **Dataset Comparison**: Bitcoin OTC vs Epinions subset baseline

### Class Imbalance Experiments (run_experiment.py)
Test positive:negative ratios: 50:50, 60:40, 70:30, 80:20, 90:10

### Analysis & Visualization
- **Summary Analysis**: 8 comprehensive comparison plots
- **Feature Analysis**: 6 distribution and impact analysis plots

## How to Run

### Quick Start
```bash
# Setup
git clone <repository-url>
cd bitcoin-trust-prediction-main
pip install -r requirements.txt

# Download dataset
wget https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz
gunzip soc-sign-bitcoinotc.csv.gz
mv soc-sign-bitcoinotc.csv data/
```

### Run All Experiments
```bash
cd notebooks

# 1. Run all batch experiments (weighted, embeddedness, aggregation, cycle length)
python run_batch_experiments.py

# 2. Run class imbalance experiments
python run_experiment.py --config ../config.yaml --override "pos_edges_ratio=0.5;min_train_embeddedness=1;cycle_length=4"
python run_experiment.py --config ../config.yaml --override "pos_edges_ratio=0.6;min_train_embeddedness=1;cycle_length=4"
python run_experiment.py --config ../config.yaml --override "pos_edges_ratio=0.7;min_train_embeddedness=1;cycle_length=4"
python run_experiment.py --config ../config.yaml --override "pos_edges_ratio=0.8;min_train_embeddedness=1;cycle_length=4"
python run_experiment.py --config ../config.yaml --override "pos_edges_ratio=0.9;min_train_embeddedness=1;cycle_length=4"

# 3. Generate all analysis plots
python analyze_all_results.py
python analyze_feature_distributions.py
```

### Alternative: Run Everything at Once
```bash
python run_comprehensive_experiments.py
```

## Core Pipeline Scripts

### Main Evaluation Pipeline
- **`validate_model.py`** : implements strict hold-out evaluation avoiding data leakage. For each test edge, features are extracted excluding the current edge.

- **`run_batch_experiments.py`**: Automated runner for 13 main experiments comparing weighted features, embeddedness levels, aggregation methods, and cycle lengths.

- **`run_experiment.py`**: Single experiment runner, primarily used for positive/negative ratio experiments with configurable parameters.

### Analysis & Visualization
- **`analyze_all_results.py`**: Generates 8 summary plots comparing all experiments:
  1. Weighted vs Unweighted comparison
  2. Dataset comparison (Bitcoin vs Epinions)
  3. Embeddedness filtering comparison
  4. Aggregation method comparison
  5. Performance summary table
  6. Positive ratio comparison
  7. Cycle length comparison
  8. Enhanced pos/neg ratio analysis

- **`analyze_feature_distributions.py`**: Creates 6 feature analysis plots:
  1. Individual feature distributions
  2. Embeddedness filtering impact
  3. Weighted vs unweighted features
  4. Feature statistics analysis
  5. Scaler comparison
  6. Class distribution analysis

### Core Processing
- **`preprocess.py`**: Data preprocessing including embeddedness filtering, connectivity assurance, and BFS-based fold creation
- **`train_model.py`**: Cross-validation training with logistic regression and feature scaling
- **`feature_extraction.py`**: Optimized higher-order cycle feature extraction (~100x speedup)

## Key Results

### Methodological Contribution
Our strict evaluation method addresses data leakage by:
1. BFS-based edge splitting for train/validation/test folds
2. For each test edge: remove edge → extract features → predict
3. Ensures no information leakage from test edges during feature extraction

### Comprehensive Experiment Results
After running all 18 experiments, results show:
- **Cycle Length**: 4-cycles generally outperform 3-cycles and 5-cycles
- **Embeddedness**: Moderate filtering (≥1 common neighbor) improves performance
- **Weighting**: Weighted features show marginal improvements over unweighted
- **Class Balance**: Performance relatively stable across different pos:neg ratios
- **Aggregation**: "max" bidirectional handling performs best

### Performance Metrics
All experiments evaluated on:
- **Accuracy**: Overall classification accuracy
- **F1 Score**: Harmonic mean of precision and recall
- **AUC**: Area under ROC curve
- **Precision**: Positive prediction accuracy

## Technical Details

### Key Features
- **No Data Leakage**: Strict BFS-based splitting and evaluation
- **Higher-Order Cycles**: Up to 5-cycle features with sparse matrix optimization
- **Embeddedness Filtering**: Common neighbor thresholding (0/1/2 levels)
- **Bidirectional Handling**: Multiple aggregation strategies (max/sum/stronger)
- **Class Imbalance**: Systematic evaluation across different pos:neg ratios
- **Comprehensive Analysis**: 14 visualization plots covering all aspects

### Optimizations
- **Feature Extraction**: 100x speedup through sparse matrix operations
- **Memory Efficiency**: Optimized graph operations and node reindexing
- **Reproducibility**: Complete configuration tracking and result serialization

## Output Structure
```
results/experiment_name/
├── preprocess/         # Preprocessed graphs and folds
├── training/          # Training fold metrics and models
├── testing/           # Final test metrics (metrics.json)
└── config_used.yaml   # Exact configuration used

plots/
├── complete_results_analysis/     # 8 summary comparison plots
└── enhanced_feature_analysis/     # 6 feature distribution plots
```

## References
- Chiang et al. (2014): "Exploiting Longer Cycles for Link Prediction in Signed Networks"
- Dataset: [Bitcoin OTC Web of Trust](https://snap.stanford.edu/data/soc-sign-bitcoinotc.html)