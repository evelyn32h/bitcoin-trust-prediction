# TODO List

## Part 1: Edge Sign Prediction (Classic Method) ✓

### Yingjia He (Evelyn) - Completed ✓
- [x] Project environment and structure setup
- [x] Dataset download and loading
- [x] Basic statistical analysis and visualization
- [x] Edge weight distribution visualization
- [x] Edge embeddedness distribution visualization
- [x] Data preprocessing
  - [x] Map weighted signed network to unweighted signed network
  - [x] Filter neutral edges (weights close to 0)
  - [x] Ensure weak connectivity
  - [x] Node reindexing
- [x] Evaluation framework implementation
  - [x] Evaluation metrics (accuracy, AUC, false positive rate, precision, recall, F1)
  - [x] ROC curve visualization
  - [x] Confusion matrix visualization
  - [x] Cross-validation framework (10-fold)
- [x] Preprocessing optimization functions
  - [x] Edge filtering by embeddedness threshold
  - [x] Balanced dataset creation (equal positive/negative edges)
- [x] Experiment execution
  - [x] Cross-validation experiments with k=3 and k=4
  - [x] Performance comparison visualizations
  - [x] False positive rate analysis

### Requirements Rabo - Completed ✓
- [x] Feature extraction implementation
  - [x] Higher-order cycle features (k=3, 4, 5)
  - [x] Matrix-based feature computation
  - [x] Feature matrix construction
- [x] Model implementation
  - [x] Logistic regression classifier
  - [x] Model training pipeline
  - [x] Prediction functions
- [x] Integration with evaluation framework

### Wenbo Xia - In Progress
- [ ] Project report writing 


## Key Results and Findings

### Performance Metrics (10-fold Cross-validation)
- **k=3 Results**:
  - Accuracy: 92.75% ± 0.33%
  - AUC: 0.8514 ± 0.0123
  - False Positive Rate: 65.32% ± 2.52%
  - Precision: 93.18% ± 0.43%
  - Recall: 99.20% ± 0.18%
  - F1 Score: 96.10% ± 0.19%

- **k=4 Results**:
  - Accuracy: 97.10% ± 0.20%
  - AUC: 0.9669 ± 0.0104
  - False Positive Rate: 26.28% ± 1.84%
  - Precision: 97.15% ± 0.24%
  - Recall: 99.69% ± 0.11%
  - F1 Score: 98.41% ± 0.12%

### Key Improvements (k=3 → k=4)
- Accuracy: +4.69%
- AUC: +13.56%
- False Positive Rate: -59.77% (significant reduction)
- Precision: +4.26%
- Recall: +0.49%
- F1 Score: +2.40%

## Dataset Characteristics
- **Total edges**: 35,592
- **Nodes**: 5,881
- **Positive edges**: 89% (highly imbalanced)
- **Edge weight distribution**:
  - Positive ratings concentrated around +1
  - Negative ratings tend to be extreme (mostly -10)
- **Embeddedness**: ~30% of edges have 0 embeddedness (no shared neighbors)

## Files and Deliverables

### Code Files
- `src/data_loader.py`: Dataset loading utilities
- `src/preprocessing.py`: Graph preprocessing functions
- `src/feature_extraction.py`: Cycle-based feature extraction
- `src/models.py`: Machine learning models
- `src/evaluation.py`: Evaluation metrics and visualization
- `notebooks/cross_validation_eval.py`: Main experiment execution

### Result Files
- `results/roc_curve_k3.png`: ROC curve for k=3
- `results/roc_curve_k4.png`: ROC curve for k=4
- `results/confusion_matrix_k3.png`: Confusion matrix for k=3
- `results/confusion_matrix_k4.png`: Confusion matrix for k=4
- `results/comparison_k3_vs_k4.png`: Performance comparison
- `results/fpr_comparison.png`: False positive rate comparison
- `results/results_summary_table.png`: Comprehensive results table

## Future Work (Part 2 & 3)
- [ ] Part 2: Improve accuracy by using edge weights in input features
- [ ] Part 3: Extend algorithm to predict weight in addition to sign

## Notes for Report
- Emphasize the significant reduction in false positive rate with k=4
- Highlight the model's performance on highly imbalanced dataset
- Discuss the impact of embeddedness on prediction accuracy
- Compare results with Chiang et al.'s paper findings