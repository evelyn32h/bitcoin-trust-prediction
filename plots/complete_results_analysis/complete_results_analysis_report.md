# Complete Results Analysis Report

## Executive Summary

- **Total Experiments Analyzed**: 16
- **Best Overall Accuracy**: 0.972
- **Best Overall F1 Score**: 0.986
- **Best Overall ROC AUC**: 0.900

## Optimal Split Analysis (74:12:14)

- **Optimal Split Experiments**: 2
- **Standard Split Experiments**: 14
- **Best Optimal Split Accuracy**: 0.875
- **Average Optimal Split Accuracy**: 0.853

### Performance Improvements with Optimal Split:

- **Accuracy**: 0.679 ¡ú 0.853 (+25.7%)
- **F1_Score**: 0.711 ¡ú 0.917 (+29.0%)
- **Roc_Auc**: 0.595 ¡ú 0.892 (+49.9%)
- **Precision**: 0.723 ¡ú 0.863 (+19.4%)

## Feature Type Analysis

- **Binary (Sign)**: 8 experiments, max accuracy=0.913, avg accuracy=0.643
- **Unknown (raw)**: 6 experiments, max accuracy=0.943, avg accuracy=0.848
- **Weighted (Binned)**: 1 experiments, max accuracy=0.0, avg accuracy=0.0
- **Weighted (Raw)**: 1 experiments, max accuracy=0.972, avg accuracy=0.972

## Cycle Length Analysis (HOC)

- **HOC4**: 16 experiments, max accuracy=0.972, avg accuracy=0.7

## Embeddedness Level Analysis

- **Embeddedness ¡Ý0**: 16 experiments, max accuracy=0.972, avg accuracy=0.7

## Top Performing Experiments

1. **experiment_weighted_raw**: 0.972 accuracy, 0.986 F1, HOC4, Embed¡Ý0
2. **baseline_bitcoin_verification**: 0.943 accuracy, 0.969 F1, HOC4, Embed¡Ý0
3. **experiment_binary_optimized**: 0.913 accuracy, 0.951 F1, HOC4, Embed¡Ý0
4. **experiment_max**: 0.907 accuracy, 0.948 F1, HOC4, Embed¡Ý0
5. **experiment_epinions_aligned**: 0.893 accuracy, 0.943 F1, HOC4, Embed¡Ý0

## Generated Plots

- `optimal_split_comparison.png` - Comparison of optimal (74:12:14) vs standard split performance
- `weighted_vs_unweighted_performance.png` - Feature type comparison
- `cycle_length_comparison.png` - HOC3 vs HOC4 vs HOC5 performance
- `embeddedness_level_comparison.png` - Embeddedness filtering impact
- `positive_ratio_impact.png` - Effect of positive edge ratios
- `performance_summary_table.png` - Complete results table

---
*Report generated automatically by the results analysis pipeline*
