# Comprehensive Feature Analysis Report

Generated based on Requirements's specific requirements from chat logs

## Analysis Completed

DONE 1. Overall Feature Value Distribution
DONE 2. Feature Value Ranges by Dataset (Box plots)
DONE 3. Feature-wise Statistics (Mean vs Std)
DONE 4. Class Distribution by Dataset
DONE 5. Feature distribution with/without embeddedness filtering (MAIN)
DONE 6. Enhanced scaler comparison with skewness and kurtosis
DONE 7. Weighted vs unweighted feature analysis

## Key Findings

### Embeddedness Filtering Impact (Main Analysis)

- **Edge reduction**: 31.5%
- **Impact**: Significant change in feature distributions
- **Conclusion**: Embeddedness filtering substantially alters feature characteristics

### Scaler Recommendation

- **Recommended scaler**: MinMaxScaler
- **Basis**: Lowest combined skewness + kurtosis score
- **Benefits**: Best feature distribution normalization

### Why Weighted Features Failed

- **Variance ratio**: 34746.77x higher for weighted features
- **Problem**: High variance leads to overfitting
- **Solution**: Binary (±1) features are more robust
- **Conclusion**: Simple sign-based features work better for HOC

## Files Generated

- `requirements_requested_feature_plots.png` - The 4 specific plots Requirements requested
- `embeddedness_filtering_comparison.png` - Main analysis Requirements wanted
- `enhanced_scaler_comparison.png` - With skewness/kurtosis as requested
- `weighted_vs_unweighted_comparison.png` - Explains weighted failure
- This comprehensive report

## Recommendations for Requirements's Presentation

1. **Use the embeddedness filtering comparison** - your main requested analysis
2. **Show the enhanced scaler comparison** - with skewness/kurtosis
3. **Include weighted vs unweighted analysis** - explains why weighted failed
4. **Reference the 4 core plots** - exactly what you listed in chat
5. **All plots are high-resolution** - ready for presentation

