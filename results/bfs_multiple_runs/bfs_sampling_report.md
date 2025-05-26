# Multiple BFS Sampling Analysis Report

Generated in response to Vide's request for better edge balance

## Objective
Find Epinions subsets closer to original 85% positive / 15% negative distribution.
Current BFS result: 93% positive / 7% negative

## Results Summary
- Total attempts: 6
- Successful runs: 6
- Failed runs: 0

## Best Result
- **Run #5**
- Positive ratio: 85.4%
- Distance from 85% target: 0.4%
- Strategy: random_high_degree, percentile 90
- Network: 9,284 nodes, 35,000 edges

## Recommendation
✅ **Found better balanced subsets!**

Use Run #5 for 7.6% better balance:
- New ratio: 85.4% positive
- Saved as: `data/bfs_subsets/epinions_bfs_run5_*`

### Next Steps
1. Update config.yaml to use the improved subset
2. Re-run experiments with better balanced data
3. Compare results in final report

## Files Generated
- `bfs_runs_comparison.csv` - Comparison table
- `bfs_runs_detailed.json` - Detailed results
- `data/bfs_subsets/` - Promising subset files
