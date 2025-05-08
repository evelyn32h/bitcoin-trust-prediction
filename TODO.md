# TODO List

## Completed -Yingjia He
- [x] Project environment and structure setup
- [x] Dataset download and loading
- [x] Basic statistical analysis and visualization
- [x] Edge weight distribution visualization
- [x] Edge embeddedness distribution visualization
- [x] Part 1 Pre-process network
  - [x] Map weighted signed network to unweighted signed network
  - [x] Prune network e.g. ensure weak connectivity

## Next Steps 
- [ ] Part 1 Implementation of prediction algorithm
  - [ ] Extraction of features from network (according to Chiang et. al)
  - [ ] Implement logistic regression model (according to Chiang et. al)
- [ ] 
  - [ ] 
  - [ ] 
- [ ] 
  - [ ] 
  - [ ] 
  - [ ] 
- [ ] 
  - [ ] 
  - [ ] 
  - [ ] 

## Future Work (Part 2 & 3)
- [ ] Extend algorithm to incorporate edge weights in features
- [ ] Modify algorithm to predict both sign and weight of edges

## Notes and Observations
- Dataset is highly imbalanced (89% positive edges)
- Most positive ratings are concentrated around +1
- Negative ratings tend to be more extreme (mostly -10)
- These characteristics may impact model performance and require specific handling