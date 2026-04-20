# DetectUA

This repository contains experiment scripts and results for the DetectUA model. Experiments with Random Forest, Support Vector Machine, and RobBERT have been conducted, using ```data/detectua_data_combined_no_social_media.csv```. The (cross-dataset) results of the final model can be found in ```/results/svm_experiments/svm_final```. The model itself is stored in the repository of the DetectUA app.

Three main types of experiments have been conducted: *baseline* experiments refer to settings where all data was distributed across the training and test sets (stratified by label frequency. In *cross-dataset* each dataset is iteratively used as a held-out test set, whereas the others are all used for training and (cross-)validation. Finally, *cross-genre* is similar to cross-dataset, but the focus in on testing on specific genres rather than datasets.
