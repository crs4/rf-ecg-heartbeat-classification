#  ECG heartbeat classification using mutual information-based feature selection and Random Forests
This repo contains the code used for investigating the use of Random Forests for classifiying selected heartbeats features.
Features were selected using a filter method based on the mutual information ranking criterion on the training set.

# Requirements
The following Python libraires are required to execute the code: 

* wfdb
* numpy
* PyWavelets
* scipy
* sklearn
* pandas
* matplotlib (optional)

# How to run the code
Execute the Python files in the _/src_ folder in the order suggested by their names:

1. _step1_read_mitdb.py_: Create test and train sets by reading MIT-BIH Arrhythmia Database according to the literature defined inter-patient paradigm.
2. _step2_extract_heartbeat_features.py_: Extract heatbeat features from ECG signals using the extractors defined in the _/src/feature_extractors_ folder
3. _step3_rank_features.py_: Construct features vectors, calculates the mutual information of features relative to S and V labels, and rank features accordingly.
4. _step4_train_test_classifier.py_: Train and test the Random Forest classifier.

# License
[GNU GPLv3 License](https://www.gnu.org/licenses/gpl-3.0.html)