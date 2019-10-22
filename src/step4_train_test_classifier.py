#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train and test the Random Forest classifier

Created on September 18 2019
CRS4 - Center for Advanced Studies, Research and Development in Sardinia
@author: Jose F. Saenz-Cogollo
"""

import numpy as np
import pandas as pd
import pickle
from ecgtypes import BeatType
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

# Dataset source (input path)
dataset_path = "../datasets/"
# Dataset destination (output path)
output_data_path = "../datasets/heartbeatClassifier.pickle"
# Number of ranked features to consider
number_of_features = 6
# Number of trees for the Random Forest classifier
number_of_trees = 40

pickle_in = open(dataset_path + "train_set_features.pickle","rb")
data = pickle.load(pickle_in)
pickle_in.close()
train_set_labels = data['labels']
train_set_features = data['ranked_features'][:,:number_of_features]
train_set_sources = np.array(data['sources'])

pickle_in = open(dataset_path + "test_set_features.pickle","rb")
data = pickle.load(pickle_in)
pickle_in.close()
test_set_labels = data['labels']
test_set_features = data['ranked_features'][:,:number_of_features]
test_set_sources = np.array(data['sources'])

output_labels = [
    BeatType.NORMAL.symbol(), 
    BeatType.AURICULAR_PREMATURE_CONTRACTION.symbol(),
    BeatType.PREMATURE_VENTRICULAR_CONTRACTION.symbol(),
    BeatType.FUSION.symbol(),
    BeatType.UNKNOWN.symbol()
    ]

print('scaling data...')
scaler = StandardScaler()
train_set_tr = train_set_features#scaler.fit_transform(train_set_features)
#%%
print('training model...')
forest_clf = RandomForestClassifier(random_state=42, n_estimators=number_of_trees)
forest_clf.fit(train_set_tr, train_set_labels)
print(forest_clf.feature_importances_)

print('testing model...')

def leave_one_out_cv():
    set_sources = list(set(train_set_sources))
    for s in set_sources:
        test_indexes = []
        train_indexes = []
        for i, ts in enumerate(train_set_sources):
            if ts == s:
                test_indexes.append(i)
            else:
                train_indexes.append(i)
        yield train_indexes, test_indexes

train_predictions = cross_val_predict(forest_clf, train_set_tr, train_set_labels, cv=leave_one_out_cv())
acc_train = accuracy_score(train_set_labels, train_predictions)
print(f'Train accuracy with {number_of_features} features, and {number_of_trees} trees:')
print(acc_train)
#%%
print('calculating quality parameters...')
def evaluate_classifier(conf_mx, outputs):
    qualityMeasures = ['Se', 'Sp', 'Pp', 'FPR', 'Ac', 'F1'] 
    Q = np.empty((len(qualityMeasures), len(outputs)))
    for k, label in enumerate(outputs):
        tp = conf_mx[k,k]
        fn = np.sum(conf_mx[k,:]) - tp
        tn = np.sum(conf_mx) - np.sum(conf_mx[k,:])
        fp = np.sum(conf_mx[:,k]) - tp 
        Q[0, k] = tp / (tp + fn)
        Q[1, k] = tn / (tn + fp)
        Q[2, k] = tp / (tp + fp)
        Q[3, k] = fp / (tp + fn)
        Q[4, k] = (tp + tn) / (tp + tn + fp + fn)
        Q[5, k] = 2 * (Q[2, k] * Q[0, k]) / (Q[2, k] + Q[0, k])
    
    return pd.DataFrame(Q, columns=outputs, index=qualityMeasures)

#%%
print('validating model...')
test_predict = forest_clf.predict(test_set_features)#scaler.fit_transform(test_set))
conf_mx_test = confusion_matrix(test_set_labels, test_predict)
print('conf_mx_test:')
print(conf_mx_test)
acc_test = accuracy_score(test_set_labels, test_predict)
print(f'Test accuracy with {number_of_features} features, and {number_of_trees} trees:')
print(acc_test)
Evaluation_test = evaluate_classifier(conf_mx_test, output_labels)
print(f'Evaluation details with {number_of_features} features, and {number_of_trees} trees:')
print(Evaluation_test)

print(f'Classification report with {number_of_features} features, and {number_of_trees} trees:')
print(classification_report(test_set_labels, test_predict, target_names=output_labels, digits=4))

# print('Classification details by sources:')
# train_pred_counts = count_labels_by_source(train_predictions, train_set_sources)
# test_set['features']_counts = count_labels_by_source(test_set['features'], test_sources)
# print(train_pred_counts)

# test_set['features']_counts = count_labels_by_source(test_set_labels, test_sources, test_set['features'])
# print(test_set['features']_counts)

#%%
print('saving model...')

pickle_out = open(output_data_path,"wb")
pickle.dump({'preprocessor': None, 'model' : forest_clf, 'Evaluation_test': Evaluation_test}, pickle_out)
pickle_out.close()
