
import numpy as np
import pandas as pd
import pickle
from ecgtypes import BeatType
from inter_patient_division import create_train_test_sets
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

# Dataset source (input path)
input_dataset_path = "./datasets/ranked_features.pickle"
# Dataset destination (output path)
output_data_path = "./datasets/heartbeatClassifier.pickle"

pickle_in = open(input_dataset_path,"rb")
data = pickle.load(pickle_in)
pickle_in.close()
labels = data['labels']
features = data['ranked_features'][:6]
sources = np.array(data['sources'])

output_labels = [
    BeatType.NORMAL.symbol, 
    BeatType.AURICULAR_PREMATURE_CONTRACTION.symbol,
    BeatType.PREMATURE_VENTRICULAR_CONTRACTION.symbol,
    BeatType.UNKNOWN.symbol,
    BeatType.FUSION.symbol
    ]

labels_L = labels.tolist()
# # Normalizing normal beat quantity
N_ind = np.array([k for k, l in enumerate(labels_L) if l == BeatType.NORMAL.value], dtype=int)
S_ind = np.array([k for k, l in enumerate(labels_L) if l == BeatType.AURICULAR_PREMATURE_CONTRACTION.value], dtype=int)
V_ind = np.array([k for k, l in enumerate(labels_L) if l == BeatType.PREMATURE_VENTRICULAR_CONTRACTION.value], dtype=int)
Q_ind = np.array([k for k, l in enumerate(labels_L) if l == BeatType.UNKNOWN.value], dtype=int)
R_ind = np.array([k for k, l in enumerate(labels_L) if l == BeatType.OTHER.value], dtype=int)
F_ind = np.array([k for k, l in enumerate(labels_L) if l == BeatType.FUSION.value], dtype=int)
#import random
#N_rnd_ind = np.array(random.choices(N_ind, k=cV*2))
indixes = np.block([N_ind, S_ind, V_ind, Q_ind, F_ind])
labels = labels[indixes]
features = features[indixes]
sources = np.array(sources)[indixes]

print("Obtaining training set...")
train_set, test_set = create_train_test_sets(features, labels, sources)

train_features = train_set["features"]
train_set["labels"] = train_set["labels"]

print('scaling data...')
scaler = StandardScaler()
train_set_tr = train_set["features"]#scaler.fit_transform(train_set["features"])
#%%
print('training model...')
forest_clf = RandomForestClassifier(random_state=42, n_estimators=30)
forest_clf.fit(train_set_tr, train_set["labels"])
print(forest_clf.feature_importances_)

print('testing model...')

def leave_one_out_cv():
    set_sources = list(set(train_set['sources']))
    for s in set_sources:
        test_indexes = []
        train_indexes = []
        for i, ts in enumerate(train_set['sources']):
            if ts == s:
                test_indexes.append(i)
            else:
                train_indexes.append(i)
        yield train_indexes, test_indexes

train_predictions = cross_val_predict(forest_clf, train_set_tr, train_set["labels"], cv=leave_one_out_cv())
conf_mx_train = confusion_matrix(
    train_set["labels"], train_predictions)
print('conf_mx_train:')
print(conf_mx_train)
acc_train = accuracy_score(train_set["labels"], train_predictions)
print('Train accuracy:')
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


Evaluation_train = evaluate_classifier(conf_mx_train, output_labels)
print(Evaluation_train)
print(classification_report(train_set["labels"], train_predictions,
                            target_names=output_labels, digits=4))
#%%
print('validating model...')
test_set['features'] = forest_clf.predict(test_set)#scaler.fit_transform(test_set))
conf_mx_test = confusion_matrix(test_set['labels'], test_set['features'])
print('conf_mx_test:')
print(conf_mx_test)
acc_test = accuracy_score(test_set['labels'], test_set['features'])
print('Test accuracy:')
print(acc_test)
Evaluation_test = evaluate_classifier(conf_mx_test, output_labels)
print(Evaluation_test)

print(classification_report(test_set['labels'], test_set['features'], target_names=output_labels, digits=4))

print('DETAILS:')
# train_pred_counts = count_labels_by_source(train_predictions, train_set['sources'])
# test_set['features']_counts = count_labels_by_source(test_set['features'], test_sources)
# print(train_pred_counts)

# test_set['features']_counts = count_labels_by_source(test_set['labels'], test_sources, test_set['features'])
# print(test_set['features']_counts)
#%%

#%%
print('saving model...')

pickle_out = open(output_data_path,"wb")
pickle.dump({'preprocessor': None, 'model' : forest_clf, 'Evaluation_train'  : Evaluation_train , 'Evaluation_test': Evaluation_test}, pickle_out)
pickle_out.close()
