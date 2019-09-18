import numpy as np
import pickle
from ecgtypes import BeatType
from inter_patient_division import create_train_test_sets
from sklearn.feature_selection import mutual_info_classif

# Dataset source (input path)
input_dataset_path = "/Users/josefranciscosaenzcogollo/crs4/ecg-heartbeat-classification//datasets/features.pickle"
# Dataset destination (output path)
output_dataset_path = "../datasets/ranked_features.pickle"

pickle_in = open(input_dataset_path, "rb")
data = pickle.load(pickle_in)
pickle_in.close()

beats = data["beats"]

feature_names = list(beats[0]["rr"].keys())
feature_names += list(beats[0]["morph"].keys())
feature_names += ["wt_coef_" + str(i) for i, x in enumerate(beats[0]["wt"])]
feature_names += ["hos_" + str(i) for i, x in enumerate(beats[0]["hos"])]
feature_names += ["mg_" + str(i) for i, x in enumerate(beats[0]["mg"])]
feature_names += ["hbf_" + str(i) for i, x in enumerate(beats[0]["hbf"])]
feature_names += ["lbp_" + str(i) for i, x in enumerate(beats[0]["lbp"])]

feature_names = np.array(feature_names)

all_features = np.empty((len(beats), len(feature_names)))

labels = np.empty((len(beats)), dtype=int)
sources = list(range(len(beats)))
print("constructing features vector...")
for beatIndex, beat in enumerate(beats):
    labels[beatIndex] = beat["beatType"].value
    sources[beatIndex] = beat["source"]
    features = list(beat["rr"].values())
    features += list(beat["morph"].values())
    features += list(beat["wt"])
    features += list(beat["hos"])
    features += list(beat["mg"])
    features += list(beat["hbf"])
    features += list(beat["lbp"])

    all_features[beatIndex] = np.array(features)

print("Obtaining training set...")
train_set, test_set = create_train_test_sets(all_features, labels, sources)

train_features = train_set["features"]
train_labels = train_set["labels"]

print("Estimating the most informative features for S and V...")
labels_L = train_labels.tolist()
N_ind = np.array([k for k, l in enumerate(labels_L) if l == BeatType.NORMAL.value], dtype=int)
S_ind = np.array([k for k, l in enumerate(labels_L) if l == BeatType.AURICULAR_PREMATURE_CONTRACTION.value], dtype=int)
V_ind = np.array([k for k, l in enumerate(labels_L) if l == BeatType.PREMATURE_VENTRICULAR_CONTRACTION.value], dtype=int)
indixes = np.block([N_ind, S_ind, V_ind])
labels_SV = train_labels[indixes]
features_SV = train_features[indixes]

mi_features = mutual_info_classif(features_SV, labels_SV, random_state=42)
# sorted and ordered as higher first
mi_rank = np.argsort(mi_features)[-1:0:-1]
ranked_features = all_features[:, mi_rank]
ranked_features_names = feature_names[mi_rank]
print("MI ranked features: " + str(ranked_features_names))
print("MI of ranked features: " + str(mi_features[mi_rank]))


print("saving dataset file...")
pickle_out = open(output_dataset_path, "wb")
pickle.dump(
    {
        "features": all_features,
        "feature_names": feature_names,
        "labels": labels,
        "sources": sources,
        "ranked_features": ranked_features,
        "ranked_features_names": ranked_features_names,
    },
    pickle_out,
)
pickle_out.close()
