#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Construct features vectors, calculates the mutual information of features relative to S and V labels, and rank features accordingly

Created on September 18 2019
CRS4 - Center for Advanced Studies, Research and Development in Sardinia
@author: Jose F. Saenz-Cogollo
"""

import numpy as np
import pickle
from ecgtypes import BeatType
from sklearn.feature_selection import mutual_info_classif

# Dataset source (input path)
dataset_path = "../datasets/"


def construct_vectors(beats):
    features_names = list(beats[0]["rr"].keys())
    features_names += list(beats[0]["morph"].keys())
    features_names += ["wt_coef_" + str(i) for i, x in enumerate(beats[0]["wt"])]
    features_names += ["hos_" + str(i) for i, x in enumerate(beats[0]["hos"])]
    features_names += ["mg_" + str(i) for i, x in enumerate(beats[0]["mg"])]
    features_names += ["hbf_" + str(i) for i, x in enumerate(beats[0]["hbf"])]
    features_names += ["lbp_" + str(i) for i, x in enumerate(beats[0]["lbp"])]

    features_names = np.array(features_names)

    features_vector = np.empty((len(beats), len(features_names)))

    labels = np.empty((len(beats)), dtype=int)
    sources = list(range(len(beats)))

    for beatIndex, beat in enumerate(beats):
        labels[beatIndex] = beat["beatType"].value
        sources[beatIndex] = beat["source"]
        beat_features = list(beat["rr"].values())
        beat_features += list(beat["morph"].values())
        beat_features += list(beat["wt"])
        beat_features += list(beat["hos"])
        beat_features += list(beat["mg"])
        beat_features += list(beat["hbf"])
        beat_features += list(beat["lbp"])

        features_vector[beatIndex] = np.array(beat_features)
    return features_vector, features_names, labels, sources


print("constructing train set features vector...")
pickle_in = open(dataset_path + "train_set_beats.pickle", "rb")
data = pickle.load(pickle_in)
pickle_in.close()
train_features, train_features_names, train_labels, train_sources = construct_vectors(
    data["beats"]
)
print("Estimating the most informative features for S and V from training set...")
labels_L = train_labels.tolist()
N_ind = np.array(
    [k for k, l in enumerate(labels_L) if l == BeatType.NORMAL.value], dtype=int
)
S_ind = np.array(
    [
        k
        for k, l in enumerate(labels_L)
        if l == BeatType.AURICULAR_PREMATURE_CONTRACTION.value
    ],
    dtype=int,
)
V_ind = np.array(
    [
        k
        for k, l in enumerate(labels_L)
        if l == BeatType.PREMATURE_VENTRICULAR_CONTRACTION.value
    ],
    dtype=int,
)
indixes = np.block([N_ind, S_ind, V_ind])
labels_SV = train_labels[indixes]
features_SV = train_features[indixes]

mi_features = mutual_info_classif(features_SV, labels_SV, random_state=42)
# sorted and ordered as higher first
mi_rank = np.argsort(mi_features)[-1:0:-1]

ranked_features_names = train_features_names[mi_rank]
print("MI ranked features: " + str(ranked_features_names))
print("MI of ranked features: " + str(mi_features[mi_rank]))
print("saving features rank file...")
pickle_out = open(dataset_path + "train_set_mi_rank.pickle", "wb")
pickle.dump(
    {
        "ranked_features_names": ranked_features_names,
        "mi_ranked_features": mi_features[mi_rank],
        "mi_rank": mi_rank,
    },
    pickle_out,
)
pickle_out.close()

print("saving train set file...")
pickle_out = open(dataset_path + "train_set_features.pickle", "wb")
pickle.dump(
    {
        "features": train_features,
        "features_names": train_features_names,
        "labels": train_labels,
        "sources": train_sources,
        "ranked_features": train_features[:, mi_rank],
        "ranked_features_names": ranked_features_names,
    },
    pickle_out,
)
pickle_out.close()

print("constructing test set features vector...")
pickle_in = open(dataset_path + "test_set_beats.pickle", "rb")
data = pickle.load(pickle_in)
pickle_in.close()
test_features, test_features_names, test_labels, test_sources = construct_vectors(
    data["beats"]
)

print("saving train set file...")
pickle_out = open(dataset_path + "test_set_features.pickle", "wb")
pickle.dump(
    {
        "features": test_features,
        "features_names": test_features_names,
        "labels": test_labels,
        "sources": test_sources,
        "ranked_features": test_features[:, mi_rank],
        "ranked_features_names": ranked_features_names,
    },
    pickle_out,
)
pickle_out.close()

