#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read MIT-BIH Arrhythmia Database files and store signals, labels and record names in a pickle file for easy use

@author: Jose F. Saenz"""

import os
import wfdb
from scipy import signal as signal
import numpy as np
import pickle
import sys
import ecgtypes

# Database location (input path)
db_path = "/Users/josefranciscosaenzcogollo/crs4/databases/mitdb/"
# Dataset destination (output path)
dataset_path = "../datasets/mitdb.pickle"

# Find available records
files = os.listdir(db_path)
records = []
for k in range(len(files)):
    filename = files[k].split(".")[0]
    file_extension = files[k].split(".")[1]
    if file_extension == "dat":
        records.append(files[k].split(".")[0])
try:
    records.remove("")
except ValueError:
    pass
records = [x for x in set(records)]

# Prepare containers
dataset, labels = [], []
# Iterate files
for record_name in records:
    print("*** Reading record: " + record_name)
    channel = 0
    record = wfdb.rdrecord(db_path + record_name)
    annotations = wfdb.rdann(db_path + record_name, "atr")
    data = record.p_signal[:, channel]
    header = {
        "label": record.sig_name[channel],
        "dimension": record.units[channel],
        "sample_rate": record.fs,
        "digital_max": (2 ** record.adc_res[channel]) - 1,
        "digital_min": 0,
        "transducer": "transducer type not recorded",
        "prefilter": "prefiltering not recorded",
    }
    header["physical_max"] = (
        header["digital_max"] - record.baseline[channel]
    ) / record.adc_gain[channel]
    header["physical_min"] = (
        header["digital_min"] - record.baseline[channel]
    ) / record.adc_gain[channel]
    # print(header)
    xo = data
    fs = 150
    duration = len(xo) / record.fs
    print("resampling signal...")
    xr = signal.resample(xo, round(duration) * fs)
    print("reading annotations...")
    rhythmClass = ecgtypes.HeartRhythm.NORMAL
    label = []
    for s in range(len(annotations.sample)):
        t = annotations.sample[s] / record.fs
        ann = annotations.symbol[s]

        if len(annotations.aux_note[s]) > 0:
            if annotations.aux_note[s][0] == "(":
                rhythmClass = annotations.aux_note[s].strip("\x00")[1:]

        if len(ann) == 0:
            continue
        elif ann:
            label.append(
                {
                    "time": t,
                    "beat": ecgtypes.BeatType.new_from_symbol(ann),
                    "rhythm": ecgtypes.HeartRhythm.new_from_symbol(rhythmClass),
                }
            )

    # Cumulate
    dataset.append(xr)
    labels.append(label)

# Write dataset to disk
print("saving dataset file...")
pickle_out = open(dataset_path, "wb")
pickle.dump({"signals": dataset, "labels": labels, "records": records}, pickle_out)
pickle_out.close()
