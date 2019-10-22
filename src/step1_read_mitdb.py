#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create test and train sets by reading MIT-BIH Arrhythmia Database according to the literature defined inter-patient paradigm.
Read database from the online public directory, resample signals, and store data in pickle files for easy use

Created on September 18 2019
CRS4 - Center for Advanced Studies, Research and Development in Sardinia
@author: Jose F. Saenz-Cogollo
"""

import os
import wfdb
from scipy import signal as signal
import numpy as np
import pickle
import sys
import ecgtypes

# Database online public directory (input path)
db_name = "mitdb"
# Database records for training (DS1) and testing (DS2)
DS1 = {
    "101",
    "106",
    "108",
    "109",
    "112",
    "114",
    "115",
    "116",
    "118",
    "119",
    "122",
    "124",
    "201",
    "203",
    "205",
    "207",
    "208",
    "209",
    "215",
    "220",
    "223",
    "230",
}
DS2 = {
    "100",
    "103",
    "105",
    "111",
    "113",
    "117",
    "121",
    "123",
    "200",
    "202",
    "210",
    "212",
    "213",
    "214",
    "219",
    "221",
    "222",
    "228",
    "231",
    "232",
    "233",
    "234",
}
# Dataset destination (output path)
dataset_path = "../datasets/"


def get_database_data(db, records):
    """ 
    Read signals and labels from database records in the public directory  
    """
    # Prepare containers
    signals, labels = [], []
    # Iterate files
    for record_name in records:
        print("*** Reading record: " + record_name)
        channel = 0
        record = wfdb.rdrecord(record_name, pb_dir=db)
        annotations = wfdb.rdann(record_name, "atr", pb_dir=db)
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
        duration = len(data) / record.fs
        # add zeros to the end in order to achieve a duration of an integer number of seconds that facicilatetes resampling
        new_length = int(np.ceil(duration) * record.fs)
        xo = np.zeros(new_length)
        xo[0 : len(data)] = data
        fs = 150
        print("resampling signal...")
        xr = signal.resample(xo, int(np.ceil(duration) * fs))
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
        signals.append(xr)
        labels.append(label)
    return signals, labels


print("Creating training set ...")
signals, labels = get_database_data(db_name, DS1)
print("saving train_set file...")
pickle_out = open(dataset_path + "train_set_signals.pickle", "wb")
pickle.dump({"signals": signals, "labels": labels, "records": DS1}, pickle_out)
pickle_out.close()
print("Creating test set ...")
signals, labels = get_database_data(db_name, DS2)
print("saving test_set file...")
pickle_out = open(dataset_path + "test_set_signals.pickle", "wb")
pickle.dump({"signals": signals, "labels": labels, "records": DS2}, pickle_out)
pickle_out.close()
