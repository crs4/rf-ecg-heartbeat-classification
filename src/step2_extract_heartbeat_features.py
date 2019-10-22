#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract heatbeat features from ECG signals

Created on September 18 2019
CRS4 - Center for Advanced Studies, Research and Development in Sardinia
@author: Jose F. Saenz-Cogollo
"""

import numpy as np
import pandas as pd
import pickle
import sys
from ecgtypes import BeatType
from feature_extractors.rr_features import RRFeatures
from feature_extractors.pqrs_features import ExtractQRS
from feature_extractors.modejarguerra_features import compute_wavelet_descriptor as wt_features
from feature_extractors.modejarguerra_features import compute_hos_descriptor as hos_features
from feature_extractors.modejarguerra_features import compute_my_own_descriptor as mg_features
from feature_extractors.modejarguerra_features import compute_HBF as hbf_features
from feature_extractors.modejarguerra_features import compute_Uniform_LBP as lbp_features
from time import process_time 
import matplotlib.pyplot as plt

# Dataset destination (output path)
dataset_path = '../datasets/'
# Debugging options
DEBUG = False
DEBUG_RECORD = '207'
DEBUG_BEAT = 0
# Dataset destination (output path)
dataset_path = '../datasets/'
# Timer for measuring the execution time of feature extractors
timer = process_time()
def tic():
    """
    Reset timer
    """
    global timer
    timer = process_time()

def toc(reset=False):
    """
    Get elapsed time
    """
    global timer
    t = process_time() - timer
    if reset:
        tic()
    return t

def get_qrs_waveform(beatTime, signal, window=180):
    """
    Extract a segment of signal around the beat time (R spike time)
    """
    beatSample = int(beatTime * 150)
    qrsWaveform = np.zeros(window)
    k = int(window / 2)
    for n in range(beatSample, -1, -1):
        if k >= 0:
            qrsWaveform[k] = signal[n]
        else: 
            break
        k -= 1
    k = int(window / 2 + 1)
    for n in range(beatSample + 1, len(signal)):
        if k < window:
            qrsWaveform[k] = signal[n]
        else:
            break
        k += 1
    return qrsWaveform

def extract_beat_features(signals, labels, records):
    """
    Extract features from all labeled heartbeats in a set of records
    """
    beats = []
    morph_features = ExtractQRS()
    rr_features = RRFeatures()
    for recordIndex, recordName in enumerate(records):
        if DEBUG and recordName != DEBUG_RECORD:
            continue
        print(f'Processing record {recordName} ({recordIndex} of {len(records)})')
        for labelIndex, label in enumerate(labels[recordIndex]):
            labeledBeatTime = label['time']
            labeledBeat = label['beat']
            # ignore noise and label artifacts
            if labeledBeat == BeatType.OTHER:
                continue
            tic()
            rr = rr_features(labels[recordIndex], labelIndex)
            rr_time = toc(True)
            morph = morph_features(labeledBeatTime, signals[recordIndex])
            morph_time = toc(True)        
            qrsWaveform = get_qrs_waveform(labeledBeatTime, signals[recordIndex], 76)
            if DEBUG and labelIndex >= DEBUG_BEAT:
                plt.plot(signals[recordIndex])
                plt.title(labeledBeat.symbol())
                plt.show()
                pass
            wt = wt_features(qrsWaveform)
            wt_time = toc(True)
            hos = hos_features(qrsWaveform)
            hos_time = toc(True)
            mg = mg_features(qrsWaveform)
            mg_time = toc(True)
            hbf = hbf_features(qrsWaveform)
            hbf_time = toc(True)
            lbp = lbp_features(qrsWaveform)
            lbp_time = toc(True)
            beat = {
                'beatType': labeledBeat,
                'source': recordName,
                'rr': rr,
                'morph': morph,
                'wt': wt,
                'hos': hos,
                'rr_time': rr_time,
                'morph_time': morph_time,
                'wt_time': wt_time,
                'hos_time': hos_time,
                'mg': mg,
                'mg_time': mg_time,
                'hbf': hbf,
                'hbf_time': hbf_time,
                'lbp': lbp,
                'lbp_time': lbp_time
            }
            beats.append(beat)
    return beats

print('Extracting train set heartbeats features...')
pickle_in = open(dataset_path + 'train_set_signals.pickle', "rb")
data = pickle.load(pickle_in)
pickle_in.close()
beats = extract_beat_features(data['signals'], data['labels'], data['records'])
print('saving train_set file...')
pickle_out = open(dataset_path + 'train_set_beats.pickle', "wb")
pickle.dump({'beats': beats}, pickle_out)
pickle_out.close()

print('Extracting test set heartbeats features...')
pickle_in = open(dataset_path + 'test_set_signals.pickle', "rb")
data = pickle.load(pickle_in)
pickle_in.close()
beats = extract_beat_features(data['signals'], data['labels'], data['records'])
print('saving test_set file...')
pickle_out = open(dataset_path + 'test_set_beats.pickle', "wb")
pickle.dump({'beats': beats}, pickle_out)
pickle_out.close()
