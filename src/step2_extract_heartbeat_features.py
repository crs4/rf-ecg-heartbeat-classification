#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 15:09:22 2018

@author: Jose F. Saenz"""

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
input_dataset_path = '../datasets/mitdb.pickle'
# input_dataset_path = '../../ecg-analyzer/modelDevelopment/datasets/mitdb.pickle'
DEBUG = True
DEBUG_RECORD = '207'
DEBUG_BEAT = 0
# Dataset destination (output path)
output_dataset_path = '../datasets/features.pickle'

pickle_in = open(input_dataset_path, "rb")
data = pickle.load(pickle_in)
pickle_in.close()
signals = data['signals']
labels = data['labels']
records = data['records']

beat_counters = [0, 0, 0, 0, 0, 0]
for i, beats in enumerate(labels):
    for j, label in enumerate(beats):
        # beat = BeatType.new_from_symbol(label['beat'])
        # labels[i][j]['beat'] = beat
        beat = label['beat']
        beat_counters[beat.value] += 1
print(beat_counters)
timer = process_time()
def tic():
    global timer
    timer = process_time()

def toc(reset=False):
    global timer
    t = process_time() - timer
    if reset:
        tic()
    return t

def get_qrs_waveform(beatTime, signal, window=180):
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

beats = []
missed_beats = 0
true_beats = 0
detected_beats = 0
false_beats = 0
false_beats_records = []
missed_labels = []
morph_features = ExtractQRS()
rr_features = RRFeatures()
for recordIndex, recordName in enumerate(records):
    resultIndex = 0
    fp_record = 0
    if DEBUG and recordName != DEBUG_RECORD:
        continue
    print(f'Processing record {recordName} ({recordIndex} of {len(records)})')
    for labelIndex, label in enumerate(labels[recordIndex]):
        # if label['time'] < 1 or label['time'] > labels[recordIndex][len(labels[recordIndex]) - 1]['time'] - 1:
        #     continue
        labeledBeatTime = label['time']
        labeledBeat = label['beat']
        # ignore noise and label artifacts
        if labeledBeat != BeatType.OTHER:
            true_beats += 1
        else:
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
            'beatType': labeledBeat.value,
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
        detected_beats += 1


print('True beats: ' + str(true_beats))
print('saving dataset file...')
pickle_out = open(output_dataset_path, "wb")
pickle.dump({'beats': beats}, pickle_out)
pickle_out.close()
