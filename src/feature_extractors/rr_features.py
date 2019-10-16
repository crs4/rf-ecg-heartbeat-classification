from .common import SignalBuffer
from .common import safe_normalizer
import statistics as stats


class RRFeatures():
    """ Caculates fetures from RR intervals """

    def __init__(self):
        self.rrBuffer = SignalBuffer(32)

    def __call__(self, beats, beat_index):
        labeledBeatTime = beats[beat_index]['time']
        if beat_index < 2:
            prevLabeledBeatTime = 0
        else:
            prevLabeledBeatTime = beats[beat_index - 2]['time']
        if beat_index == 0:
            lastLabeledBeatTime = 0
        else:
            lastLabeledBeatTime = beats[beat_index - 1]['time']
        if beat_index + 1 == len(beats):
            nextLabeledBeatTime = beats[-1]['time']
        else:
            nextLabeledBeatTime = beats[beat_index + 1]['time']
        if beat_index < 32:
            startbeat_index = 0
        else:
            startbeat_index = beat_index - 32
        rrBuffer = []
        for k in range(startbeat_index, beat_index):
            rrBuffer.append(beats[k + 1]
                            ['time'] - beats[k]['time'])

        previousRR = lastLabeledBeatTime - prevLabeledBeatTime
        currentRR = labeledBeatTime - lastLabeledBeatTime
        nextRR = nextLabeledBeatTime - labeledBeatTime
        self.rrBuffer.push(currentRR)
        averageRR = self.rrBuffer.mean()
        stddevRR = self.rrBuffer.std()
        if stddevRR == 0:
            student = 0
        else:
            student = (currentRR - averageRR) / stddevRR
        return {
            'RR0': currentRR,
            'RR-1': previousRR,
            'RR+1': nextRR,
            'RR0/avgRR': safe_normalizer(currentRR, averageRR),
            'tRR0': student,
            'RR-1/avgRR': safe_normalizer(previousRR, averageRR),
            'RR-1/RR0': safe_normalizer(previousRR, currentRR),
            'RR+1/avgRR': safe_normalizer(nextRR, averageRR),
            'RR+1/RR0': safe_normalizer(nextRR, currentRR),
        }
