from array import array
import statistics as stats

class SignalBuffer:
    """ Represent a buffer with the last x samples of a signal """
    def __init__(self, size, initialValue=None):
        self.size = size
        self.empty = False
        if initialValue == None:
            initialValue = 0
            self.empty = True
        self.array = array('f', [initialValue for i in range(self.size * 2)])
        self.index = self.size
        self.sum = 0
    def push(self, x):
        self.sum -= self.array[self.index]
        self.array[self.index] = x
        self.array[self.index - self.size] =  x
        self.sum += x
        self.index += 1
        if self.index >= len(self.array):
            self.index = self.size  
            self.empty = False      
    def getBuffer(self):
        return self.array[self.index - self.size : self.index]
    def mean(self):
        if self.empty:
            if self.index == self.size:
                return 0
            return self.sum / (self.index - self.size)
        return self.sum / self.size

    def std(self):
        if self.empty:
            if self.index <= self.size + 1:
                return 0
            return stats.stdev(self.array[0:self.index - self.size])
        return stats.stdev(self.array[0:self.size])
    def partialMean(self, samples):
        if samples <= self.size:
            if self.empty and (samples > self.index - self.size):
                if self.index == self.size:
                    return 0
                samples = self.index - self.size
            partialBuffer = self.getBuffer()[self.size - samples:self.size]
            return stats.mean(partialBuffer)
        else:
            return self.mean()


def safe_normalizer(x, x_mean):
    if x == x_mean and x != 0:
        return 1
    elif x_mean == 0:
        return x
    else:
        return x / x_mean
