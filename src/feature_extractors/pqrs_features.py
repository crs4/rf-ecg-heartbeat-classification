from array import array
import math
from .common import SignalBuffer
from .common import safe_normalizer
import statistics as stats

def derivative(X, dt):
    dXdt = array('f', [0 for x in range(len(X))])
    for n in range(len(X)):
        if n > 0:
            dXdt[n] = (X[n] - X[n -1]) / dt # / 20 (when ploting)
    dXdt[0] = dXdt[1]
    return dXdt

def zeroCrossPoints(X):
    def isPositive(x):
        if x >= 0:
            return True
        return False
    lastSign = isPositive(X[0])
    zeroCross = array('B', [False for i in range(len(X))])
    for n in range(len(X)):
        if isPositive(X[n]) and not lastSign or (not isPositive(X[n]) and lastSign):
            zeroCross[n] = True
        lastSign = isPositive(X[n])
    return zeroCross

class ExtractQRS():
    """ QRS detection based on the Pan&Tompkins algorithm """
    def __init__(self):
        self.sampleCounter = 0
        self.fs = 150
        self.samplesSinceLastDetection = 0                          # contador de muestras luego del ultimo QRS 
        self.noiseLevel = 0.002                         # nivel de ruido esperado
        self.signalLevel = 0.006                      # nivel de seÒal esperado
        self.localMax = 0                         # valor maximo del pico detectado                    
        self.missedMax = 0                      # valor maximo del pico omitido
        self.samplesToLastMissed = 0                      # valor del contador de muestras en el pico omitido
        self.missedPeaks = 0                     # contador de picos omitidos
        self.threshold1 = 0.005                  # umbral de deteccion 1
        self.threshold2 = 0.0025                 # umbral de detecciÛn 2
        self.peakDetected = False                    # bandera de pico detectado
        self.qrsMissing = False                    # bandera de ausencia de QRS
        self.rrBuffer = SignalBuffer(32)              # vector para calculo de ultimos 8 RR
        self.qrsMaxBuffer = SignalBuffer(32)
        self.qrsMinBuffer = SignalBuffer(32)
        self.normalRRBuffer = SignalBuffer(8)
        self.samplesSinceLastLocalMax = 25                        # numero de muestras antes del punto de deteccion (peaki / 2) donde se encuentrÛ el QRS
        self.signalBuffer = SignalBuffer(96)             # vector que almacena los ˙ltimos 32 datos de la seÒal filtrada
        self.lastRSample = 0
        self.ready4NewPeak = False
        self.lastMissedQRSData = {}
        self.processedBuffer = SignalBuffer(64)
        self.filteredBuffer = SignalBuffer(96)
        self.starting = True
        self.lastMaxSlope = 0
        self.samplesSinceLastNoise = 0
        self.pPeakBuffer = SignalBuffer(32)
        self.rPeakBuffer = SignalBuffer(32)
        self.qPeakBuffer = SignalBuffer(32)
        self.sPeakBuffer = SignalBuffer(32)
        self.prWidthBuffer = SignalBuffer(32)
        self.qsWidthBuffer = SignalBuffer(32)
        self.qrsWidthBuffer = SignalBuffer(32)
        self.qrsWidth2Buffer = SignalBuffer(32)
        self.qrsWidth4Buffer = SignalBuffer(32)
        self.qrsSlopeBuffer = SignalBuffer(32)
        self.pqDiffBuffer = SignalBuffer(32)
        self.rqDiffBuffer = SignalBuffer(32)
        self.rsDiffBuffer = SignalBuffer(32)

    def resetCounters(self):
        self.sampleCounter = 0
        self.samplesSinceLastDetection = 0
        self.localMax = 0                         # valor maximo del pico detectado                    
        self.missedMax = 0                      # valor maximo del pico omitido
        self.samplesToLastMissed = 0                      # valor del contador de muestras en el pico omitido
        self.missedPeaks = 0                     # contador de picos omitidos
        self.peakDetected = False                    # bandera de pico detectado
        self.qrsMissing = False                    # bandera de ausencia de QRS
        self.lastRSample = 0
        self.ready4NewPeak = False
        self.lastMissedQRSData = {}
        
    def findQRSInSignalBuffer(self):
        # filtered_signal = self.filteredBuffer.getBuffer()
        signal_mean = self.signalBuffer.mean()
        signal = array('f', [s - signal_mean for s in self.signalBuffer.getBuffer()])
        L = len(signal)
        startIndex = L - 55                 
        endIndex = L - 25
        # rDelay = L - signalPeakIndex  

        # Calculo del ancho del QRS
        # signal = filtered_signal
        # startIndex = signalPeakIndex
        # endIndex = L - 1
        signalMax = max(signal[startIndex:endIndex])
        sMaxIndex = signal[startIndex:endIndex].index(signalMax) + startIndex
        signalMin = min(signal[startIndex:endIndex])
        sMinIndex = signal[startIndex:endIndex].index(signalMin) + startIndex
        signalPeak = signalMax
        signalPeakIndex = sMaxIndex
        if math.fabs(signalMin) > signalPeak:
            signalPeak = signalMin
            signalPeakIndex = sMinIndex

        maxQRSWidth = 30
        startIndex = int(signalPeakIndex - maxQRSWidth / 2)
        if startIndex < 0:
            startIndex = 0 
        endIndex = int(signalPeakIndex + maxQRSWidth / 2)
        if endIndex >=  L:
            endIndex = L
        dt = 1 / self.fs
        signalDerivative = derivative(signal, dt)
        derivativeZeroCrosses = zeroCrossPoints(signalDerivative)
        zeroCrosses = 0
        qrsStartIndex = startIndex
        qrsWaveform = array('f', [0 for i in range(31)])
        k = 15
        rPeak = 0
        rIndex = signalPeakIndex
        qPeak = 0
        qIndex = 0
        sIndex = 0
        sPeak = 0
        if signalPeak >= 0:
            rPeak = signalPeak
        lastSample = signalPeak
        qrsStartReady = False
        halfQrsStartIndex = 0
        quarterQrsStartIndex = 0
        for n in range(signalPeakIndex, startIndex, -1):    
            qrsWaveform[k] = signal[n]
            k -= 1  
            # if qrsStartReady == True and signal[n] * lastSample <= 0:
            #     # Found a signal sign change (zero cross) while ready to finish
            #     qrsStartIndex = n
            #     break
            if math.fabs(signal[n]) <= math.fabs(signalPeak) / 2 and halfQrsStartIndex == 0:
                halfQrsStartIndex = k
            if math.fabs(signal[n]) <= math.fabs(signalPeak) / 4 and quarterQrsStartIndex == 0:
                quarterQrsStartIndex = k
            if derivativeZeroCrosses[n] and n < signalPeakIndex - 1:
                zeroCrosses += 1
                if zeroCrosses == 1 and rPeak > 0:
                    # First peak before R peak
                    if signal[n-1] < 0:
                        # If negative then it's a Q peak
                        qPeak = signal[n-1]
                        qIndex = n - 1
                    # Assume QRS is finishing
                    qrsStartReady = True
                elif zeroCrosses == 1 and rPeak == 0:
                    # First peak before a main negative peak (Q or S)
                    if signal[n-1] > 0:
                        # It's a R peak
                        rPeak = signal[n - 1]
                        rIndex = n -1
                        # Main peak is a S peak
                        sPeak = signalPeak
                        sIndex = signalPeakIndex
                    else:
                        # Main peak is a Q peak
                        qPeak = signalPeak
                        qIndex = signalPeakIndex
                        # Assume QRS is finishing
                        qrsStartReady = True
                if zeroCrosses == 2:
                    if qPeak == 0 and signal[n-1] < 0:
                        qPeak = signal[n-1]
                        qIndex = n - 1
                        qrsStartReady = True
                    else:
                        # Found a second positive peak before R or Q Peak
                        qrsStartIndex = n -1
                        break
            
            lastSample = signal[n]
            
        if qIndex == 0:
            qIndex = qrsStartIndex
        qrsEndIndex = endIndex
        k = 16
        zeroCrosses = 0
        lastSample = signalPeak
        qrsEndReady = False
        halfQrsEndIndex = 30
        quarterQrsEndIndex = 30
        for n in range(signalPeakIndex + 1, endIndex):  
            qrsWaveform[k] = signal[n]
            k += 1
            if math.fabs(signal[n]) <= math.fabs(signalPeak) / 2 and halfQrsEndIndex == 30:
                halfQrsEndIndex = k
            if math.fabs(signal[n]) <= math.fabs(signalPeak) / 2 and quarterQrsEndIndex == 30:
                quarterQrsEndIndex = k
            if qrsEndReady == True and signal[n] * lastSample <= 0:
                # Found a signal sign change (zero cross) while ready to finish
                qrsEndIndex = n
                break
            if derivativeZeroCrosses[n] and n > signalPeakIndex + 2:
                zeroCrosses += 1
                if zeroCrosses == 1 and rPeak == signalPeak:
                    # First peak after R peak
                    if signal[n-1] < 0:
                        # If negative then it's a S peak
                        sPeak = signal[n-1]
                        sIndex = n -1 
                # Otherwise assume QRS is over
                qrsEndReady = True
                if zeroCrosses == 2:
                    # Found a second peak after R or S so assume it's the end
                    qrsEndIndex = n -1
                    break
            
            lastSample = signal[n]
        if sIndex == 0:
            sIndex = qrsEndIndex
        # P wave finding
        pStart = qrsStartIndex - 35
        if pStart < 10:
            pStart = 10
        pEnd = qrsStartIndex - 10
        baseStart = pStart - 10
        if pEnd < 20:
            pEnd = 20
        std_noise = stats.stdev(signal[baseStart:pStart])
        mean_noise = stats.mean(signal[baseStart:pStart])
        pPeak = max(signal[pStart:pEnd])
        pIndex = signal[pStart:pEnd].index(pPeak) + pStart
        if pPeak < mean_noise + 3 * std_noise or pIndex == pStart or pIndex == pEnd:
            pPeak = 0
            pIndex = qrsStartIndex
        qsWidth = (sIndex - qIndex) * dt
        prWidth = (qrsStartIndex - pIndex) * dt
        maxSlope = abs(max(signalDerivative[qrsStartIndex:qrsEndIndex]))
        qrsWidth = (qrsEndIndex - qrsStartIndex) * dt
        halfQrsWidth = (halfQrsEndIndex - halfQrsStartIndex) * dt
        quarterQrsWidth = (quarterQrsEndIndex - quarterQrsStartIndex) * dt
        if False: #Debug
            import matplotlib.pyplot as plt
            plt.plot(signal, '-', pIndex, pPeak, 'r*', qIndex, qPeak, 'ro', rIndex, rPeak, 'rx', sIndex,
                     sPeak, 'r+', qrsStartIndex, signal[qrsStartIndex], 'g.', qrsEndIndex, signal[qrsEndIndex], 'g.')
            plt.show()
        
        pqDiff = pPeak - qPeak
        rqDiff = rPeak - qPeak
        rsDiff = rPeak - sPeak
        self.pPeakBuffer.push(pPeak)
        self.rPeakBuffer.push(rPeak)
        self.qPeakBuffer.push(qPeak)
        self.sPeakBuffer.push(sPeak)
        self.prWidthBuffer.push(prWidth)
        self.qsWidthBuffer.push(qsWidth)
        self.qrsWidthBuffer.push(qrsWidth)
        self.qrsWidth2Buffer.push(halfQrsWidth)
        self.qrsWidth4Buffer.push(quarterQrsWidth)
        self.qrsSlopeBuffer.push(maxSlope)
        self.pqDiffBuffer.push(pqDiff)
        self.rqDiffBuffer.push(rqDiff)
        self.rsDiffBuffer.push(rsDiff)
        return {
            # 'rDelay': rDelay, 
            # 'qrsWaveform': qrsWaveform2, 
            # 'qrsMax': qrsMax, 
            # 'qrsMin': qrsMin, 
            'QRSw': qrsWidth, 
            'QRSw2': halfQrsWidth,
            'QRSw4': quarterQrsWidth,
            'QSw': qsWidth,
            'PRw': prWidth,
            'Ppeak': pPeak,
            'Rpeak': rPeak,
            'Qpeak': qPeak,
            'Speak': sPeak,
            'PQa': pqDiff,
            'RQa': rqDiff,
            'RSa': rsDiff,
            'Ppeak_norm': safe_normalizer(pPeak,self.pPeakBuffer.mean()),
            'Rpeak_norm': safe_normalizer(rPeak,self.rPeakBuffer.mean()),
            'Qpeak_norm': safe_normalizer(qPeak,self.qPeakBuffer.mean()),
            'Speak_norm': safe_normalizer(sPeak,self.sPeakBuffer.mean()),
            'PQa_norm': safe_normalizer(pqDiff, self.pqDiffBuffer.mean()),
            'RQa_norm': safe_normalizer(rqDiff, self.rqDiffBuffer.mean()),
            'RSa_norm': safe_normalizer(rsDiff, self.rsDiffBuffer.mean()),
            'PRa_norm': safe_normalizer(prWidth, self.prWidthBuffer.mean()),
            'QSa_norm': safe_normalizer(qsWidth, self.qsWidthBuffer.mean()),
            'QRSw_norm': safe_normalizer(qrsWidth, self.qrsWidthBuffer.mean()),
            'QRSw2_norm': safe_normalizer(halfQrsWidth, self.qrsWidth2Buffer.mean()),
            'QRSw4_norm': safe_normalizer(quarterQrsWidth, self.qrsWidth4Buffer.mean()),
            'QRSs': maxSlope, 
            'QRSs_norm': safe_normalizer(maxSlope, self.qrsSlopeBuffer.mean())
        }

    def __call__(self, beatTime, signal):
        beatSample = int(beatTime * 150)
        for n in range(beatSample - 128, beatSample + 40):#32
            rawSample = signal[n]
            self.signalBuffer.push(rawSample)
        return self.findQRSInSignalBuffer()
                         


