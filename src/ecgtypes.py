from enum import Enum

class HeartRhythm(Enum):
    NORMAL = 0
    BIGEMINY = 1
    TRIGEMINY = 2
    VENTRICULAR_TACHYCARDIA = 3
    SUPRAVENTRICULAR_TACHYCARDIA = 4
    ATRIAL_FIBRILATION = 5
    VENTRICULAR_FIBRILATION = 6
    SINUS_BRADYCARDIA = 7
    SECOND_DEGREE_BLOCK = 8
    PACED = 9
    OTHER = 10
    def symbol(self):
        symbols = {
            self.NORMAL.value: 'N',
            self.BIGEMINY.value: 'B',
            self.TRIGEMINY.value: 'T',
            self.VENTRICULAR_TACHYCARDIA.value: 'VT',
            self.SUPRAVENTRICULAR_TACHYCARDIA.value: 'SVTA',
            self.ATRIAL_FIBRILATION.value: 'AFIB',
            self.VENTRICULAR_FIBRILATION.value: 'VFL',
            self.SINUS_BRADYCARDIA.value: 'SBR',
            self.SECOND_DEGREE_BLOCK.value: 'BII',
            self.PACED.value: 'P',
            self.OTHER.value: 'O'
        }
        return symbols[self.value]
    @classmethod
    def new_from_symbol(cls, symbol):
        for rhythm in cls:
            if symbol == rhythm.symbol():
                return rhythm
        return cls.OTHER

class BeatType(Enum):
    NORMAL = 0
    AURICULAR_PREMATURE_CONTRACTION = 1
    PREMATURE_VENTRICULAR_CONTRACTION = 2
    FUSION = 3
    UNKNOWN = 4
    OTHER = 5
    def symbol(self):
        symbols = {
            self.NORMAL.value: 'N',
            self.AURICULAR_PREMATURE_CONTRACTION.value: 'S',
            self.PREMATURE_VENTRICULAR_CONTRACTION.value: 'V',
            self.FUSION.value: 'F',
            self.UNKNOWN.value: 'Q',
            self.OTHER.value: '?'
        }
        return symbols[self.value]
    @classmethod
    def new_from_symbol(cls, symbol):
        # cls here is the enumeration
        N = {'N', 'R', 'L', '.', 'e', 'j', 'n'}
        S = {'S', 'A', 'a', 'J'}
        V = {'V', 'E'}
        F = {'F'}
        Q = {'P', '/', 'U', 'f', 'Q'}
        if symbol in N:
            return cls.NORMAL
        elif symbol in S:
            return cls.AURICULAR_PREMATURE_CONTRACTION
        elif symbol in V:
            return cls.PREMATURE_VENTRICULAR_CONTRACTION
        elif symbol in F:
            return cls.FUSION
        elif symbol in Q:
            return cls.UNKNOWN
        else:
            return cls.OTHER