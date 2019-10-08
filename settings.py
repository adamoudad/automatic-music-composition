# Data representation
NOTE_DICTIONARY = {
    'A': 0,
    'A#': 1,
    'B-': 1,
    'B': 2,
    'C': 3,
    'C#': 4,
    'D-': 4,
    'D': 5,
    'D#': 6,
    'E-': 6,
    'E': 7,
    'F': 8,
    'F#': 9,
    'G-': 9,
    'G': 10,
    'G#': 11
}
from fractions import Fraction
DURATION_DICTIONARY = {
    0.25: 0,
    Fraction(1,3): 1,
    1/3: 1,
    0.5: 2,
    2/3: 3,
    Fraction(2,3): 3,
    0.75: 4,
    1: 5,
    1.25: 6,
    Fraction(4,3): 7,
    1.5: 8,
    1.75: 9,
    Fraction(5,3): 10,
    2.0: 11,
    2.25: 12,
    Fraction(8,3): 13,
    3.0: 14,
}                          # quarter length x[0.25,1/3,0.5,2/3,0.75,1]

N_SCALES = len(set(NOTE_DICTIONARY.values()))
N_DURATIONS = len(set(DURATION_DICTIONARY.values()))

# Training (default to same as deepjazz github repo)
N_CLASSES = N_SCALES * N_DURATIONS
BATCH_SIZE = 128
EPOCHS = 100

# Sequences
STEP = 3
MAX_LENGTH = 20
