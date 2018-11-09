import re
from music21 import note
import numpy as np

data = ['A4', 'F4', 'E.G']
# data = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B' ]

keyboard = np.zeros(88)

for d in data:
    n = d.split(".")
    print('Dlugosc', len(n))
    for m in n:
        f = note.Note(m)
        # print(m, int("0x" + f.pitch.pitchClassString, 0), f.octave if f.octave else 0)
        # print(m, int("0x" + f.pitch.pitchClassString, 0), f.octave if f.octave else 0)
        octave = int(f.octave if f.octave else 0)
        mantise = int("0x" + f.pitch.pitchClassString, 0)
        # number = mantise + 12 * octave
        # print(m, number)
        keyboard[mantise] = 1
    print(keyboard)
    keyboard = np.zeros(88)