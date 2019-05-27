import pickle
from music21 import converter, instrument, note, chord, common, pitch
from bitarray import bitarray
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

octave_numbers = [0, 0, 12, 24, 36, 48, 60, 72, 84, 96]
data = pickle.load(open("dataset/folk_music_803_tune0.notes", "rb"))
data = data[0:5]
# data = data[0:50]
# print(Counter(data))

# labels, values = zip(*Counter(data).items())
#
#
# indexes = np.arange(len(labels))
# width = 0.5
#
# plt.bar(indexes, values, width)
# plt.xticks(indexes + width * 0.5, labels)
# plt.show()

# <music21.note.Note C>
# bitarray('10000000000001000000')

# data = data[0:5]
# print(data)
NUM_BITS = 19
notes_binary = []
notes_digits = []
keyboard = bitarray(NUM_BITS)
keyboard.setall(0)
for n in data:
    nb = n.split('.')
    # print(nb)
    octave = -1
    for m in nb:
        try:
            # print(m)
            f = note.Note(m)

            mantise = int("0x" + f.pitch.pitchClassString, 0)
            # print(f)
            # 0 0 0 0 0 0 0  0 0 0 0 0 0 0 0 0 0 0 0
            #if octave == -1:

            keyboard[7 + mantise] = 1
            if f.octave == 1:
                print("Stawiam 1")
                octave = 2
            if f.octave is None or f.octave == 0:
                octave = 4
            else:
                octave = f.octave
            # print(octave)
            # print(f)
            try:
                keyboard[octave-1] = 1
                print(keyboard)
                print("{} {}".format(f, f.octave))
                # print(int(keyboard.to01(), 2))
            except IndexError:
                print("ERROR: ", f)
                print(octave)
            # print(keyboard)
            #exit()
        except pitch.PitchException:
            # error += 1
            pass

    if int(keyboard.to01(), 2) < 150000:
        notes_digits.append(int(keyboard.to01(), 2))
        notes_binary.append(keyboard.to01())
    else:
        print("Dziwne dane: {} {}".format(nb, keyboard.to01()))
    # print(keyboard.to01())
    # print(int(keyboard.to01(), 2))
    # notes.append(str(int(keyboard.to01(), 2)))
    keyboard = bitarray(NUM_BITS)
    keyboard.setall(0)
    # print(note_number)
print(notes_binary)
exit()
pickle.dump(notes_binary, open("dataset/folk_music_803_tune0.bits", "wb"))
pickle.dump(notes_digits, open("dataset/folk_music_803_tune0.digits", "wb"))
