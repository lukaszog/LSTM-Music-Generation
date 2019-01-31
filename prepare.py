from music21 import note, stream, pitch, converter, instrument, chord
import numpy as np
from bitarray import bitarray
# import re
import pickle
import glob
from sklearn.preprocessing import MinMaxScaler
import utils
import matplotlib.pyplot as plt
from collections import Counter
from music21 import corpus

notes = []
# for file in glob.glob("ballada/*.mid"):
#     print("Parsing %s" % file)
#     midi = converter.parse(file)
#     notes_to_parse = None
#     try:
#         ins = instrument.partitionByInstrument(midi)
#         notes_to_parse = ins.parts[0].recurse()
#     except:
#         notes_to_parse = midi.flat.notes
#
#     for element in notes_to_parse:
#         if isinstance(element, note.Note):
#             notes.append(str(element.pitch))
#         elif isinstance(element, chord.Chord):
#             notes.append('.'.join(str(n) for n in element.pitches))
#             # print(type(element))
#             # print('.'.join(str(n) for n in element.pitches))


# pickle.dump(notes, open("ballada.notes", "wb"))
data = pickle.load(open("all_chopin", "rb"))

print(Counter(data))
# s = converter.parse('ballada/ballade3.mid')
# s.plot('histogram', 'pitch')

# exit()
# data = ['A2.F2.D2', 'F4', 'E.G']

notes = []
NUM_BITS = 89
keyboard = bitarray(NUM_BITS)
keyboard.setall(0)

error = 0

print(len(data))



for n in data:
    nb = n.split('.')
    # print(nb)
    octave = ""
    for m in nb:
        try:
            # print(m)
            f = note.Note(m)
            mantise = int("0x" + f.pitch.pitchClassString, 0)
            octave = int(f.octave if f.octave else 4)
            mantise = int("0x" + f.pitch.pitchClassString, 0)
            number = mantise + 12 * octave - 1
            keyboard[number] = 1
            # if f.octave is None:
            #     octave = 4
            # else:
            #     octave = f.octave
            # keyboard[12 + octave - 1] = 1
        except pitch.PitchException:
            error += 1

    print(octave)
    print(keyboard)
    print("Int: ", int(keyboard.to01(), 2))
    notes.append(str(int(keyboard.to01(), 2)))
    keyboard = bitarray(NUM_BITS)
    keyboard.setall(0)

keyboard = int(keyboard.to01(), 2)
# exit()
print(notes)
note_data = []

for k in notes:
    note_data.append(int(''.join(str(x) for x in k)))

print(note_data)

note_data = np.array(note_data)
print(Counter(note_data))

# fig_per_hour = plt.figure()
# per_hour = fig_per_hour.add_subplot(111)
# counts, bins, patches = per_hour.hist(
#     note_data, bins = 100, normed = False, color = 'g',linewidth=0)
# plt.gca().set_xlim(note_data.min(), note_data.max())
# plt.show()

note_data = note_data.reshape(len(note_data), 1)

print(note_data)

note_data = np.array(note_data, dtype=np.int64)

pickle.dump(note_data, open("chopin88bits.binary", "wb"))


