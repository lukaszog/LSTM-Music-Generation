from music21 import note, stream, pitch, converter, instrument, chord
import numpy as np
from bitarray import bitarray
# import re
import pickle
import glob
from sklearn.preprocessing import MinMaxScaler
import utils

# notes = []
# for file in glob.glob("chopin/*.mid"):
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
#             print(str(element.pitch))
#         elif isinstance(element, chord.Chord):
#             notes.append('.'.join(str(n) for n in element.pitches))
#             # print(type(element))
#             print('.'.join(str(n) for n in element.pitches))
#
#     with open('./all_chopin', 'wb') as filepath:
#         pickle.dump(notes, filepath)
#
# exit()
data = pickle.load(open("notes_tbt_classical", "rb"))

# print(data[0:10])
# exit()

notes = []
NUM_BITS = 98
keyboard = bitarray(NUM_BITS)
keyboard.setall(0)

error = 0

for n in data:
    nb = n.split('.')
    # print(nb)
    octave = ""
    for m in nb:
        # try:
            # print(m)
        print(m)
        f = note.Note(m)
        octave = int(f.octave if f.octave else 4)
        mantise = int("0x" + f.pitch.pitchClassString, 0)
        keyboard[mantise + 12 * octave] = 1
            # keyboard[12 + octave - 1] = 1
        # except pitch.PitchException:
        #     error += 1

    # print(octave)
    print(keyboard)
    notes.append(str(int(keyboard.to01(), 2)))
    keyboard = bitarray(NUM_BITS)
    keyboard.setall(0)

keyboard = int(keyboard.to01(), 2)

print(notes)

note_data = []

for k in notes:
    note_data.append(''.join(str(x) for x in k))

print(note_data)

note_data = np.array(note_data)
note_data = note_data.reshape(len(note_data), 1)

print(note_data)

scaler = MinMaxScaler(feature_range=(0, 1))
note_data = scaler.fit_transform(note_data)
print(note_data)

pickle.dump(note_data, open("all_chopin_data", "wb"))
print('Error: ', error)

