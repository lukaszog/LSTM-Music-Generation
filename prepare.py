from music21 import converter, instrument, note, chord, common
import numpy as np
from bitarray import bitarray
# import re
import pickle
import glob
# from sklearn.preprocessing import MinMaxScaler
# import utils
# import matplotlib.pyplot as plt
from collections import Counter
from music21 import corpus
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

MIDI_PATH = "/home/lukasz/Pobrane/session"
notes = []
error = 0
parsed = 1
for file in glob.glob(MIDI_PATH + "/*.mid"):
    print("Parsing %s" % file)
    try:
        midi = converter.parseFile(file)
    except IndexError:
        error = error + 1
        print("Blad index error numer {}".format(error))
    print("Przeparsowalem {}".format(parsed))
    parsed = parsed + 1
    notes_to_parse = None
    try:
        ins = instrument.partitionByInstrument(midi)
        notes_to_parse = ins.parts[0].recurse()
    except:
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.pitches))
            # print(type(element))
            # print('.'.join(str(n) for n in element.pitches))


# def get_notes_parallel(file):
#     notes = []
#     midi = converter.parse(file)
#     print("Parsing %s" % file)
#     notes_to_parse = None
#
#     try:
#         s2 = instrument.partitionByInstrument(midi)
#         notes_to_parse = s2.parts[0].recurse()
#     except:
#         notes_to_parse = midi.flat.notes
#
#     for element in notes_to_parse:
#         if isinstance(element, note.Note):
#             notes.append(str(element.pitch))
#         elif isinstance(element, chord.Chord):
#             notes.append('.'.join(str(n) for n in element.pitches))
#
#     return notes
#
# files = []
#
# for file in glob.glob(MIDI_PATH + "/*.mid"):
#     files.append(file)
#
# notes = common.runParallel(files, parallelFunction=get_notes_parallel)
# notes = [item for list in notes for item in list]



pickle.dump(notes, open("folk_music.notes", "wb"))
data = pickle.load(open("folk_music.notes", "rb"))

print(Counter(data))
# s = converter.parse('ballada/ballade3.mid')
# s.plot('histogram', 'pitch')

exit()

octave_numbers = [0, 0, 12, 24, 36, 48, 60, 72, 84, 96]

# data = ['G', 'C', 'C7', 'E.G', 'C', 'C7']
# encoder = LabelEncoder()
# print(encoder.fit_transform(data))
#
# exit()

notes = []
NUM_BITS = 88
keyboard = bitarray(NUM_BITS)
keyboard.setall(0)

error = 0

print(len(data))


minimum = 44
maximum = 44

for n in data:
    nb = n.split('.')
    # print(nb)
    octave = ""
    note_number = 88 * [0]
    for m in nb:
        try:
            # print(m)
            f = note.Note(m)
            # print(f)
            octave = int(f.octave if f.octave else 4)
            mantise = int("0x" + f.pitch.pitchClassString, 0)
            # print(octave)

            number = octave_numbers[octave] + mantise + 3
            # print(f)
            # print("Oktawa, ", octave)
            # print("Klasa:", mantise)
            # print("Number: ", number)
            if number > maximum:
                maximum = number
            if number < minimum:
                minimum = number
            keyboard[number] = 1
            note_number[number] = 1
            # if f.octave is None:
            #     octave = 4
            # else:
            #     octave = f.octave
            # keyboard[12 + octave - 1] = 1
        except pitch.PitchException:
            error += 1
    # print(note_number)


    # print(octave)
    # print(type(keyboard.to01()))
    # print("Int: ", int(keyboard.to01(), 2))
    notes.append(note_number)
    keyboard = bitarray(NUM_BITS)
    keyboard.setall(0)


pickle.dump((notes), open("chopin88bits.vector", "wb"))


exit()
print("Minimum: ", minimum)
print("Maximum: ", maximum)

for i in notes:
    print(i)
    np.array(i, dtype=np.int64)


exit()

notes_array = np.array([])
tmp_array = np.array([])


keyboard = int(keyboard.to01(), 2)
print(notes)
note_data = []

minimum = 0x0000000000000000000000000000000000000001000000000000000000000000000000000000000000000000
print((hex(minimum)))


exit()


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



# print(note_data)
# note_data = np.array(note_data, dtype=np.int64)

pickle.dump(note_data, open("chopin88bits.binary", "wb"))


