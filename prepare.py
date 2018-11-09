from music21 import note, stream, pitch, converter, instrument, chord
import numpy as np
from bitarray import bitarray
# import re
import pickle
import glob
from sklearn.preprocessing import MinMaxScaler


# notes = []
# for file in glob.glob("../midi/preludes/*.mid"):
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
#
#     with open('./preludia', 'wb') as filepath:
#         pickle.dump(notes, filepath)
data = pickle.load(open("preludia", "rb"))


notes = []

keyboard = bitarray(32)
keyboard.setall(0)

error = 0

for n in data:
    nb = n.split('.')
    # print(nb)
    octave = ""
    for m in nb:
        try:
            # print(m)
            f = note.Note(m)
            mantise = int("0x" + f.pitch.pitchClassString, 0)
            keyboard[mantise] = 1
            if f.octave is None:
                octave = 4
            else:
                octave = f.octave
            keyboard[12 + octave - 1] = 1
        except pitch.PitchException:
            error += 1

    # print(octave)
    # print(int(keyboard.to01(), 2))
    notes.append(str(int(keyboard.to01(), 2)))
    keyboard = bitarray(32)
    keyboard.setall(0)

keyboard = int(keyboard.to01(), 2)

note_data = []

for k in notes:
    note_data.append(''.join(str(x) for x in k))
    # print(''.join(str(x) for x in k))


SEQ_LEN = 4

note_data = np.array(note_data)
note_data = note_data.reshape(len(note_data), 1)

type(note_data)

scaler = MinMaxScaler(feature_range=(0, 1))
note_data = scaler.fit_transform(note_data)

input_notes = []
label_notes = []


for i in range(0, len(note_data) - SEQ_LEN, 1):
    input_notes.append(note_data[i:i + SEQ_LEN])
    label_notes.append(note_data[i + SEQ_LEN])
#
print(input_notes)
print("=======")
print(label_notes)

pickle.dump(input_notes, open("input", "wb"))
pickle.dump(label_notes, open("output", "wb"))

print("error: ", error)

# X = np.zeros((len(input_notes), SEQ_LEN, nb_chars), dtype=np.int)
# y = np.zeros((len(input_notes), nb_chars), dtype=np.int)
# for i, input_notes in enumerate(input_notes):
#     for j, ch in enumerate(input_notes):
#         X[i, j, char2index[ch]] = 1
#     y[i, char2index[label_notes[i]]] = 1
#
# print(X)

#
# HIDDEN_SIZE = 128
# BATCH_SIZE = 128
# NUM_ITERATIONS = 25
# NUM_EPOCHS_PER_ITERATION = 1
# NUM_PREDS_PER_EPOCH = 100
# model = Sequential()
# model.add(LSTM(
#     156,
#     input_shape=(10,4),
#     return_sequences=True
# ))
# model.add(Dropout(0.3))
# model.add(LSTM(112, return_sequences=True))
# model.add(Dropout(0.3))
# model.add(Dense(nb_chars))
# model.add(Activation('softmax'))
# #
# # model.summary()
# model.summary()
# model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
# #
# history = model.fit(np.array(input_notes), np.array(label_notes),
#                     # validation_data=(X_test, y_test),
#                     validation_split=0.33,
#                     epochs=200,
#                     batch_size=64,
#                     verbose=1,
#                     )
# #
