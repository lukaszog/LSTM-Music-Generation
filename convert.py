import pickle
import numpy as np
from music21 import chord
from music21 import stream

note_values = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

get_bin = lambda x: format(x, 'b')
notes = pickle.load(open("ballada.bin", "rb"))

notes = np.array(notes, dtype=int)
notes = notes[0:50]
# notes = [notes[i] for i in notes]

n = []
for i in notes:
    for k in i:
        n.append(get_bin(k))


chords = []
offset = 0

for i in n:
    print(i[0:12])
    chord_list = []
    for k in range(0, 12):
        if i[k] == '1':
            chord_list.append(note_values[k])
            print(note_values[k])
    new_chord = chord.Chord(chord_list)
    new_chord.offset = offset
    chords.append(new_chord)
    offset += 1
    print(offset)
    print("===")

for k in chords:
    print(k)
midi_stream = stream.Stream(chords)
midi_stream.write('midi', fp='test_output.mid')



