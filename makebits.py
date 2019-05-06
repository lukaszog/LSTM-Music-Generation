import pickle
from music21 import converter, instrument, note, chord, common
from bitarray import bitarray

octave_numbers = [0, 0, 12, 24, 36, 48, 60, 72, 84, 96]
data = pickle.load(open("dataset/tpd_classical.notes", "rb"))

data = data[0:5]

keyboard = bitarray(32)
keyboard.setall(0)
for n in data:
    nb = n.split('.')
    # print(nb)
    octave = ""
    for m in nb:
        try:
            # print(m)
            f = note.Note(m)
            # print(f)
            octave = int(f.octave if f.octave else 4)
            mantise = int("0x" + f.pitch.pitchClassString, 0)
            print(octave)

            number = octave_numbers[octave] + mantise + 3
            # print(f)
            # print("Oktawa, ", octave)
            # print("Klasa:", mantise)
            print("Number: ", number)

            if f.octave is None:
                octave = 4
            else:
                octave = f.octave
            # keyboard[12 + octave - 1] = 1
        except Exception:
            pass
    # print(note_number)