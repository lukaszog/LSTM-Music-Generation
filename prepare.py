# from music21 import converter, instrument, note, chord, common, pitch, interval
import numpy as np
# from bitarray import bitarray
# import re
import pickle
import glob
# from sklearn.preprocessing import MinMaxScaler
# import utils
# import matplotlib.pyplot as plt
from collections import Counter
# from music21 import corpus
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import LabelEncoder
import midi
lowerBound = 24
upperBound = 102
from collections import defaultdict

import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
RANGE = 100

def round_tick(tick, time_step):
    return int(round(tick/float(time_step)) * time_step)

def ingest_notes(track, verbose=False):

    notes = { n: [] for n in range(RANGE) }
    current_tick = 0

    for msg in track:
        # ignore all end of track events
        if isinstance(msg, midi.EndOfTrackEvent):
            continue

        if msg.tick > 0:
            current_tick += msg.tick

        # velocity of 0 is equivalent to note off, so treat as such
        if isinstance(msg, midi.NoteOnEvent) and msg.get_velocity() != 0:
            if len(notes[msg.get_pitch()]) > 0 and \
               len(notes[msg.get_pitch()][-1]) != 2:
                if verbose:
                    print "Warning: double NoteOn encountered, deleting the first"
                    print msg
            else:
                notes[msg.get_pitch()] += [[current_tick]]
        elif isinstance(msg, midi.NoteOffEvent) or \
            (isinstance(msg, midi.NoteOnEvent) and msg.get_velocity() == 0):
            # sanity check: no notes end without being started
            if len(notes[msg.get_pitch()][-1]) != 1:
                if verbose:
                    print "Warning: skipping NoteOff Event with no corresponding NoteOn"
                    print msg
            else:
                notes[msg.get_pitch()][-1] += [current_tick]

    return notes, current_tick

def round_notes(notes, track_ticks, time_step, R=None, O=None):
    if not R:
        R = RANGE
    if not O:
        O = 0

    sequence = np.zeros((track_ticks/time_step, R))
    disputed = { t: defaultdict(int) for t in range(track_ticks/time_step) }
    for note in notes:
        for (start, end) in notes[note]:
            start_t = round_tick(start, time_step) / time_step
            end_t = round_tick(end, time_step) / time_step
            # normal case where note is long enough
            if end - start > time_step/2 and start_t != end_t:
                sequence[start_t:end_t, note - O] = 1
            # cases where note is within bounds of time step
            elif start > start_t * time_step:
                disputed[start_t][note] += (end - start)
            elif end <= end_t * time_step:
                disputed[end_t-1][note] += (end - start)
            # case where a note is on the border
            else:
                before_border = start_t * time_step - start
                if before_border > 0:
                    disputed[start_t-1][note] += before_border
                after_border = end - start_t * time_step
                if after_border > 0 and end < track_ticks:
                    disputed[start_t][note] += after_border

    # solve disputed
    for seq_idx in range(sequence.shape[0]):
        if np.count_nonzero(sequence[seq_idx, :]) == 0 and len(disputed[seq_idx]) > 0:
            # print seq_idx, disputed[seq_idx]
            sorted_notes = sorted(disputed[seq_idx].items(),
                                  key=lambda x: x[1])
            max_val = max(x[1] for x in sorted_notes)
            top_notes = filter(lambda x: x[1] >= max_val, sorted_notes)
            for note, _ in top_notes:
                sequence[seq_idx, note - O] = 1

    return sequence


def parse_midi_to_sequence(input_filename, time_step, verbose=False):
    sequence = []
    pattern = midi.read_midifile(input_filename)

    print(type(pattern))
    if len(pattern) < 1:
        raise Exception("No pattern found in midi file")

    if verbose:
        print "Track resolution: {}".format(pattern.resolution)
        print "Number of tracks: {}".format(len(pattern))
        print "Time step: {}".format(time_step)

    # Track ingestion stage
    notes = { n: [] for n in range(RANGE) }
    track_ticks = 0
    for track in pattern:
        current_tick = 0
        # print(len(track))
        track = track[0:10]
        for msg in track:
            # ignore all end of track events
            if isinstance(msg, midi.EndOfTrackEvent):
                continue

            if msg.tick > 0:
                current_tick += msg.tick

            # velocity of 0 is equivalent to note off, so treat as such
            if isinstance(msg, midi.NoteOnEvent) and msg.get_velocity() != 0:
                if len(notes[msg.get_pitch()]) > 0 and \
                   len(notes[msg.get_pitch()][-1]) != 2:
                    if verbose:
                        print "Warning: double NoteOn encountered, deleting the first"
                        print msg
                else:
                    notes[msg.get_pitch()] += [[current_tick]]
            elif isinstance(msg, midi.NoteOffEvent) or \
                (isinstance(msg, midi.NoteOnEvent) and msg.get_velocity() == 0):
                # sanity check: no notes end without being started
                if len(notes[msg.get_pitch()][-1]) != 1:
                    if verbose:
                        print "Warning: skipping NoteOff Event with no corresponding NoteOn"
                        print msg
                else:
                    notes[msg.get_pitch()][-1] += [current_tick]

        track_ticks = max(current_tick, track_ticks)

    track_ticks = round_tick(track_ticks, time_step)
    if verbose:
        print "Track ticks (rounded): {} ({} time steps)".format(track_ticks, track_ticks/time_step)

    sequence = round_notes(notes, track_ticks, time_step)

    return sequence

print (parse_midi_to_sequence("midi_datasets/irish803/sessiontune0.mid", 100, True))
exit()


MIDI_PATH = "midi_datasets"
notes = []
error = 0
parsed = 1
for file in glob.glob(MIDI_PATH + "/*.mid"):
    print("Parsing %s" % file)

    try:
        midi = converter.parseFile(file)
        k = midi.analyze('key')
        i = interval.Interval(k.tonic, pitch.Pitch('C'))
        midi = midi.transpose(i)
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



pickle.dump(notes, open("dataset/folk_music_803_tune15.notes", "wb"))
data = pickle.load(open("dataset/folk_music_803_tune15.notes", "rb"))

print(Counter(data))
# s = converter.parse('ballada/ballade3.mid')
# s.plot('histogram', 'pitch')
print("Przeprasowanych: {}".format(parsed))
print("Bledow: {}".format(error))

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


