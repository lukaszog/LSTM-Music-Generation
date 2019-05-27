import pickle

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from music21 import chord, pitch, note, stream
import utils
from functools import reduce

SEQ_LEN = 50


def generate():
    model = create_network('results/213')
    prediction_output = generate_notes(model)


def create_network(result_dir):
    results_dir = utils.get_results_dir(result_dir)

    model = utils.load_model_from_json(results_dir)

    model.compile(
        loss='mse',
        optimizer='rmsprop',
        metrics=['accuracy']
    )

    model.summary()

    return model


def generate_notes(model):
    data = pickle.load(open("dataset/folk_music_803_tune0_clean.digits", "rb"))
    # data = data[0:200]
    # print(data)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = data.reshape(-1, 1)
    data = scaler.fit_transform(np.array(data))
    # plt.hist(data)
    # plt.show()
    # data = data.reshape(-1, 1)
    # scaler = StandardScaler()
    # scaler.fit(data)
    # data = scaler.transform(data)
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # data = scaler.fit(data.reshape(-1, 1))

    # 0001000000010101111
    input_data, output_data = utils.prepare_seq(data, SEQ_LEN)

    X = input_data.reshape((input_data.shape[0], 1, input_data.shape[1]))
    y = output_data.reshape((len(output_data), 1))

    print("Shape y", y.shape)

    trainPredict = model.predict(X)
    trainPredict = scaler.inverse_transform(trainPredict)
    #
    print(np.array(trainPredict, dtype=int))
    print(len(np.array(trainPredict)))
    print(type(np.array(trainPredict)))

    #pickle.dump(trainPredict, open("ballada.bin", "wb"))
    # TODO: napisac dekoder
    offset = 0
    output_notes = []
    for i in range(0, 100):
        print([int(d) for d in str(bin(np.array(trainPredict, dtype=int)[i][0]))[2:]])
        print(bin(np.array(trainPredict, dtype=int)[i][0]))
        note = convert_binary_note([int(d) for d in str(bin(np.array(trainPredict, dtype=int)[i][0]))[2:]])
        note.offset = offset
        offset += 0.5
        output_notes.append(note)

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='test_output.mid')

    # plt.show()
    # print(trainY)

def convert_binary_note(data, order=0):
    # order = 0 octave first then notes
    # order = 1 fist notes then octave
    # 0 0 0 0 0 0 0  0 0 0 0 0 0 0 0 0 0 0 0
    octave = data[0:6]
    notes = data[6:]

    octave = [i for i, e in enumerate(octave) if e == 1]
    notes = [i for i, e in enumerate(notes) if e == 1]
    if len(notes) > 1:
        n = chord.Chord([x-1 for x in notes])
    else:
        n = note.Note(notes[0] - 1)
    n = note.Note(notes[0] - 1)
    n.octave = octave[0] + 1
    print("Oktawa: ", octave)
    print("Nuty: ", notes)
    print("Nuta: ", n)

    return n


if __name__ == '__main__':
    generate()
