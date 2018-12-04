import pickle

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import utils

SEQ_LEN = 100


def generate():
    model = create_network('results/01')
    prediction_output = generate_notes(model)


def create_network(result_dir):
    results_dir = utils.get_results_dir(result_dir)

    model = utils.load_model_from_json(results_dir)

    model.compile(
        loss='mse',
        optimizer='adam',
        metrics=['accuracy']
    )

    model.summary()

    return model


def generate_notes(model):
    data = pickle.load(open("notes_tbt_classical.n", "rb"))
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(np.array(data))

    input_data, output_data = utils.prepare_seq(data, SEQ_LEN)

    X = input_data.reshape((input_data.shape[0], 1, input_data.shape[1]))
    y = output_data.reshape((len(output_data), 1))

    print("Shape y", y.shape)

    trainPredict = model.predict(X)
    trainPredict = scaler.inverse_transform(trainPredict)
    #
    print(np.array(trainPredict,dtype=int))

    # plt.show()
    # print(trainY)


if __name__ == '__main__':
    generate()
