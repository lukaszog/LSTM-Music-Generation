from keras.models import load_model
import numpy as np
import utils


def generate():
    model = create_network('results')
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)


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

def generate_notes(model, network_input, pitchnames, n_vocab):
    pass

if __name__ == '__main__':
    generate()



