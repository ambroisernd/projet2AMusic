from keras.engine.saving import load_model

from midi_utils import generate_notes, generate_midi_file
from utils import *


def generate_music():
    with open(voc_path, 'rb') as fp:
        voc = pickle.load(fp)
    notes_to_ix = {n: i for i, n in enumerate(voc)}
    ix_to_notes = {i: n for i, n in enumerate(voc)}
    model = load_model(weights_path)
    generated_indices = predict_and_sample_random(model, notes_to_ix)
    to_play = generate_notes(generated_indices, ix_to_notes)
    generate_midi_file(output_path, to_play)


def predict_and_sample_random(model, notes_to_ix):
    notes = []

    for x in range(n_notes_before):
        notes.append(random.randint(0, len(notes_to_ix)))
    X = notes[:]
    indices = []
    for i in range(Ty):
        X_in = np.reshape(X, (1, len(notes), 1))
        pred = model.predict(X_in, verbose=1)
        idx = np.random.choice([k for k in range(len(notes_to_ix))], p=pred.ravel())
        indices.append(idx)
        X.append(idx)
        X = X.pop(0)
    return indices


if __name__ == "__main__":
    # execute only if run as a script
    n_notes_before = 20
    Ty = 500
    output_path = 'test1.mid'
    weights_path = 'models/lstm_model.h5'
    voc_path = 'data/voc'

    generate_music()
