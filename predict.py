import pickle
import random
import numpy as np

from keras.engine.saving import load_model

from utils.midi_utils import generate_notes, generate_midi_file, choose_notes


def generate_music():
    """load the vocabulary and the model to create a midi file"""
    with open(voc_path, 'rb') as fp:
        voc = pickle.load(fp)
    notes_to_ix = {n: i for i, n in enumerate(voc)}
    ix_to_notes = {i: n for i, n in enumerate(voc)}
    model = load_model(weights_path)
    generated_indices = predict_and_sample_random(model, notes_to_ix)
    to_play = generate_notes(generated_indices, ix_to_notes)
    generate_midi_file(output_path, to_play)


def predict_and_sample_random(model, notes_to_ix):
    with open(notes_path, 'rb') as fp:
        notes = pickle.load(fp)
    """generate n = n_notes_before random notes to predict the Ty-th following notes"""
    # notes = []
    #
    # for x in range(n_notes_before):
    #     notes.append(random.randint(0, len(notes_to_ix)))
    # X = notes[:]
    """-------------------------------------------------------------------------------"""
    """pick n = n_notes_before  from input files to predict the Ty-th following notes"""
    #rnd = random.randint(0, len(notes) - random.randint(0, len(notes_to_ix) - 1))
    #notes = notes[rnd:rnd + n_notes_before]
    #X = []
    #for n in notes:
    #    X.append(notes_to_ix[n])
    """-------------------------------------------------------------------------------"""
    """Choose your n_notes before"""
    notes = []
    for i in range(n_notes_before):
        print(str(notes_to_ix-i)+" notes or chord to choose")
        notes.append(choose_notes(notes_to_ix))
    X=notes
    """-------------------------------------------------------------------------------"""
    indices = []
    for i in range(Ty):
        X_in = np.reshape(X, (1, len(X), 1)) / float(len(notes_to_ix))
        pred = model.predict(X_in, verbose=0)
        #        idx = np.random.choice([k for k in range(len(notes_to_ix))], p=pred.ravel())
        idx = np.argmax(pred)
        indices.append(idx)
        X.append(idx)
        X = X[1:len(X)]
    return indices


if __name__ == "__main__":
    # execute only if run as a script
    n_notes_before = 20  # must be the same as train.py
    Ty = 500  # notes to generate
    output_path = 'generated_midi/easychoose_64.mid'  # midi output path and file name
    weights_path = 'data/models/easy_64.h5'  # file path to load model weights
    notes_path = 'data/_notes/easy_64'  # file path to load notes previously parsed in train.py
    voc_path = 'data/vocabularies/easy_64'  # file path to load vocabulary created in train.py

    generate_music()
