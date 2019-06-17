import pickle
import random
import numpy as np

from keras.engine.saving import load_model

from utils.midi_utils import *
from utils.preprocessing import *


def generate_music():
    """load the vocabulary and the model to create a midi file"""
    # with open(voc_path, 'rb') as fp:
    #     voc = pickle.load(fp)
    voc = create_vocab_array()
    notes_to_ix = {n: i for i, n in enumerate(voc)}
    ix_to_notes = {i: n for i, n in enumerate(voc)}
    model = load_model(weights_path)
    prediction = predict_and_sample_random(model, notes_to_ix)
    generate_midi_from_one_hots(output_path, prediction, ix_to_notes)


def predict_and_sample_random(model, notes_to_ix):
    print(notes_to_ix)
    with open(notes_path, 'rb') as fp:
        notes = pickle.load(fp)
    """generate n = n_notes_before random notes to predict the Ty-th following notes"""
    # notes = []
    #
    # for x in range(n_notes_before):
    #     notes.append(choose_random_note(notes_to_ix))
    # X = notes[:]
    """-------------------------------------------------------------------------------"""
    """pick n = n_notes_before  from input files to predict the Ty-th following notes"""
    # rnd = random.randint(0, len(notes) - random.randint(0, len(notes_to_ix) - 1))
    # notes = notes[rnd:rnd + n_notes_before]
    # X = []
    # for n in notes:
    #     X.append(note_to_one_hot(n, notes_to_ix))
    """-------------------------------------------------------------------------------"""
    """Choose your n_notes before"""
    # notes = []
    # for i in range(n_notes_before):
    #     print(str(n_notes_before-i)+" notes or chord to choose")
    #     notes.append(choose_notes(notes_to_ix))
    # X=notes[:]
    """-------------------------------------------------------------------------------"""
    """Enter a midi, and the ia will continue the music"""
    notes = get_notes(midi_path_to_continue, 'midiToContinue/for_test')
    X=[]
    for note in notes:
        n = note_to_one_hot(note, notes_to_ix)
        X.append(n)
    X=X[-20:]
    print(len(X))
    """-------------------------------------------------------------------------------"""
    one_hots = X[:]
    for i in range(Ty):
        X_in = np.reshape(np.array(X), (1, len(X), len(notes_to_ix)))
        pred = model.predict(X_in, verbose=1)
        note_type = np.argmax(pred[0][:5]) + 0
        note_value = []
        if note_type != 0:
            for i in range(note_type):
                max = np.argmax(pred[0][5:124]) + 5
                note_value.append(max)
                pred[0][max] = 0
        note_duration = np.argmax(pred[0][124:]) + 124
        Y = [0 for i in range(len(notes_to_ix))]
        Y[note_type] = 1
        Y[note_duration] = 1
        for x in note_value:
            Y[x] = 1
        one_hots.append(Y)
        X.append(Y)
        X = X[1:len(X)]
    return one_hots


if __name__ == "__main__":
    # execute only if run as a script
    n_notes_before = 20  # must be the same as train.py
    Ty = 500  # notes to generate
    output_path = 'generated_midi/onehoteasy5.mid'  # midi output path and file name
    weights_path = 'data/models/onehoteasy.h5'  # file path to load model weights
    notes_path = 'data/_notes/onehoteasy'  # file path to load notes previously parsed in train.py
    voc_path = 'data/vocabularies/onehoteasy'  # file path to load vocabulary created in train.py
    midi_path_to_continue = 'midiToContinue/riff.mid'

    generate_music()
