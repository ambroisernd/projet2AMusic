import pickle

import numpy as np
from keras.utils import to_categorical


def generate_X_Y_note_to_note(note_to_index, notes, Tx, m):
    """Generate vectors X and Y for training where X[i+1]=Y[i]"""
    N_values = len(note_to_index)
    np.random.seed(0)
    X = np.zeros((m, Tx, N_values), dtype=np.bool)
    Y = np.zeros((m, Tx, N_values), dtype=np.bool)
    for i in range(m):
        random_idx = np.random.choice(len(notes) - Tx)
        notes_data = notes[random_idx:(random_idx + Tx)]
        for j in range(Tx):
            idx = note_to_index[notes_data[j]]
            if j != 0:
                X[i, j, idx] = 1
                Y[i, j - 1, idx] = 1

    # Y = np.swapaxes(Y, 0, 1)
    Y = Y.tolist()
    return np.asarray(X), np.asarray(Y)


def generate_X_Y_multi(notes_to_index, notes, n_notes):
    """Generate vectors X and Y for training where X[i+n_notes]=Y[i]"""
    X = []
    Y = []
    for i in range(len(notes) - n_notes):
        note_X = notes[i: i + n_notes]
        note_Y = notes[i + n_notes]
        X.append([notes_to_index[c] for c in note_X])
        Y.append(notes_to_index[note_Y])
    X = np.reshape(X, (len(X), n_notes, 1))
    Y = to_categorical(Y)
    return X, Y


def generate_vocab(notes, voc_save_path):
    """Generate vocabulary based on the input notes"""
    voc = np.unique(np.array(notes))
    with open(voc_save_path, 'wb') as f_path:
        pickle.dump(voc, f_path)
    return voc
