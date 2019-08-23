import pickle

import numpy as np
from keras.utils import to_categorical
from utils.math_utils import RepresentsInt, parse_duration
pb  = []

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
    X = X / float(len(notes_to_index))
    Y = to_categorical(Y)
    return X, Y


def generate_X_Y_one_hot(notes_to_index, notes, n_notes):
    """Generate one-hot vectors X and Y for training where X[i+n_notes]=Y[i]"""
    X = []
    Y = []
    for i in range(len(notes) - n_notes):
        note_X = notes[i: i + n_notes]
        note_Y = notes[i + n_notes]
        X.append([note_to_one_hot(c, notes_to_index) for c in note_X])
        Y.append(note_to_one_hot(note_Y, notes_to_index))
    X = np.reshape(X, (len(X), n_notes, len(notes_to_index)))
    return np.array(X), np.array(Y)


def note_to_one_hot(note, notes_to_index):
    X = []
    problem = []
    try:
        if '$' in note:
            ch = note.split("$")
            notes = ch[:-1]
            duration = parse_duration(ch[-1])
            X.append(len(notes))
            for c in notes:
                X.append(notes_to_index[c])

            X.append(notes_to_index[duration])
        elif note[0] == 'S':
            duration = parse_duration(note.split()[-1])
            X.append(0)
            X.append(notes_to_index[duration])
        else:
            notes = note.split()[0]
            duration = parse_duration(note.split()[-1])
            X.append(1)
            X.append(notes_to_index[notes])
            X.append(notes_to_index[duration])
    except Exception as e:
        pb.append(duration)
        print(e)

    return one_hot(X, notes_to_index)


def one_hot(indice_array, notes_to_index):
    X = [0 for k in range(len(notes_to_index))]
    for i in indice_array:
        X[i] = 1
    return X


def generate_vocab(notes, voc_save_path):
    """Generate vocabulary based on the input notes"""
    voc = np.unique(np.array(notes))
    with open(voc_save_path, 'wb') as f_path:
        pickle.dump(voc, f_path)
    return voc
