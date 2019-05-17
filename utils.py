from keras.utils import to_categorical
from music21 import *
import numpy as np

def get_notes(path_to_midi):
    """Print notes from midifile"""
    notes = []

    midi = converter.parse(path_to_midi)

    s2 = instrument.partitionByInstrument(midi)
    notes_to_parse = s2.parts[0].recurse()
    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            # for n in element.pitches:
            #   print(str(n))
            notes.append('.'.join(str(n) for n in element.normalOrder))

    return notes


def generate_vocab(notes):
    """Generate vocabulary based on the input notes"""
    return np.unique(np.array(notes))


"""
# TODO: A mettre dans le main

voc = generate_vocab(get_notes())
notes_to_ix = {n: i for i, n in enumerate(voc)}
ix_to_notes = {i: n for i, n in enumerate(voc)}


# END TODO
"""


def generate_X_Y_from_one_music(name_to_index, notes, Tx, m):
    """Generate vectors X and Y for training where X[i+1]=Y[i]"""
    Tx = Tx
    N_values = len(name_to_index)
    np.random.seed(0)
    X = np.zeros((m, Tx, N_values), dtype=np.bool)
    Y = np.zeros((m, Tx, N_values), dtype=np.bool)
    for i in range(m):
        random_idx = np.random.choice(len(notes) - Tx)
        notes_data = notes[random_idx:(random_idx + Tx)]
        for j in range(Tx):
            idx = name_to_index[notes_data[j]]
            if j != 0:
                X[i, j, idx] = 1
                Y[i, j - 1, idx] = 1

    Y = np.swapaxes(Y, 0, 1)
    Y = Y.tolist()
    return np.asarray(X), np.asarray(Y)


def RepresentsInt(s):
    """helper fct to check if string is int"""
    try:
        int(s)
        return True
    except ValueError:
        return False


def generate_midi_file(outputName, notes):
    sheet = stream.Stream()
    for x in notes:
        if RepresentsInt(x[0]):
            ch = x.split(".")
            sheet.append(chord.Chord([int(k) for k in ch], quarterLength=0.25))
        else:
            sheet.append(note.Note(x, quarterLength=0.25))
    mf = midi.translate.streamToMidiFile(sheet)
    mf.open(outputName, "wb")
    mf.write()
    mf.close()
