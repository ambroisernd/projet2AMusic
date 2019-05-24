import glob

from keras.utils import to_categorical
from music21 import *
import numpy as np
import glob


def get_notes(path_to_midi):
    """Parse notes from midifile"""
    notes = []
    instr_part = []
    instr = instrument.Piano
    print("Start parsing")
    for file in glob.glob(path_to_midi):
        print("Parsing %s" % file)
        midi = converter.parse(file)
        try:
            for part in instrument.partitionByInstrument(midi):
                print(part)
                if isinstance(part.getInstrument(), instr):
                    instr_part.append(part)
        except:
            instr_part.append(midi.flat)

    for e in instr_part:
        for _note in e.recurse().notes:
            if isinstance(_note, note.Note):
                d = str(_note.duration)[:-1].split()[-1]
                notes.append((str(_note.pitch) + " " + d))
            elif isinstance(_note, chord.Chord):
                # for n in element.pitches:
                #   print(str(n))
                ch = '$'.join(str(n) for n in _note.normalOrder)
                d = str(_note.duration)[:-1].split()[-1]
                notes.append(ch + "$" + d)
    print(notes)
    return notes


def parse_duration(s):
    """return string duration to float"""
    if "/" in s:
        return float(s.split("/")[0]) / float(s.split("/")[-1])
    else:
        return float(s)


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


def generate_X_Y_from_one_music(note_to_index, notes, Tx, m):
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


def RepresentsInt(s):
    """helper fct to check if string is int"""
    try:
        int(s)
        return True
    except ValueError:
        return False


def generate_midi_file(output_name, notes):
    sheet = stream.Stream()
    for x in notes:
        if RepresentsInt(x[0]):
            ch = x.split("$")
            sheet.append(chord.Chord([int(k) for k in ch[:-1]], quarterLength=parse_duration(ch[-1])))
        else:
            sheet.append(note.Note(x.split()[0], quarterLength=parse_duration(x.split()[-1])))
    mf = midi.translate.streamToMidiFile(sheet)
    mf.open(output_name, "wb")
    mf.write()
    mf.close()


def predict_and_sample(inference_model, Ty, n_values):
    indices = []
    pred = inference_model.predict(np.zeros((1, 1, n_values)))
    for i in range(Ty):
        indices.append(np.random.choice([k for k in range(n_values)], p=pred[i].ravel()))
    results = to_categorical(indices)
    return results, indices


def generate_music(model, ix_to_notes, note_to_generate, n_values):
    _, indices = predict_and_sample(model, note_to_generate, n_values)
    to_play = []
    for x in indices:
        to_play.append(ix_to_notes[x])
    return to_play
