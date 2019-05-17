from keras.utils import to_categorical
from music21 import *
import numpy as np

def get_notes():
    """Print notes from midifile"""
    notes = []

    midi = converter.parse("elise.mid")

    s2 = instrument.partitionByInstrument(midi)
    notes_to_parse = s2.parts[0].recurse()
    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            # for n in element.pitches:
            #   print(str(n))
            notes.append('.'.join(str(n) for n in element.normalOrder))

    return notes[:30]


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

def generate_X_Y(name_to_index, notes):
    """Generate vectors X and Y for training where X[i+1]=Y[i]"""
    X = []
    Y = []
    for i in range(len(notes)):
        crt = np.zeros(len(name_to_index))
        crt[name_to_index[notes[i]]] = 1
        X.append(crt)
        if i > 0:
            Y.append(crt)
    Y.append(np.zeros(len(name_to_index)))
    return np.expand_dims(np.array(X), axis=0), np.expand_dims(np.array(Y), axis=1)

#print(generate_X_Y(notes_to_ix, get_notes()))


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

#generate_midi_file("yeayea.mid", get_notes())
