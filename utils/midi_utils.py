import pickle

from music21 import *
import glob

from utils.math_utils import RepresentsInt, parse_duration
from utils.preprocessing import note_to_one_hot


def get_notes(path_to_midi, notes_save_path):
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
                    print("adding : " + str(part))
        except:
            instr_part.append(midi.flat)

    for e in instr_part:
        for _note in e.recurse().notes:
            if isinstance(_note, note.Note):
                d = str(_note.duration)[:-1].split()[-1]
                notes.append((str(_note.pitch) + " " + d))
            elif isinstance(_note, chord.Chord):
                print(_note.pitches)
                ch = ""
                for x in _note:
                    ch += str(x.pitch).split()[-1]
                    ch += "$"
                d = str(_note.duration)[:-1].split()[-1]
                notes.append(ch + d)
            elif isinstance(_note, note.Rest):
                d = str(_note.duration)[:-1].split()[-1]
                notes.append('S' + " " + d)

    with open(notes_save_path, 'wb') as f_path:
        pickle.dump(notes, f_path)

    return notes


def create_vocab_array():
    vocab = ['rest', 'note', 'chord2', 'chord3', 'chord4']
    piano_notes = []
    octave_min = 1
    octave_max = 8
    notes = ['A-', 'A', 'A#', 'B-', 'B', 'C', 'C#', 'D-', 'D', 'D#', 'E-', 'E', 'F', 'F#', 'G-', 'G', 'G#']
    no_dot_duration = [64.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625,
                       0.015625 / 2.0, 0.015625 / 4.0, 0.015625 / 8.0, 0.0]
    for i in range(octave_min, octave_max + 1):
        for x in notes:
            piano_notes.append(x + str(i))
    piano_notes = piano_notes[1:-11]
    vocab = piano_notes + no_dot_duration

    for x in no_dot_duration:
        if x != 0:
            vocab.append(x + (1 / 2) * x)
    for x in no_dot_duration:
        if x != 0:
            vocab.append(x + (1 / 2) * x + (1 / 4) * x)
    for x in no_dot_duration:
        if x != 0:
            vocab.append(x + (1 / 2) * x + (1 / 4) * x + (1 / 8) * x)
    vocab.append(1 / 3)
    vocab.append(2 / 3)
    vocab.append(4 / 3)
    return vocab


def generate_midi_file(output_name, notes):
    """parse notes using separators to generate midi file"""
    sheet = stream.Stream()
    for x in notes:
        if RepresentsInt(x[0]):
            ch = x.split("$")
            sheet.append(chord.Chord([int(k) for k in ch[:-1]], quarterLength=parse_duration(ch[-1])))
        elif x[0] == 'S':
            sheet.append(note.Rest(quarterLength=parse_duration(x.split()[-1])))
        else:
            sheet.append(note.Note(x.split()[0], quarterLength=parse_duration(x.split()[-1])))
    mf = midi.translate.streamToMidiFile(sheet)
    mf.open(output_name, "wb")
    mf.write()
    mf.close()


def generate_notes(indices, ix_to_notes):
    """convert an indice array to a not array using vocabulary"""
    to_play = []
    for x in indices:
        to_play.append(ix_to_notes[x])
    return to_play


def generate_midi_from_one_hots(output_name, one_hots, ix_to_notes):
    sheet = stream.Stream()
    for x in one_hots:
        duration = 0
        for i in range(124, len(x)):
            if x[i] == 1:
                duration = ix_to_notes[i]
        if x[0] == 1:
            sheet.append(note.Rest(quarterLength=duration))
        elif x[1] == 1:
            note_indice = 0
            for i in range(5, 124):
                if x[i] == 1:
                    note_indice = i
            sheet.append(note.Note(ix_to_notes[note_indice], quarterLength=duration))
        else:
            chord_notes = []
            chord_length = 0
            for i in range(0, 5):
                if x[i] == 1:
                    chord_length = i
            while chord_length > 0:
                for i in range(5, 124):
                    if x[i] == 1:
                        chord_notes.append(ix_to_notes[i])
                        chord_length -= 1
            sheet.append(chord.Chord(chord_notes, quarterLength=duration))
    mf = midi.translate.streamToMidiFile(sheet)
    mf.open(output_name, "wb")
    mf.write()
    mf.close()
