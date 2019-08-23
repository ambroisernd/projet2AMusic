import pickle

from music21 import *
import glob

from utils.math_utils import RepresentsInt, parse_duration
from utils.preprocessing import note_to_one_hot, one_hot
import random as random


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
    octave_min = 0
    octave_max = 8
    notes = ['A-', 'A', 'A#', 'B-', 'B', 'C', 'C#', 'D-', 'D', 'D#', 'E-', 'E', 'F', 'F#', 'G-', 'G', 'G#']
    no_dot_duration = [64.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625,
                       0.015625 / 2.0, 0.015625 / 4.0, 0.015625 / 8.0, 0.0]
    for i in range(octave_min, octave_max + 1):
        for x in notes:
            piano_notes.append(x + str(i))
    piano_notes = piano_notes[1:-11]
    vocab = vocab + piano_notes + no_dot_duration

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
    vocab.append(1.25)
    vocab.append(5.0)
    vocab.append(2.25)
    vocab.append(2.5)
    vocab.append(7/3)
    vocab.append(8/3)
    vocab.append(3.25)
    vocab.append(4.25)
    vocab.append(4.5)
    vocab += [1.6666666666666667, 3.3333333333333335, 3.6666666666666665, 2.75, 5.75, 7.333333333333333, 5.5, 5.25, 4.666666666666667, 4.75, 5.333333333333333, 10.0, 6.666666666666667, 9.75, 5.666666666666667, 4.333333333333333, 6.25, 8.5, 10.25, 6.333333333333333, 6.75, 6.5, 7.75, 9.25, 22.25, 9.5, 11.5, 11.0, 12.25, 12.75, 7.25, 8.25, 21.0, 8.75, 8.666666666666666, 9.0, 10.333333333333334, 8.333333333333334, 7.666666666666667, 10.666666666666666, 10.5, 27.0, 42.0, 9.666666666666666, 16.333333333333332, 13.0, 40.0, 28.75, 27.75, 34.333333333333336, 27.25, 17.0, 9.333333333333334, 13.5, 15.333333333333334, 13.75, 11.75, 13.666666666666666, 10.75, 22.5, 22.0, 11.333333333333334, 11.666666666666666, 12.333333333333334, 12.5, 12.666666666666666, 14.666666666666666, 18.666666666666668, 17.5, 11.25, 16.25, 14.5, 13.333333333333334, 13.25, 16.666666666666668, 15.75, 14.75, 19.666666666666668, 24.5, 24.75, 25.666666666666668, 15.25, 15.5, 21.5, 21.25, 14.333333333333334, 23.5, 15.666666666666666, 16.5, 16.75, 20.0, 19.75, 20.25, 19.333333333333332, 19.25, 19.0, 17.25, 19.5, 35.25, 25.0, 22.75, 17.75, 20.75, 14.25, 18.333333333333332, 18.0, 17.666666666666668, 18.25, 20.5, 20.666666666666668, 20.333333333333332, 21.75, 31.666666666666668, 23.75, 23.25, 22.666666666666668, 22.333333333333332, 25.25, 24.25, 23.666666666666668, 25.75, 24.666666666666668, 62.0, 61.75, 61.5, 51.5, 50.75, 50.333333333333336, 18.75, 23.0, 17.333333333333332, 25.5, 39.25, 28.5, 34.666666666666664]
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


def choose_notes(notes_to_ix):
    print("Choose the indice of the type you want : ")
    print(notes_to_ix)
    type_note = int(input())
    notes = [type_note]
    for i in range(type_note):
        print("Choose the indice of the note you want : ")
        print(notes_to_ix)
        n = int(input())
        notes.append(n)
    print("Choose the indice of the time you want : ")
    print(notes_to_ix)
    time_note = int(input())
    notes.append(time_note)
    return one_hot(notes, notes_to_ix)

def choose_random_note(notes_to_ix):
    type_note = random.randint(0, 4)
    notes=[type_note]
    for i in range(type_note):
        n = random.randint(5,  128)
        while n in notes:
            n = random.randint(5, 128)
        notes.append(n)
    duree = random.randint(133, 136)
    notes.append(duree)
    return one_hot(notes, notes_to_ix)