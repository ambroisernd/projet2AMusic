import pickle

from music21 import *
import glob

from utils.math_utils import RepresentsInt, parse_duration


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
                ch = '$'.join(str(n) for n in _note.normalOrder)
                d = str(_note.duration)[:-1].split()[-1]
                notes.append(ch + "$" + d)
            elif isinstance(_note, note.Rest):
                d = str(_note.duration)[:-1].split()[-1]
                notes.append('S'+" "+d)

    with open(notes_save_path, 'wb') as f_path:
        pickle.dump(notes, f_path)

    return notes


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

def choose_notes(notes_to_ix):
    print(notes_to_ix)
    print("choose the indice of the note or the chord you want")
    indice = input()
    return indice

#voc_path = "../data/vocabularies/easy_64"
#with open(voc_path, 'rb') as fp:
#    voc = pickle.load(fp)
#notes_to_ix = {n: i for i, n in enumerate(voc)}
#choose_notes(notes_to_ix)