from __future__ import print_function
import tensorflow as tf
from pandas import DataFrame
import numpy as np
from utils import *
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io

"""Vocabulary"""
notes = get_notes("e/*.mid")
voc = generate_vocab(notes)
notes_to_ix = {n: i for i, n in enumerate(voc)}
ix_to_notes = {i: n for i, n in enumerate(voc)}

"""Inputs / Params"""
m = 60
Tx = 30
note_to_generate = 100
n_values = len(ix_to_notes)

print('Vectorization...')
X, Y = generate_X_Y_from_one_music(notes_to_ix, notes, Tx, m)

print('Building Model')
model = None

generate_midi_file("keras1.mid", generate_music(model, ix_to_notes))
