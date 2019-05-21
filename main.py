from __future__ import print_function
import tensorflow as tf
from utils import *
import IPython
import sys
import numpy as np
from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K

notes = get_notes("e/*.mid")
voc = generate_vocab(notes)
notes_to_ix = {n: i for i, n in enumerate(voc)}
ix_to_notes = {i: n for i, n in enumerate(voc)}
n_values = len(ix_to_notes)
print(voc)
X, Y = generate_X_Y_from_one_music(notes_to_ix, notes, 60, 100)
print(X.shape)
print(Y.shape)

Tx = X.shape[1]
Ty = 200

n_a = 64
reshapor = Reshape((1, n_values))
LSTM_cell = LSTM(n_a, return_state=True, dropout=0.2)
densor = Dense(n_values, activation='softmax')


def music_model(Tx, n_a, n_values):
    """
    Implement the model

    Arguments:
    Tx -- length of the sequence in a corpus
    n_a -- the number of activations used in our model
    n_values -- number of unique values in the music data

    Returns:
    model -- a keras model with the
    """
    X = Input(shape=(Tx, n_values))
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    outputs = []
    for t in range(Tx):
        x = Lambda(lambda x: X[:, t, :])(X)
        x = reshapor(x)
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        out = densor(a)
        outputs.append(out)
    model = Model([X, a0, c0], outputs)
    return model


model = music_model(Tx=Tx, n_a=n_a, n_values=n_values)

opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

m = X.shape[0]
a0 = np.zeros((m, n_a))
c0 = np.zeros((m, n_a))
model.fit([X, a0, c0], list(Y), epochs=500)


def one_hot(x):
    x = K.argmax(x)
    x = tf.one_hot(x, n_values)
    x = RepeatVector(1)(x)
    return x


def music_inference_model(LSTM_cell, densor, n_values=n_values, n_a=64, Ty=100):
    """
    Uses the trained "LSTM_cell" and "densor" from model() to generate a sequence of values.

    Arguments:
    LSTM_cell -- the trained "LSTM_cell" from model(), Keras layer object
    densor -- the trained "densor" from model(), Keras layer object
    n_values -- integer, umber of unique values
    n_a -- number of units in the LSTM_cell
    Ty -- integer, number of time steps to generate

    Returns:
    inference_model -- Keras model instance
    """
    x0 = Input(shape=(1, n_values))
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    x = x0
    outputs = []
    for t in range(Ty):
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        out = densor(a)
        outputs.append(out)
        x = Lambda(one_hot)(out)
    inference_model = Model([x0, a0, c0], outputs)
    return inference_model


inference_model = music_inference_model(LSTM_cell, densor, n_values=n_values, n_a=n_a, Ty=Ty)

x_initializer = np.zeros((1, 1, n_values))
#x_initializer[0, 0, 1] = 1  # initialise a la premiere note
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))


def predict_and_sample(inference_model, x_initializer=x_initializer, a_initializer=a_initializer,
                       c_initializer=c_initializer):
    """
    Predicts the next value of values using the inference model.

    Arguments:
    inference_model -- Keras model instance for inference time
    x_initializer -- numpy array of shape (1, 1, 78), one-hot vector initializing the values generation
    a_initializer -- numpy array of shape (1, n_a), initializing the hidden state of the LSTM_cell
    c_initializer -- numpy array of shape (1, n_a), initializing the cell state of the LSTM_cel

    Returns:
    results -- numpy-array of shape (Ty, 78), matrix of one-hot vectors representing the values generated
    indices -- numpy-array of shape (Ty, 1), matrix of indices representing the values generated
    """
    indices = []
    pred = inference_model.predict([x_initializer, a_initializer, c_initializer])
    for i in range(Ty):
        indices.append(np.random.choice([k for k in range(n_values)], p=pred[i].ravel()))
    results = to_categorical(indices)
    return results, indices


def generate_music():
    _, indices = predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer)
    to_play = []
    for x in indices:
        to_play.append(ix_to_notes[x])
    return to_play


generate_midi_file("maestro4.mid", generate_music())
