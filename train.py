import pickle

from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.engine.saving import load_model
from keras.layers import Dense, Activation, CuDNNLSTM, Dropout

from utils.midi_utils import get_notes, create_vocab_array
from utils.preprocessing import generate_X_Y_one_hot
from utils.preprocessing import pb
from utils.math_utils import unique
from settings import *


def train_lstm():
    if resume_model:
        with open(notes_load_path, 'rb') as fp:
            notes = pickle.load(fp)
    else:
        notes = get_notes(path_to_midi, notes_save_path)
    voc = create_vocab_array()
    notes_to_ix = {n: i for i, n in enumerate(voc)}
    ix_to_notes = {i: n for i, n in enumerate(voc)}
    n_values = len(ix_to_notes)
    X, Y = generate_X_Y_one_hot(notes_to_ix, notes, n_notes_before)
    #print(pb)
    #print({n: i for i, n in enumerate(pb)})
    #print(enumerate(pb))
    print(unique(pb))
    if resume_model:
        model = load_model(weights_load_path)
    else:
        model = lstm(X, n_values)

    generate_weights(X, Y, model)


def lstm(X, n_values):
    """Build an LSTM RNN"""
    model = Sequential()
    model.add(CuDNNLSTM(
        512,
        input_shape=(X.shape[1], X.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(CuDNNLSTM(512, return_sequences=True)) # use keras.layers.LSTM if running on CPU
    model.add(Dropout(0.3))
    model.add(CuDNNLSTM(512))
    model.add(Dense(512 // 2))
    model.add(Dropout(0.3))
    model.add(Dense(n_values))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model


def generate_weights(X, Y, model):
    """save weights to weights_save_path"""
    filepath = weights_save_path
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    model.fit(X, Y, epochs=epochs, callbacks=callbacks_list, batch_size=batch_size, verbose=1)


if __name__ == "__main__":
    # execute only if run as a script
    train_lstm()
