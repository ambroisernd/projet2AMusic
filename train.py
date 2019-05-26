import tensorflow
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation

from midi_utils import get_notes
from preprocessing import generate_vocab, generate_X_Y_multi


def train_lstm():
    notes = get_notes(path_to_midi)
    voc = generate_vocab(notes, voc_save_path)
    notes_to_ix = {n: i for i, n in enumerate(voc)}
    ix_to_notes = {i: n for i, n in enumerate(voc)}
    n_values = len(ix_to_notes)
    X, Y = generate_X_Y_multi(notes_to_ix, notes, n_notes_before)

    model = lstm(X, n_values)

    generate_weights(X, Y, model)


def lstm(X, n_values):
    """TODO: Paul doit construire notre propre model"""
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(X.shape[1], X.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_values))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model


def generate_weights(X, Y, model):
    model.fit(X, Y, epochs=20, batch_size=32)
    model.save(weights_save_path)


if __name__ == "__main__":
    # execute only if run as a script
    path_to_midi = 'e/*.mid'
    n_notes_before = 20
    weights_save_path = 'models/lstm_model.h5'
    voc_save_path = 'data/voc'

    train_lstm()
