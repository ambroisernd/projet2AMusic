import pickle

from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.engine.saving import load_model
from keras.layers import LSTM, Dense, Activation, CuDNNLSTM, Dropout

from utils.midi_utils import get_notes
from utils.preprocessing import generate_vocab, generate_X_Y_multi


def train_lstm():
    if resume_model:
        with open(notes_load_path, 'rb') as fp:
            notes = pickle.load(fp)
    else:
        notes = get_notes(path_to_midi, notes_save_path)
    voc = generate_vocab(notes, voc_save_path)
    notes_to_ix = {n: i for i, n in enumerate(voc)}
    ix_to_notes = {i: n for i, n in enumerate(voc)}
    n_values = len(ix_to_notes)
    X, Y = generate_X_Y_multi(notes_to_ix, notes, n_notes_before)
    if resume_model:
        model = load_model(weights_load_path)
    else:
        model = lstm(X, n_values)

    generate_weights(X, Y, model)


def lstm(X, n_values):
    """Build an LSTM RNN"""
    """TODO: Paul doit construire notre propre model"""
    model = Sequential()
    model.add(CuDNNLSTM(
        512,
        input_shape=(X.shape[1], X.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(CuDNNLSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(CuDNNLSTM(512))
    model.add(Dense(256))
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
    model.fit(X, Y, epochs=epochs, callbacks=callbacks_list, batch_size=batch_size, verbose=2)


if __name__ == "__main__":
    # execute only if run as a script
    path_to_midi = 'training_data/e/*.mid'
    notes_save_path = 'data/_notes/notes'
    notes_load_path = 'data/_notes/notes'
    n_notes_before = 500
    epochs = 1000
    batch_size = 2048
    weights_save_path = 'data/models/my_model.h5'
    weights_load_path = 'data/models/my_model.h5'
    voc_save_path = 'data/vocabularies/my_midi_voc'

    resume_model = False

    train_lstm()
