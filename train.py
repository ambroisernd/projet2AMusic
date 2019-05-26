from keras import Sequential
from keras.layers import LSTM, Dense, Activation

from utils.midi_utils import get_notes
from utils.preprocessing import generate_vocab, generate_X_Y_multi


def train_lstm():
    notes = get_notes(path_to_midi, notes_save_path)
    voc = generate_vocab(notes, voc_save_path)
    notes_to_ix = {n: i for i, n in enumerate(voc)}
    ix_to_notes = {i: n for i, n in enumerate(voc)}
    n_values = len(ix_to_notes)
    X, Y = generate_X_Y_multi(notes_to_ix, notes, n_notes_before)

    model = lstm(X, n_values)

    generate_weights(X, Y, model)


def lstm(X, n_values):
    """Build an LSTM RNN"""
    """TODO: Paul doit construire notre propre model"""
    model = Sequential()
    model.add(LSTM(
        64,
        input_shape=(X.shape[1], X.shape[2]),
        return_sequences=False
    ))
    model.add(Dense(n_values))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model


def generate_weights(X, Y, model):
    """save weights to weights_save_path"""
    model.fit(X, Y, epochs=epochs, batch_size=batch_size)
    model.save(weights_save_path)


if __name__ == "__main__":
    # execute only if run as a script
    path_to_midi = 'training_data/e/*.mid'
    notes_save_path = 'data/_notes/notes'
    n_notes_before = 50
    epochs = 500
    batch_size = 64
    weights_save_path = 'data/models/elise_500_epochs_50_note.h5'
    voc_save_path = 'data/vocabularies/my_midi_voc'

    train_lstm()
