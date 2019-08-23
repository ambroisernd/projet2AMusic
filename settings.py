n_notes_before = 20  # sequence length

# Train
resume_model = True  # resume training : True / override and start new training : False
path_to_midi = 'training_data/easy/*.mid'  # path to input midi files
notes_save_path = 'data/_notes/maestro2018'  # file path to save notes parsed from input midi files
notes_load_path = 'data/_notes/maestro2018'  # file path to load notes previously parsed to resume training
epochs = 10000
batch_size = 64
weights_save_path = 'data/models/onehotmaestro2018.h5'  # file path to save model weights
weights_load_path = 'data/models/onehotmaestro2018.h5'  # file path to load model weights

# Predict
output_path = 'generated_midi/onehoteasy5.mid'  # midi output path and file name
weights_path = 'data/models/onehoteasy.h5'  # file path to load model weights
notes_path = 'data/_notes/onehoteasy'  # file path to load notes previously parsed in train.py
midi_path_to_continue = 'midiToContinue/riff.mid'
Ty = 500  # notes to generate

random_notes = False
pick_from_training_data = False
choose_n_notes_before = False
continue_midi_file = True
