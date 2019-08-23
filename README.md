# Génération de musique par approche neuronale

## Comment l'utiliser

Paramétrer via settings

settings.py :

```python
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

##Predict used, select one
random_notes = False
pick_from_training_data = False
choose_n_notes_before = False
continue_midi_file = True
```

Il vous suffit de modifier les variables afin d'utiliser les options et les modèles que vous souhaitez

Un modèle est déjà entrainé pour vous faciliter la première utilisation : data/models/onehoteasy.h5

Renseignez ensuite le output_path, il s'agira du fichier généré par la prédiction

midi_path_to_continue est utilisé si vous sélectionnez continue_midi_file

Ty est le nombre de notes que vous souhaitez générer
Une première génération

Une fois les variables définis comme convenues, il suffit d'exécuter le script predict.py

```python
if __name__ == "__main__":
    # execute only if run as a script
    generate_music()
```

**Des erreurs sont survenues ?**

Dans ce modèle il est nécessaire d'avoir installé

1/ CUDA toolkit : <https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Windows&target_arch=x86_64>

2/ CUDNN : <https://developer.nvidia.com/rdp/cudnn-download>

3/ Avoir installé keras : <https://keras.io/>

4/ Avoir installé tensorflow-gpu : <https://www.tensorflow.org/install/gpu>
