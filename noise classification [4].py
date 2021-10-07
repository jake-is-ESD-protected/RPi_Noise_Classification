import tensorflow as tf             
# pip3 install tensorflow installs the wrong version and "==2.2.0" results in an error
# workaround:
#
# pip3 install gdown
# gdown https://drive.google.com/uc?id=11mujzVaFqa7R1_lB7q0kVPW22Ol51MPg
# pip3 install tensorflow-2.2.0-cp37-cp37m-linux_armv7l.whl

import tensorflow_hub as hub        # pip3 install --upgrade tensorflow-hub
import numpy as np                  
import csv
import librosa as lr
# "pip3 install librosa" fails to build llvmlite-wheel
# workaround is to build it from source:
# 
# pip3 install scipy (might be already satisfied)
# pip3 install Cython (might be already satisfied)
# wget https://github.com/librosa/librosa/archive/refs/tags/0.8.1.tar.gz
# tar xzf 0.8.1.tar.gz
# cd librosa-0.8.1
# python setup.py install
# compilation may take about 15 min
# be sure to set python 3.7 as interpreter

from pathlib import Path
import os

print(f'Tensor Flow on version ' + tf.__version__)

def class_names_from_csv(class_map_csv_text):
#   Returns list of class names corresponding to score vector
    _class_names = []
    with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
        reader = csv.DictReader(csvfile)
    for row in reader:
        _class_names.append(row['display_name'])

    return _class_names

model = hub.load('/home/pi/Workspace/NTi_Noise_Classification/yamnet_1/')
class_map_path = model.class_map_path().numpy()
class_names = class_names_from_csv(class_map_path)

wav_dir_name = '/home/pi/Workspace/NTi_Noise_Classification/yams_testfiles_custom/'

for wav_name in os.listdir(wav_dir_name):
    if wav_name.endswith(".wav"):

        full_name = os.path.join(wav_dir_name, wav_name)
        wav_data, sample_rate = lr.load(full_name, mono=True, sr=16000, duration=5) 
        # it is important to resample down to 16kHz for specified accuracy of YAMNET

        scores, embeddings, spectrogram = model(wav_data)

        scores_np = scores.numpy()
        spectrogram_np = spectrogram.numpy()
        infered_class = class_names[scores_np.mean(axis=0).argmax()]

        print(Path(full_name).stem + f' got classified as {infered_class}')

        continue
    else:
        continue