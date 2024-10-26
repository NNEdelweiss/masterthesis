from EEG_Models import EEGTCNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
import os
import re
from glob import glob
import time
from tqdm import tqdm
import logging
import numpy as np
import pandas as pd
import mne
from mne.io import read_raw_edf
from sklearn.preprocessing import LabelEncoder
from scipy import io, signal
import tensorflow as tf
import tensorflow.keras.utils as np_utils # type: ignore
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from scipy.signal import resample, butter, filtfilt, lfilter, iirnotch 

class BCICIV2aLoader_EEGTCNet:
    def __init__(self, filepath):
        self.filepath = filepath
        self.stimcodes = ['769', '770', '771', '772']
        self.sample_freq = None
        self.batch_size = 64

    def load_data(self, filename):
        gdf_name = filename.split(".")[0]
        raw_data = mne.io.read_raw_gdf(os.path.join(self.filepath, filename), preload=True, eog=['EOG-left', 'EOG-central', 'EOG-right'])
        raw_data.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])
        self.sample_freq = int(raw_data.info['sfreq'])
        before_trial = int(0.5 * self.sample_freq)
        data = raw_data.get_data() 

        logging.info(f"Loading data from {filename}...")
        
        if "T" in gdf_name:
            return self._process_training_data(raw_data, data, before_trial)
        elif "E" in gdf_name:
            return self._process_evaluation_data(raw_data, data, gdf_name, before_trial)
        else:
            raise ValueError(f"Unknown file format for {filename}")

    def _process_training_data(self, raw_data, data, before_trial):
        trials, labels = [], []
        for annotation in raw_data.annotations:
            description = annotation['description']
            onset = annotation['onset']
            onset_idx = int(onset * self.sample_freq)
            if description in self.stimcodes:
                trial = data[:, onset_idx - before_trial:onset_idx + int(4 * self.sample_freq)]
                label = int(description)
                trials.append(trial)
                labels.append(label)
        labels = self._get_labels(np.array(labels))
        logging.info("Training data loaded successfully")
        return np.array(trials), labels

    def _process_evaluation_data(self, raw_data, data, gdf_name, before_trial):
        trials = []
        for annotation in raw_data.annotations:
            if annotation['description'] == "783":
                onset_idx = int(annotation['onset'] * self.sample_freq)
                trial = data[:, onset_idx - before_trial:onset_idx + int(4 * self.sample_freq)]
                trials.append(trial)
        try:
            labels = io.loadmat(os.path.join(self.filepath, "true_labels", gdf_name + ".mat"))["classlabel"][:, 0] - 1
        except FileNotFoundError:
            raise FileNotFoundError(f"Label file for {gdf_name} not found in 'true_labels' directory.")
        labels = np_utils.to_categorical(labels, num_classes=4)
        logging.info("Testing data loaded successfully")
        return np.array(trials), labels

    def _get_labels(self, labels):
        unique_labels = np.sort(np.unique(labels))
        label_map = {old: new for new, old in enumerate(unique_labels)}
        mapped_labels = np.vectorize(label_map.get)(labels)
        return np.eye(len(unique_labels))[mapped_labels.astype(int)]

    def load_dataset(self):
        """
        Load and preprocess dataset from files for all subjects.
        
        Returns:
            dict: Dictionary containing training and testing datasets and labels for each subject.
        """
        filenames = [name for name in os.listdir(self.filepath) if name.endswith(".gdf")]
        eeg_data = {}

        for filename in filenames:
            logging.info(f"Loading data from {filename}...")
            gdf_name = filename.split(".")[0]
            subject = gdf_name[1:3]  # Extracts subject number from filename
            trials, labels = self.load_data(filename)
            logging.info(f"Trial shape: {trials.shape}, Label shape: {labels.shape}")

            if subject not in eeg_data:
                eeg_data[subject] = {}
            if "T" in gdf_name:
                eeg_data[subject]["X_train"] = trials
                eeg_data[subject]["y_train"] = labels
            elif "E" in gdf_name:
                eeg_data[subject]["X_test"] = trials
                eeg_data[subject]["y_test"] = labels
        return eeg_data
    

filepath = '../Dataset/BCICIV_2a_gdf/'
data_loader = BCICIV2aLoader_EEGTCNet(filepath)

# Load the dataset for all subjects
eeg_data = data_loader.load_dataset()
print("Available subjects in eeg_data:", eeg_data.keys())

F1 = 8
KE = 32
KT = 4
L = 2
FT = 12
pe = 0.2
pt = 0.3
classes = 4
channels = 22
crossValidation = False
batch_size = 64
epochs = 750
lr = 0.001

for subject, datasets in eeg_data.items():
    X_train = datasets.get('X_train')
    y_train = datasets.get('y_train')
    X_test = datasets.get('X_test')
    y_test = datasets.get('y_test')

    # path = data_path+'s{:}/'.format(subject+1)
    # X_train,y_train,y_train_onehot,X_test,y_test,y_test_onehot = prepare_features(data_path,subject,crossValidation)
    print("Training model...")
    # model = DeepConvNet(nb_classes = 4,nchan=22, trial_length=1125)    
    model = EEGTCNet(nb_classes = 4,nchan=22, trial_length=1125, layers=L, kernel_s=KT,filt=FT, dropout=pt, activation='elu', F1=F1, D=2, kernLength=KE, dropout_eeg=pe)
    for j in range(22):
        scaler = StandardScaler()
        scaler.fit(X_train[:,0,j,:])
        X_train[:,0,j,:] = scaler.transform(X_train[:,0,j,:])
        X_test[:,0,j,:] = scaler.transform(X_test[:,0,j,:])

    model.fit(X_train, y_train, batch_size=batch_size, epochs=750, verbose=1)

    y_pred = model.predict(X_test).argmax(axis=-1)
    labels = y_test.argmax(axis=-1)
    accuracy_of_test = accuracy_score(labels, y_pred)
    with open('results_nhi_test_again_v2.txt', 'a') as f:
        f.write('Subject: {:} Accuracy: {:}\n'.format(subject,accuracy_of_test))
    print(accuracy_of_test)
