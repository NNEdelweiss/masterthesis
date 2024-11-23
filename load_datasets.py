import os
import re
import random
from glob import glob
import time
from tqdm import tqdm
import logging
import numpy as np
import pandas as pd
import mne
from mne.io import read_raw_edf
from collections import defaultdict
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import pyedflib
from moabb.datasets import Schirrmeister2017
from moabb.paradigms import MotorImagery
from sklearn.preprocessing import LabelEncoder
from scipy import io, signal
import tensorflow as tf
import tensorflow.keras.utils as np_utils # type: ignore
import matplotlib.pyplot as plt
from scipy.signal import resample, butter, filtfilt, lfilter, iirnotch 

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Disable GPU

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class BCICIV2aLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.stimcodes = ['769', '770', '771', '772']
        self.sample_freq = None
        self.batch_size = 16

    def load_data(self, filename):
        gdf_name = filename.split(".")[0]
        raw_data = mne.io.read_raw_gdf(os.path.join(self.filepath, filename), preload=True, eog=['EOG-left', 'EOG-central', 'EOG-right'])
        raw_data.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])
        # raw_data.filter(l_freq=4, h_freq=40)
        raw_data.resample(128)
        self.sample_freq = int(raw_data.info.get('sfreq'))
        before_trial = int(0.5 * self.sample_freq)
        data = raw_data.get_data() * 1e6  # Convert to microvolts

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

        trials = np.array(trials)
        trials = self.normalize_channels(trials)

        labels = self._get_labels(np.array(labels))
        print(f"Train data loaded: trials {trials.shape}, labels {labels.shape}")
        return trials, labels

    def _process_evaluation_data(self, raw_data, data, gdf_name, before_trial):
        trials = []
        for annotation in raw_data.annotations:
            if annotation['description'] == "783":
                onset_idx = int(annotation['onset'] * self.sample_freq)
                trial = data[:, onset_idx - before_trial:onset_idx + int(4 * self.sample_freq)]
                trials.append(trial)

        trials = np.array(trials)
        trials = self.normalize_channels(trials)

        labels = io.loadmat(os.path.join(self.filepath, "true_labels", gdf_name + ".mat"))["classlabel"][:, 0] - 1
        labels = np_utils.to_categorical(labels, num_classes=4)
        print(f"Test data loaded: trials {trials.shape}, labels {labels.shape}")
        return trials, labels

    def normalize_channels(self, trials):
        """
        Normalize each channel in the trials using StandardScaler.

        Parameters:
        trials (numpy.ndarray): Input data of shape (n_trials, n_channels, n_timepoints).
                                Each channel of each trial will be normalized independently.

        Returns:
        numpy.ndarray: The normalized trials with the same shape as the input.
        """
        for j in range(trials.shape[1]):  # Iterate over channels
            scaler = StandardScaler()
            # Fit the scaler to the data of the current channel across all trials
            scaler.fit(trials[:, j, :])
            # Transform the data for the current channel
            trials[:, j, :] = scaler.transform(trials[:, j, :])
        return trials

    def _get_labels(self, labels):
        unique_labels = np.sort(np.unique(labels))
        label_map = {old: new for new, old in enumerate(unique_labels)}
        mapped_labels = np.vectorize(label_map.get)(labels)
        return np.eye(len(unique_labels))[mapped_labels.astype(int)]

    def extract_features(self, trials):
        features = np.array([[np.mean(channel), np.std(channel), np.min(channel), np.max(channel)] 
                             for trial in trials for channel in trial])
        logging.info("Features extracted successfully.")
        return features

    def create_datasets(self, trials, labels, win_length, stride, train=True):
        """
        Create datasets from trials using sliding windows.

        Parameters:
            trials (numpy.ndarray): Array of shape (n_trials, n_channels, n_timepoints) containing trial data.
            labels (numpy.ndarray): Array of shape (n_trials, n_classes) containing one-hot encoded labels.
            win_length (int): Length of each sliding window in timepoints.
            stride (int): Step size between sliding windows in timepoints.

        Returns:
            tf.data.Dataset: TensorFlow dataset containing sliding windows and their corresponding labels.
        """
        windowed_data, windowed_labels = [], []

        # Iterate over each trial and create sliding windows
        for trial, label in zip(trials, labels):
            n_windows = (trial.shape[1] - win_length) // stride + 1
            windows = []
            for i in range(n_windows):
                start = i * stride
                end = start + win_length
                window = trial[:, start:end]
                windows.append(window)
            if train:
                np.random.shuffle(windows)  # Shuffle windows within each trial for training data
            windowed_data.extend(windows)
            windowed_labels.extend([label] * len(windows))

        # Convert lists to numpy arrays
        windowed_data = np.array(windowed_data)
        windowed_labels = np.array(windowed_labels)

        print(f"Number of windows: {len(windowed_data)}, trials shape: {windowed_data.shape}, labels: {windowed_labels.shape}") 
        # Create TensorFlow dataset
        # dataset = tf.data.Dataset.from_tensor_slices((trials, labels))
        dataset = tf.data.Dataset.from_tensor_slices((windowed_data, windowed_labels))
        if train:
            # dataset = dataset.shuffle(buffer_size=10000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
            dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        else:
            dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        return dataset

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
            print(f"trial shape: {trials.shape}, label shape: {labels.shape}")
            win_length = 2 * self.sample_freq  # Window length for crops (2 seconds)
            stride = 1 * self.sample_freq  # Define stride length here
            # dataset = self.create_datasets(trials, labels, win_length, stride)

            if subject not in eeg_data:
                eeg_data[subject] = {}
            if "T" in gdf_name:
                dataset = self.create_datasets(trials, labels, win_length, stride, train=True)
                eeg_data[subject]["train_ds"] = dataset
            elif "E" in gdf_name:
                dataset = self.create_datasets(trials, labels, win_length, stride, train=False)
                eeg_data[subject]["test_ds"] = dataset

        return eeg_data

class BCICIV2bLoader:
    def __init__(self, filepath):
        self.data_file_dir = filepath
        self.sample_freq = None
        self.data = {}
        self.batch_size = 16  

    def load_data(self, filename, gdf_name):
        filepath = self.data_file_dir
        raw_data = mne.io.read_raw_gdf(filepath + filename, preload=True,
                                       eog=['EOG:ch01', 'EOG:ch02', 'EOG:ch03'])
        raw_data.drop_channels(['EOG:ch01', 'EOG:ch02', 'EOG:ch03'])
        # raw_data.filter(l_freq=4, h_freq=40)
        raw_data.resample(128)
        self.sample_freq = int(raw_data.info.get('sfreq'))
        before_trial = int(0.5 * self.sample_freq)
        data = raw_data.get_data() * 1e6  # Convert to microvolts

        logging.info(f"Loading data from {filename}...")

        trials, labels = [], []
        if "T" in gdf_name:
            for annotation in raw_data.annotations:
                description = annotation['description']
                onset = annotation['onset']
                onset_idx = int(onset * self.sample_freq)

                if description in ['769', '770']:
                    trial = data[:, onset_idx - before_trial : onset_idx + int(4.5 * self.sample_freq)]
                    label = int(description)
                    trials.append(trial)
                    labels.append(label)

            trials = np.array(trials)
            trials = self.normalize_channels(trials)

            labels = np.array(labels)
            labels = self.get_labels(labels)

        elif "E" in gdf_name:
            for annotation in raw_data.annotations:
                description = annotation['description']
                onset = annotation['onset']
                onset_idx = int(onset * self.sample_freq)

                if description == "783":
                    trial = data[:, onset_idx - before_trial : onset_idx + int(4.5 * self.sample_freq)]
                    trials.append(trial)

            mat_data = io.loadmat(os.path.join(self.data_file_dir, "true_labels", gdf_name + ".mat"))
            labels = mat_data["classlabel"][:, 0] - 1
            labels = np_utils.to_categorical(labels, num_classes=2)

            trials = np.array(trials)
            trials = self.normalize_channels(trials)

        return trials, labels

    def get_labels(self, labels):
        labels_unique = np.sort(np.unique(labels))
        for new_label, old_label in enumerate(labels_unique):
            labels[labels == old_label] = new_label

        labels = labels.astype(int)
        labels = np.eye(len(labels_unique))[labels]
        return labels

    def create_datasets(self, trials, labels, win_length, stride, train=True):
        """
        Create datasets from trials using sliding windows.

        Parameters:
            trials (numpy.ndarray): Array of shape (n_trials, n_channels, n_timepoints) containing trial data.
            labels (numpy.ndarray): Array of shape (n_trials, n_classes) containing one-hot encoded labels.
            win_length (int): Length of each sliding window in timepoints.
            stride (int): Step size between sliding windows in timepoints.

        Returns:
            tf.data.Dataset: TensorFlow dataset containing sliding windows and their corresponding labels.
        """
        windowed_data, windowed_labels = [], []

        # Iterate over each trial and create sliding windows
        for trial, label in zip(trials, labels):
            n_windows = (trial.shape[1] - win_length) // stride + 1
            windows = []
            for i in range(n_windows):
                start = i * stride
                end = start + win_length
                window = trial[:, start:end]
                windows.append(window)
            if train:
                np.random.shuffle(windows)  # Shuffle windows within each trial for training data
            windowed_data.extend(windows)
            windowed_labels.extend([label] * len(windows))

        # Convert lists to numpy arrays
        windowed_data = np.array(windowed_data)
        windowed_labels = np.array(windowed_labels)

        print(f"Number of windows: {len(windowed_data)}, trials shape: {windowed_data.shape}, labels: {windowed_labels.shape}") 
        # Create TensorFlow dataset
        # dataset = tf.data.Dataset.from_tensor_slices((trials, labels))
        dataset = tf.data.Dataset.from_tensor_slices((windowed_data, windowed_labels))
        if train:
            dataset = dataset.shuffle(buffer_size=10000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        else:
            dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        return dataset
    
    def normalize_channels(self, trials):
        """
        Normalize each channel in the trials using StandardScaler.

        Parameters:
        trials (numpy.ndarray): Input data of shape (n_trials, n_channels, n_timepoints).
                                Each channel of each trial will be normalized independently.

        Returns:
        numpy.ndarray: The normalized trials with the same shape as the input.
        """
        for j in range(trials.shape[1]):  # Iterate over channels
            scaler = StandardScaler()
            # Fit the scaler to the data of the current channel across all trials
            scaler.fit(trials[:, j, :])
            # Transform the data for the current channel
            trials[:, j, :] = scaler.transform(trials[:, j, :])
        return trials
    
    def load_dataset(self):
        data_files = os.listdir(self.data_file_dir)
        for data_file in data_files:
            if not re.search(r".*\.gdf", data_file):
                continue

            gdf_name = data_file.split(".")[0]
            info = re.findall(r'B0([0-9])0([0-9])[TE]\.gdf', data_file)

            if not info:
                continue

            subject = "subject" + info[0][0]
            session = "session" + info[0][1]
            filename = os.path.join(self.data_file_dir, data_file)
            print(f"Processing {filename}")
            trials, labels = self.load_data(data_file, gdf_name)

            if subject not in self.data:
                self.data[subject] = {
                    'train_trials': [],
                    'train_labels': [],
                    'test_trials': [],
                    'test_labels': []
                }

            if info[0][1] in ['1', '2', '3']:
                self.data[subject]['train_trials'].append(trials)
                self.data[subject]['train_labels'].append(labels)
            elif info[0][1] in ['4', '5']:
                self.data[subject]['test_trials'].append(trials)
                self.data[subject]['test_labels'].append(labels)

        for subject in self.data.keys():
            print(f"Processing data for {subject}")
            all_train_trials = np.concatenate(self.data[subject]['train_trials'], axis=0)
            all_train_labels = np.concatenate(self.data[subject]['train_labels'], axis=0)
            all_test_trials = np.concatenate(self.data[subject]['test_trials'], axis=0)
            all_test_labels = np.concatenate(self.data[subject]['test_labels'], axis=0)

            win_length = 2 * self.sample_freq
            stride = 1 * self.sample_freq

            self.data[subject]['train_ds'] = self.create_datasets(all_train_trials, all_train_labels, win_length, stride, train=True)
            self.data[subject]['test_ds'] = self.create_datasets(all_test_trials, all_test_labels, win_length, stride, train=False)


            # Remove unnecessary keys after creating the datasets
            del self.data[subject]['train_trials']
            del self.data[subject]['train_labels']
            del self.data[subject]['test_trials']
            del self.data[subject]['test_labels']

        return self.data

class eachSubjectDREAMERLoader:
    def __init__(self, filepath, label_type, chunk_size=128, overlap_rate=0.5, baseline_chunk_size=128, test_size=0.2):
        self.mat_path = filepath
        self.chunk_size = chunk_size
        self.overlap_rate = overlap_rate
        self.num_channel = 14
        self.num_baseline = 61
        self.baseline_chunk_size = baseline_chunk_size
        self.test_size = test_size
        self.batch_size = 16
        self.label_type = label_type
    
    def normalize_channels(self, trials):
        """
        Normalize each channel in the trials using StandardScaler.

        Parameters:
        trials (numpy.ndarray): Input data of shape (n_trials, n_channels, n_timepoints).
                                Each channel of each trial will be normalized independently.

        Returns:
        numpy.ndarray: The normalized trials with the same shape as the input.
        """
        for j in range(trials.shape[1]):  # Iterate over channels
            scaler = StandardScaler()
            # Fit the scaler to the data of the current channel across all trials
            scaler.fit(trials[:, j, :])
            # Transform the data for the current channel
            trials[:, j, :] = scaler.transform(trials[:, j, :])
        return trials
    
    def old_normalize_channels(self, trials, scaler=None):
        if scaler is None:
            scaler = {}
            for j in range(trials.shape[1]):  # Iterate over channels
                scaler[j] = StandardScaler()
                scaler[j].fit(trials[:, j, :].reshape(trials.shape[0], -1))
                trials[:, j, :] = scaler[j].transform(trials[:, j, :].reshape(trials.shape[0], -1)).reshape(trials[:, j, :].shape)
            return trials, scaler
        else:
            for j in range(trials.shape[1]):  # Iterate over channels
                trials[:, j, :] = scaler[j].transform(trials[:, j, :].reshape(trials.shape[0], -1)).reshape(trials[:, j, :].shape)
            return trials
        
    def load_dreamer_data_by_subject(self):
        eeg_dataset = {}
        mat_data = io.loadmat(self.mat_path, verify_compressed_data_integrity=False)
        subject_len = len(mat_data['DREAMER'][0, 0]['Data'][0])  # 23 subjects
        
        overlap = int(self.chunk_size * self.overlap_rate)

        for subject in range(subject_len):
            trials_data = []
            trials_labels = []
            trial_len = len(mat_data['DREAMER'][0, 0]['Data'][0, 0]['EEG'][0, 0]['stimuli'][0, 0])  # 18 trials

            for trial_id in range(trial_len):
                trial_baseline_sample = mat_data['DREAMER'][0, 0]['Data'][0, subject]['EEG'][0, 0]['baseline'][0, 0][trial_id, 0]
                trial_baseline_sample = trial_baseline_sample[:, :self.num_channel].swapaxes(1, 0)
                trial_baseline_sample = trial_baseline_sample[:, :self.num_baseline * self.baseline_chunk_size].reshape(self.num_channel, self.num_baseline, self.baseline_chunk_size).mean(axis=1)

                trial_samples = mat_data['DREAMER'][0, 0]['Data'][0, subject]['EEG'][0, 0]['stimuli'][0, 0][trial_id, 0]
                trial_samples = trial_samples[:, :self.num_channel].swapaxes(1, 0)

                start_at = 0
                dynamic_chunk_size = self.chunk_size if self.chunk_size > 0 else trial_samples.shape[1] - start_at
                step = dynamic_chunk_size - overlap
                end_at = dynamic_chunk_size

                trial_data = []
                trial_labels = []

                while end_at <= trial_samples.shape[1]:
                    clip_sample = trial_samples[:, start_at:end_at]

                    # Determine label based on valence or arousal
                    if self.label_type == 'valence':
                        valence = mat_data['DREAMER'][0, 0]['Data'][0, subject]['ScoreValence'][0, 0][trial_id, 0]
                        label = 1 if valence >= 3.0 else 0
                    elif self.label_type == 'arousal':
                        arousal = mat_data['DREAMER'][0, 0]['Data'][0, subject]['ScoreArousal'][0, 0][trial_id, 0]
                        label = 1 if arousal >= 3.0 else 0

                    trial_data.append(clip_sample)
                    trial_labels.append(label)

                    start_at += step
                    end_at = start_at + dynamic_chunk_size

                trials_data.append(np.array(trial_data))
                trials_labels.append(np.array(trial_labels))

            # Store the trials data and labels for each subject
            eeg_dataset[subject+1] = {
                'eeg_data': trials_data,
                'labels': trials_labels
            }

        return eeg_dataset

    def old_prepare_data_for_training(self, eeg_dataset):
        for subject in eeg_dataset:
            print("Preparing data for subject", subject)
            trials_data = eeg_dataset[subject]['eeg_data']
            trials_labels = eeg_dataset[subject]['labels']
            
            # Split trials into training and testing sets
            num_trials = len(trials_data)
            num_test_trials = int(num_trials * self.test_size)
            test_indices = np.random.choice(num_trials, num_test_trials, replace=False)
            train_indices = list(set(range(num_trials)) - set(test_indices))

            # Prepare training and testing datasets
            train_data = []
            train_labels = []
            test_data = []
            test_labels = []

            for i in train_indices:
                trial_data = trials_data[i]
                trial_labels = trials_labels[i]

                # Shuffle windows within each trial for training only
                shuffle_idx = np.random.permutation(len(trial_data))
                trial_data = trial_data[shuffle_idx]
                trial_labels = trial_labels[shuffle_idx]

                train_data.extend(trial_data)
                train_labels.extend(trial_labels)

            for i in test_indices:
                test_data.extend(trials_data[i])
                test_labels.extend(trials_labels[i])

            # Convert lists to numpy arrays and one-hot encode labels
            train_data = np.array(train_data)
            train_data = self.normalize_channels(train_data)
            train_labels = np_utils.to_categorical(np.array(train_labels))
            test_data = np.array(test_data)
            test_data = self.normalize_channels(test_data)
            
            test_labels = np_utils.to_categorical(np.array(test_labels))

            print(f"Subject {subject} - Shape of X_train: {train_data.shape}, y_train: {train_labels.shape}")
            print(f"Subject {subject} - Shape of X_test: {test_data.shape}, y_test: {test_labels.shape}")

            # Convert to TensorFlow Datasets
            train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
            test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
            train_dataset = train_dataset.shuffle(buffer_size=10000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
            test_dataset = test_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

            # Update the eeg_dataset to keep only train_ds and test_ds
            eeg_dataset[subject] = {
                'train_ds': train_dataset,
                'test_ds': test_dataset
            }
        return eeg_dataset

    def prepare_data_for_training(self, eeg_dataset):
        for subject in eeg_dataset:
            print("Preparing data for subject", subject)
            trials_data = eeg_dataset[subject]['eeg_data']  # List of trials
            trials_labels = eeg_dataset[subject]['labels']  # List of labels for each trial
            
            # Aggregate labels at the trial level by taking the majority label or representative label
            trial_labels_flat = [np.argmax(np.bincount(trial_labels)) for trial_labels in trials_labels]

            # Use StratifiedShuffleSplit to ensure balanced classes in train and test
            strat_split = StratifiedShuffleSplit(n_splits=1, test_size=self.test_size, random_state=42)
            for train_indices, test_indices in strat_split.split(trials_data, trial_labels_flat):
                # Prepare training and testing datasets
                train_data = []
                train_labels = []
                test_data = []
                test_labels = []

                # Process training trials
                for i in train_indices:
                    trial_data = trials_data[i]
                    trial_labels = trials_labels[i]

                    # Shuffle windows within each trial for training only
                    shuffle_idx = np.random.permutation(len(trial_data))
                    trial_data = trial_data[shuffle_idx]
                    trial_labels = trial_labels[shuffle_idx]

                    train_data.extend(trial_data)
                    train_labels.extend(trial_labels)

                # Process testing trials
                for i in test_indices:
                    test_data.extend(trials_data[i])
                    test_labels.extend(trials_labels[i])

                # Convert lists to numpy arrays and normalize using training statistics
                train_data = np.array(train_data)
                train_data, scaler = self.normalize_channels(train_data)
                train_labels = np_utils.to_categorical(np.array(train_labels))

                test_data = np.array(test_data)
                test_data = self.normalize_channels(test_data, scaler=scaler)
                test_labels = np_utils.to_categorical(np.array(test_labels))

                print(f"Subject {subject} - Shape of X_train: {train_data.shape}, y_train: {train_labels.shape}")
                print(f"Subject {subject} - Shape of X_test: {test_data.shape}, y_test: {test_labels.shape}")

                # Convert to TensorFlow Datasets
                train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
                test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
                train_dataset = train_dataset.shuffle(buffer_size=10000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
                test_dataset = test_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

                # Update the eeg_dataset to keep only train_ds and test_ds
                eeg_dataset[subject] = {
                    'train_ds': train_dataset,
                    'test_ds': test_dataset
                }
        return eeg_dataset
    
    def load_dataset(self):
        # Call functions to load the DREAMER dataset
        eeg_dataset = self.load_dreamer_data_by_subject()
        eeg_dataset = self.prepare_data_for_training(eeg_dataset)
        return eeg_dataset

class DREAMERLoader:
    def __init__(self, filepath, label_type, chunk_size=128, overlap_rate=0.5, baseline_chunk_size=128, n_splits=3):
        self.mat_path = filepath
        self.chunk_size = chunk_size
        self.overlap_rate = overlap_rate
        self.num_channel = 14
        self.num_baseline = 61
        self.baseline_chunk_size = baseline_chunk_size
        self.n_splits = n_splits
        self.label_type = label_type
        self.batch_size = 16
        self.eeg_data = {}  # Dictionary to store all data and fold indices

    def normalize_channels(self, trials):
        """Normalize each channel independently across all trials."""
        for j in range(trials.shape[1]):  # Iterate over channels
            scaler = StandardScaler()
            scaler.fit(trials[:, j, :])
            trials[:, j, :] = scaler.transform(trials[:, j, :])
        return trials

    def load_dreamer_data_by_subject(self):
        """Load the DREAMER dataset for each subject and return a dictionary."""
        eeg_dataset = {}
        mat_data = io.loadmat(self.mat_path, verify_compressed_data_integrity=False)
        subject_len = len(mat_data['DREAMER'][0, 0]['Data'][0])  # 23 subjects

        overlap = int(self.chunk_size * self.overlap_rate)

        for subject in range(subject_len):
            trials_data = []
            trials_labels = []
            trial_len = len(mat_data['DREAMER'][0, 0]['Data'][0, 0]['EEG'][0, 0]['stimuli'][0, 0])  # 18 trials

            for trial_id in range(trial_len):
                trial_samples = mat_data['DREAMER'][0, 0]['Data'][0, subject]['EEG'][0, 0]['stimuli'][0, 0][trial_id, 0]
                trial_samples = trial_samples[:, :self.num_channel].swapaxes(1, 0)

                start_at = 0
                step = self.chunk_size - overlap
                end_at = self.chunk_size

                trial_data = []
                trial_labels = []

                while end_at <= trial_samples.shape[1]:
                    clip_sample = trial_samples[:, start_at:end_at]
                    label = self.get_label(mat_data, subject, trial_id)  # Retrieve label based on arousal or valence
                    trial_data.append(clip_sample)
                    trial_labels.append(label)

                    start_at += step
                    end_at = start_at + self.chunk_size

                trials_data.append(np.array(trial_data))
                trials_labels.append(np.array(trial_labels))

            eeg_dataset[subject+1] = {
                'eeg_data': trials_data,
                'labels': trials_labels
            }

        return eeg_dataset

    def get_label(self, mat_data, subject, trial_id):
        """Helper function to get the label based on valence or arousal."""
        if self.label_type == 'valence':
            valence = mat_data['DREAMER'][0, 0]['Data'][0, subject]['ScoreValence'][0, 0][trial_id, 0]
            return 1 if valence >= 3.0 else 0
        elif self.label_type == 'arousal':
            arousal = mat_data['DREAMER'][0, 0]['Data'][0, subject]['ScoreArousal'][0, 0][trial_id, 0]
            return 1 if arousal >= 3.0 else 0

    def prepare_data_for_cross_validation(self, eeg_dataset):
        """Prepare data for subject-level k-fold cross-validation."""
        all_data = []
        all_labels = []
        subject_ids = []

        for subject, data in eeg_dataset.items():
            trials_data = np.concatenate(data['eeg_data'], axis=0)
            trials_labels = np.concatenate(data['labels'], axis=0)
            trials_data = self.normalize_channels(trials_data)

            all_data.append(trials_data)
            all_labels.append(trials_labels)
            subject_ids.extend([subject] * len(trials_data))

            print(f"Subject {subject} - Total trials: {len(trials_data)}, Labels shape: {trials_labels.shape}")

        # Concatenate all subjects' data and labels
        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_labels = np_utils.to_categorical(all_labels, num_classes=2)

        self.eeg_data['all'] = {'trials': all_data, 'labels': all_labels}

        # Use GroupKFold to split by subject
        gkf = GroupKFold(n_splits=self.n_splits)
        for fold, (train_indices, test_indices) in enumerate(gkf.split(all_data, groups=subject_ids), start=1):
            self.eeg_data[fold] = {'train_indices': train_indices, 'test_indices': test_indices}

            # Print debugging information for each fold
            train_subjects = set(np.array(subject_ids)[train_indices])
            test_subjects = set(np.array(subject_ids)[test_indices])
            print(f"\nFold {fold}")
            print(f"Training subjects: {sorted(train_subjects)}")
            print(f"Testing subjects: {sorted(test_subjects)}")
            print(f"Train data shape: {len(train_indices)}, Test data shape: {len(test_indices)}")

        return self.eeg_data

    def load_dataset(self):
        """Load the DREAMER dataset and prepare it for k-fold cross-validation."""
        eeg_dataset = self.load_dreamer_data_by_subject()
        self.eeg_data = self.prepare_data_for_cross_validation(eeg_dataset)
        return self.eeg_data

class DEAPLoader:
    def __init__(self, filepath, label_type, win_length=8, stride=4):
        self.data_path = filepath
        self.sample_freq = 128
        self.win_length = win_length * self.sample_freq  # Convert window length to samples
        self.stride = stride * self.sample_freq  # Convert stride to samples
        self.label_type = label_type
        self.batch_size = 16
        self.eeg_data = {}
        self.n_splits = 3

    def load_data_per_subject(self, subject_id):
        """
        Loads data for a specific subject and excludes the first 3 seconds of baseline.
        """
        sub_filename = "s{:02d}.dat".format(subject_id)
        print(f"Loading subject {sub_filename}")
        subject_path = os.path.join(self.data_path, sub_filename)

        if not os.path.exists(subject_path):
            print(f"File {sub_filename} not found.")
            return None, None
        
        print(f"Loading data from {sub_filename}...")
        with open(subject_path, 'rb') as f:
            subject = pickle.load(f, encoding='latin1')
        
        labels = subject['labels']
        data = subject['data'][:, 0:32, 3 * self.sample_freq:]  # Exclude first 3s of baseline, 32 channels
        return data, labels

    def get_labels(self, labels):
        """
        Processes the labels according to the label type ('valence' or 'arousal') and makes them binary.
        """
        if self.label_type == 'valence':
            labels = labels[:, 0]
        elif self.label_type == 'arousal':
            labels = labels[:, 1]
        
        labels = np.where(labels > 5, 1, 0)
        labels = np_utils.to_categorical(labels)
        return labels
    
    def normalize_channels(self, trials):
        """
        Normalize each channel in the trials using StandardScaler.

        Parameters:
        trials (numpy.ndarray): Input data of shape (n_trials, n_channels, n_timepoints).
                                Each channel of each trial will be normalized independently.

        Returns:
        numpy.ndarray: The normalized trials with the same shape as the input.
        """
        for j in range(trials.shape[1]):  # Iterate over channels
            scaler = StandardScaler()
            # Fit the scaler to the data of the current channel across all trials
            scaler.fit(trials[:, j, :])
            # Transform the data for the current channel
            trials[:, j, :] = scaler.transform(trials[:, j, :])
        return trials

    def create_datasets(self, trials, labels, train=True):
        """
        Create datasets from trials using sliding windows.

        Parameters:
            trials (numpy.ndarray): Array of shape (n_trials, n_channels, n_timepoints) containing trial data.
            labels (numpy.ndarray): Array of shape (n_trials, n_classes) containing one-hot encoded labels.
            train (bool): Whether to shuffle windows within each trial (used for training data only).

        Returns:
            numpy.ndarray, numpy.ndarray: Windowed data and corresponding labels.
        """
        windowed_data, windowed_labels = [], []

        # Iterate over each trial and create sliding windows
        for trial, label in zip(trials, labels):
            n_windows = (trial.shape[1] - self.win_length) // self.stride + 1
            windows = [trial[:, i * self.stride:i * self.stride + self.win_length] for i in range(n_windows)]
            
            if train:
                np.random.shuffle(windows)  # Shuffle windows within each trial for training data
            
            windowed_data.extend(windows)
            windowed_labels.extend([label] * len(windows))

        # Convert lists to numpy arrays
        windowed_data = np.array(windowed_data)
        windowed_labels = np.array(windowed_labels)

        return windowed_data, windowed_labels


    def load_dataset(self):
        """
        Loads and processes the data for all subjects, and prepares subject-level k-fold cross-validation indices.
        """
        all_data, all_labels, subject_ids = [], [], []

        # Load and process data for each subject
        for subject_id in range(1, 33):
            data, labels = self.load_data_per_subject(subject_id)

            if data is None or labels is None:
                continue

            labels = self.get_labels(labels)
            data = self.normalize_channels(data)

            # Convert data to sliding windows
            subject_windows, subject_labels = self.create_datasets(data, labels)

            all_data.append(subject_windows)
            all_labels.append(subject_labels)
            subject_ids.extend([subject_id] * len(subject_windows))  # Assign subject_id for each window

        # Concatenate all subjects' data and labels
        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Store the full dataset
        self.eeg_data['all'] = {'trials': all_data, 'labels': all_labels}

        # Create k-folds at the subject level
        gkf = GroupKFold(n_splits=self.n_splits)
        for fold, (train_indices, test_indices) in enumerate(gkf.split(all_data, groups=subject_ids), start=1):
            # Store only the indices for each fold
            self.eeg_data[fold] = {
                'train_indices': train_indices,
                'test_indices': test_indices
            }
            print(f"Fold {fold} - Training samples: {len(train_indices)}, Testing samples: {len(test_indices)}")

        return self.eeg_data

class PhysionetMILoader:
    def __init__(self, filepath, trial_length=4, n_splits=3):
        self.data_file_dir = filepath
        self.original_freq = None  # Original sample frequency (160 Hz)
        self.target_freq = 128  # Target sample frequency (128 Hz)
        self.trial_length = trial_length  # Trial length in seconds
        self.eeg_data = {}  # Store train/test datasets for each subject
        self.batch_size = 16
        self.n_splits = n_splits

    def extract_file_paths(self, subject_folder):
        """
        Extracts file paths for EDF files based on a pattern.
        """
        subject_dir = os.path.join(self.data_file_dir, subject_folder)
        data_files = os.listdir(subject_dir)
        pattern = re.compile(r"^S(\d{3})R(\d{2}).edf$")
        filtered_filenames = list(filter(lambda x: pattern.match(x), data_files))
        file_paths = [os.path.join(self.data_file_dir, subject_folder, filename) for filename in filtered_filenames]
        print(f"Extracted {len(file_paths)} EDF files for {subject_folder}")
        return file_paths

    def load_edf(self, path):
        """
        Loads an EDF file and extracts the signals and annotations. Resamples the data from 160 Hz to 128 Hz.
        """
        try:
            print(f"Loading EDF file {path}...")
            sig = pyedflib.EdfReader(path)
            n_channels = sig.signals_in_file
            self.original_freq = sig.getSampleFrequency(0)
            sigbuf = np.zeros((n_channels, sig.getNSamples()[0]))

            for j in np.arange(n_channels):
                sigbuf[j, :] = sig.readSignal(j)

            annotations = sig.read_annotation()
        except Exception as e:
            print(f"Error loading EDF file {path}: {e}")
            return None, None

        sig._close()
        del sig

        # Resample the signals from original_sample_rate to target_sample_rate
        num_samples = int(sigbuf.shape[1] * self.target_freq / self.original_freq)
        resampled_signals = resample(sigbuf, num_samples, axis=1)

        print(f"Resampled EDF file: {path} from {self.original_freq} Hz to {self.target_freq} Hz")
        print(f"Resampled signals shape: {resampled_signals.shape}")

        return resampled_signals.transpose(), annotations  # Return resampled signals

    def load_all_data(self, file_paths):
        """
        Loads all EDF files and concatenates their signals and annotations.
        """
        all_signals = []
        all_annotations = []

        for path in file_paths:
            signals, annotations = self.load_edf(path)
            if signals is not None and annotations is not None:
                all_signals.append(signals)
                all_annotations.append(annotations)
        
        if all_signals:
            all_signals = np.concatenate(all_signals, axis=0)
            all_annotations = np.concatenate(all_annotations, axis=0)
            print(f"Loaded and concatenated signals and annotations: {all_signals.shape}, {all_annotations.shape}")
        else:
            print(f"No valid signals or annotations found for paths: {file_paths}")
            return None, None
        
        return all_signals, all_annotations

    def parse_annotations(self, annotations):
        """
        Parses annotations from byte strings to readable values.
        """
        parsed_annotations = []
        for annotation in annotations:
            start_time = float(annotation[0].decode('utf-8'))
            duration = float(annotation[1].decode('utf-8'))
            event_type = annotation[2].decode('utf-8')
            parsed_annotations.append([start_time, duration, event_type])
        
        print(f"Parsed {len(parsed_annotations)} annotations")
        return parsed_annotations
    
    def normalize_channels(self, trials):
        """
        Normalize each channel in the trials using StandardScaler.

        Parameters:
        trials (numpy.ndarray): Input data of shape (n_trials, n_channels, n_timepoints).
                                Each channel of each trial will be normalized independently.

        Returns:
        numpy.ndarray: The normalized trials with the same shape as the input.
        """
        for j in range(trials.shape[1]):  # Iterate over channels
            scaler = StandardScaler()
            # Fit the scaler to the data of the current channel across all trials
            scaler.fit(trials[:, j, :])
            # Transform the data for the current channel
            trials[:, j, :] = scaler.transform(trials[:, j, :])
        return trials

    def extract_trials_and_labels(self, signals, annotations):
        """
        Extracts trials and labels from signals and annotations.
        """
        num_samples = self.trial_length * self.target_freq  # Calculate number of samples after resampling
        trials = []
        labels = []

        parsed_annotations = self.parse_annotations(annotations)

        for annotation in parsed_annotations:
            start_time = int(annotation[0] * self.target_freq * 1e-7)  # Convert to samples
            end_time = start_time + num_samples

            if end_time <= len(signals):
                trial = signals[start_time:end_time]
                trials.append(trial)

                label = [0] * 4  # Assuming 4 classes
                event_type = annotation[2]
                if event_type == 'T0':
                    label = [1, 0, 0, 0]  # Rest
                elif event_type == 'T1':
                    label = [0, 1, 0, 0]  # Fist Left
                elif event_type == 'T2':
                    label = [0, 0, 1, 0]  # Fist Right
                elif event_type == 'T3':
                    label = [0, 0, 0, 1]  # Feet
                labels.append(label)
        
        trials = np.array(trials)
        labels = np.array(labels)

        # Reshape trials from (n_trials, time_steps, n_channels) to (n_trials, n_channels, time_steps)
        trials = trials.transpose(0, 2, 1)
        trials = self.normalize_channels(trials)

        print(f"Extracted {trials.shape} trials and {labels.shape} labels")
        return trials, labels

    def load_dataset(self):
        # Get a list of all subject folders
        subject_folders = [f for f in os.listdir(self.data_file_dir) if os.path.isdir(os.path.join(self.data_file_dir, f))]
        
        subject_data = []  # List to store trials and labels per subject
        subject_ids = []   # To track subject ID for each trial

        # Load and process each subject's data
        for subject_folder in subject_folders:
            subject_id = subject_folder  # Assume each folder name is unique to a subject
            file_paths = self.extract_file_paths(subject_folder)
            signals, annotations = self.load_all_data(file_paths)
            
            if signals is None or annotations is None:
                continue

            trials, labels = self.extract_trials_and_labels(signals, annotations)
            subject_data.append((trials, labels))
            
            # Add subject ID for each trial in this subject's data
            subject_ids.extend([subject_id] * len(trials))

        # Check if data was loaded
        if not subject_data:
            print("No valid data found.")
            return None

        # Concatenate all trials and labels into a single dataset
        all_trials = np.concatenate([data[0] for data in subject_data], axis=0)
        all_labels = np.concatenate([data[1] for data in subject_data], axis=0)

        # Normalize the entire dataset if needed
        all_trials = self.normalize_channels(all_trials)

        # Store the entire normalized dataset in eeg_data['all']
        self.eeg_data['all'] = {'trials': all_trials, 'labels': all_labels}

        # Use GroupKFold to split by subject (grouping by subject ID)
        gkf = GroupKFold(n_splits=self.n_splits)
        for fold, (train_indices, test_indices) in enumerate(gkf.split(all_trials, groups=subject_ids)):
            # Extract unique subject IDs for the train and test sets
            train_subjects = set(subject_ids[i] for i in train_indices)
            test_subjects = set(subject_ids[i] for i in test_indices)

            # Store only the indices for each fold
            self.eeg_data[fold+1] = {'train_indices': train_indices, 'test_indices': test_indices}

            # Debugging output
            print(f"\nFold {fold + 1}")
            print(f"Training subjects: {sorted(train_subjects)}")
            print(f"Testing subjects: {sorted(test_subjects)}")
            print(f"Number of training samples: {len(train_indices)}")
            print(f"Number of testing samples: {len(test_indices)}")

        print("Data loaded successfully with subject-level fold indices saved in eeg_data.")
        return self.eeg_data

class SEEDLoader:
    def __init__(self, filepath, label_path, window_size=8, stride=4):
        self.folder_path = filepath  # Folder containing the EEG .mat files
        self.sample_freq = 200  # Sampling frequency
        self.target_sample_freq = 128  # Target sampling frequency
        self.window_size = window_size * self.target_sample_freq  # Convert window size from seconds to samples
        self.stride = stride * self.target_sample_freq  # Convert stride from seconds to samples
        self.batch_size = 16  # Batch size for the datasets
        self.label_path = label_path  # Path to the labels .mat file

        # Load labels from the label file
        print(f"Loading labels from {label_path}...")
        label_mat = io.loadmat(label_path)
        self.labels = label_mat["label"].flatten()
        print(f"Loaded {len(self.labels)} labels")

        # Prepare a dictionary to hold train and test datasets for each subject
        self.eeg_data = {}

        # Extract subject ids
        file_names = os.listdir(filepath)
        pattern = re.compile(r'^(\d{1,2})_\d{8}\.mat$')
        subjects = set()

        for filename in file_names:
            match = pattern.match(filename)
            if match:
                subjects.add(int(match.group(1)))
        self.subjects = sorted(list(subjects))
        print(f"Subjects found: {self.subjects}")

    def preprocess_eeg_data(self, raw_eeg):
        """
        Preprocess the EEG data using CAR, filtering, and resampling to 128 Hz.
        """
        print("Applying Common Average Reference (CAR) and bandpass filtering...")
        # Apply common average reference (CAR) across all channels
        average_reference = np.mean(raw_eeg, axis=0)
        car_eeg = raw_eeg - average_reference

        # Apply filter between 0.15 Hz and 40 Hz to the CAR-corrected EEG data
        b, a = signal.butter(4, [0.15, 40], btype='bandpass', fs=self.sample_freq)
        filtered_eeg = signal.filtfilt(b, a, car_eeg, axis=1)

        print(f"Filtered EEG shape: {filtered_eeg.shape}")

        # Resample the EEG data from 200 Hz to 128 Hz
        num_samples = int(filtered_eeg.shape[1] * self.target_sample_freq / self.sample_freq)
        resampled_eeg = signal.resample(filtered_eeg, num_samples, axis=1)

        print(f"Resampled EEG shape: {resampled_eeg.shape} at {self.target_sample_freq} Hz")

        return resampled_eeg

    def normalize_channels(self, trials):
        """
        Normalize each channel in the trials using StandardScaler.

        Parameters:
        trials (numpy.ndarray): Input data of shape (n_trials, n_channels, n_timepoints).
                                Each channel of each trial will be normalized independently.

        Returns:
        numpy.ndarray: The normalized trials with the same shape as the input.
        """
        for j in range(trials.shape[1]):  # Iterate over channels
            scaler = StandardScaler()
            # Fit the scaler to the data of the current channel across all trials
            scaler.fit(trials[:, j, :])
            # Transform the data for the current channel
            trials[:, j, :] = scaler.transform(trials[:, j, :])
        return trials
    
    def get_labels(self, labels):
        """
        Perform one-hot encoding for the labels.
        """
        labels_unique = np.sort(np.unique(labels))
        print(f"Unique labels found: {labels_unique}")
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(labels_unique)}
        print(f"Label mapping: {label_mapping}")

        # Map the original labels to their new indices
        labels = np.array([label_mapping[label] for label in labels])

        # Perform one-hot encoding
        labels = np.eye(len(labels_unique))[labels]
        print(f"One-hot encoded labels shape: {labels.shape}")
        return labels

    def sliding_windows(self, eeg_data):
        """
        Generate sliding window segments from EEG data.
        """
        print(f"Generating sliding windows with window size {self.window_size} and stride {self.stride}")
    
        windows = []
        num_channels, total_length = eeg_data.shape
        num_windows = (total_length - self.window_size) // self.stride + 1
        print(f"Number of windows to generate: {num_windows}")

        for start in range(0, total_length - self.window_size + 1, self.stride):
            end = start + self.window_size
            segment = eeg_data[:, start:end]
            windows.append(segment)

        print(f"Generated {len(windows)} windows")
        return np.array(windows)

    def load_data(self, subject_id):
        """
        Load and preprocess EEG data for a specific subject.
        """
        print(f"\nLoading data for subject {subject_id}...")
        eeg_substring = 'eeg'
        keys_to_ignore = ['__header__', '__version__', '__globals__']
        windows_list = []
        label_list = []

        # Find all files for this subject
        subject_files = [f for f in os.listdir(self.folder_path) if f.startswith(f"{subject_id}_") and f.endswith('.mat')]
        if not subject_files:
            print(f"No files found for subject {subject_id}. Skipping...")
            return None, None

        for file_name in subject_files:
            file_path = os.path.join(self.folder_path, file_name)
            print(f"Loading file: {file_path}")
            data = io.loadmat(file_path)

            labelcount = 0
            for trial_name, trial_data in data.items():
                if trial_name not in keys_to_ignore and eeg_substring in trial_name:
                    print(f"Processing trial: {trial_name}")
                    preprocessed_eeg = self.preprocess_eeg_data(trial_data)

                    # Split preprocessed EEG data into sliding windows
                    windows = self.sliding_windows(preprocessed_eeg)

                    # Shuffle the windows within each trial to avoid sequential bias
                    print("Shuffling windows within trial...")
                    trial_indices = np.arange(windows.shape[0])
                    np.random.shuffle(trial_indices)
                    windows = windows[trial_indices]

                    windows_list.append(windows)
                    label_list += [self.labels[labelcount]] * windows.shape[0]
                    labelcount += 1

        if windows_list:
            windows_array = np.concatenate(windows_list, axis=0)
            label_array = np.array(label_list)
            print(f"Concatenated windows shape: {windows_array.shape}, labels shape: {label_array.shape}")

            # Convert labels to one-hot encoded format
            label_array = self.get_labels(label_array)

            return windows_array, label_array
        else:
            print(f"No valid data for subject {subject_id}.")
            return None, None

    def create_tf_datasets(self, windows_array, label_array):
        """
        Create TensorFlow datasets from the windows and labels.
        """
        print(f"Splitting data into train and test datasets...")
        X_train, X_test, y_train, y_test = train_test_split(
            windows_array, label_array, test_size=0.2, random_state=None)
        
        # Normalize the channels in the data
        X_train = self.normalize_channels(X_train)
        X_test = self.normalize_channels(X_test)

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

        # Shuffle and batch the train dataset only
        train_dataset = train_dataset.shuffle(buffer_size=10000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        test_dataset = test_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        print(f"Train dataset shape: {X_train.shape}, Test dataset shape: {X_test.shape}")
        return train_dataset, test_dataset

    def load_dataset(self):
        """
        Load EEG data for all subjects and prepare train and test datasets.
        """
        for subject_id in self.subjects:
            print(f"Loading data for subject {subject_id}...")
            windows_array, label_array = self.load_data(subject_id)
            
            if windows_array is None or label_array is None:
                print(f"No data available for subject {subject_id}. Skipping...")
                continue

            train_dataset, test_dataset = self.create_tf_datasets(windows_array, label_array)

            # Store the datasets in the dictionary
            self.eeg_data[subject_id] = {
                'train_ds': train_dataset,
                'test_ds': test_dataset
            }

        print("All subjects have been processed.")
        return self.eeg_data

class STEWLoader:
    def __init__(self, filepath):
        self.folder_path = filepath
        self.channels = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
        self.sfreq = 128
        self.window_size = 512
        self.step_size = 128
        self.overlap = self.window_size - self.step_size
        self.batch_size = 16
        self.all_files_path = glob(os.path.join(filepath, '*.txt'))
        self.file_names = [x for x in os.listdir(filepath) if x.endswith(".txt") and x.startswith("sub")]
        self.eeg_data = {}
        self.n_splits = 5
        logging.info(f"Found {len(self.all_files_path)} EEG files.")

    def create_windows(self, data, step):
        num_windows = (data.shape[0] - self.window_size) // step + 1
        windows = np.array([data[i * step:i * step + self.window_size] for i in range(num_windows)])
        return windows

    def normalize_channels(self, trials):
        """
        Normalize each channel in the trials using StandardScaler.

        Parameters:
        trials (numpy.ndarray): Input data of shape (n_trials, n_channels, n_timepoints).
                                Each channel of each trial will be normalized independently.

        Returns:
        numpy.ndarray: The normalized trials with the same shape as the input.
        """
        for j in range(trials.shape[1]):  # Iterate over channels
            scaler = StandardScaler()
            # Fit the scaler to the data of the current channel across all trials
            scaler.fit(trials[:, j, :])
            # Transform the data for the current channel
            trials[:, j, :] = scaler.transform(trials[:, j, :])
        return trials
    
    def process_eeg_data(self, file_path):
        data = pd.read_csv(file_path, sep='\s+', header=None)
        data.columns = self.channels

        # Average Referencing
        # data_avg_ref = data.sub(data.mean(axis=1), axis=0)

        # Windowing the Data
        windowed_data = {ch: self.create_windows(data[ch], self.step_size) for ch in self.channels}
        num_windows = windowed_data[self.channels[0]].shape[0]
        windowed_eeg = np.array([np.array([windowed_data[ch][i] for ch in self.channels]) for i in range(num_windows)])
        return windowed_eeg

    def load_subject_data(self, subject_id):
        subject_files = [file for file in self.all_files_path if f"sub{subject_id:02d}_" in file]
        hi_files = [file for file in subject_files if '_hi' in file]
        lo_files = [file for file in subject_files if '_lo' in file]

        hi_epochs = [self.process_eeg_data(file) for file in hi_files]
        lo_epochs = [self.process_eeg_data(file) for file in lo_files]

        if not hi_epochs or not lo_epochs:
            return None, None

        hi_data = np.vstack(hi_epochs)
        lo_data = np.vstack(lo_epochs)

        # Create labels
        hi_labels = np.ones(hi_data.shape[0], dtype=int)
        lo_labels = np.zeros(lo_data.shape[0], dtype=int)

        # Combine data and labels
        data_array = np.concatenate((hi_data, lo_data), axis=0)
        label_array = np.concatenate((hi_labels, lo_labels), axis=0)
        label_array = np_utils.to_categorical(label_array, num_classes=2)

        data_array = self.normalize_channels(data_array)

        return data_array, label_array

    def load_dataset(self):
        unique_subject_ids = sorted(set(int(file.split('_')[0][3:]) for file in self.file_names))

        all_data, all_labels, subject_ids = [], [], []

        # Load data for each subject
        for subject_id in unique_subject_ids:
            data_array, label_array = self.load_subject_data(subject_id)
            if data_array is not None and label_array is not None:
                all_data.append(data_array)
                all_labels.append(label_array)
                subject_ids.extend([subject_id] * data_array.shape[0])
            print(f"Loaded data for subject {subject_id}: Data shape: {data_array.shape}, Labels shape: {label_array.shape}")

        # Concatenate all data from all subjects
        all_data = np.vstack(all_data)
        all_labels = np.vstack(all_labels)
        self.eeg_data['all'] = {'trials': all_data, 'labels': all_labels}

        # Print shapes of full dataset
        print(f"\nFull dataset shape - Data: {all_data.shape}, Labels: {all_labels.shape}")
        
        # Use GroupKFold for subject-level splits
        gkf = GroupKFold(n_splits=self.n_splits)
        for fold, (train_indices, test_indices) in enumerate(gkf.split(all_data, groups=subject_ids), start=1):
            # Find unique subjects in train and test for the fold
            train_subjects = set(np.array(subject_ids)[train_indices])
            test_subjects = set(np.array(subject_ids)[test_indices])
            
            # Store indices for this fold
            self.eeg_data[fold] = {'train_indices': train_indices, 'test_indices': test_indices}
            
            # Print debugging information for each fold
            print(f"\nFold {fold}")
            print(f"Training subjects: {sorted(train_subjects)}")
            print(f"Testing subjects: {sorted(test_subjects)}")
            print(f"Number of training samples: {len(train_indices)}, Number of testing samples: {len(test_indices)}")

        logging.info("Subject-level k-fold cross-validation indices created.")
        return self.eeg_data

class oldSEEDIVLoader:
    def __init__(self, filepath, num_participants=15, window_size=8, stride=4):
        """
        Initializes the SEEDIVLoader class with paths and parameters.
        :param filepath: The root directory where the EEG dataset is stored.
        :param num_participants: Total number of participants.
        :param window_size: Size of each sliding window segment (in samples).
        :param stride: Step size for the sliding window.
        :param downsample_rate: Target downsampling rate (128Hz in this case).
        :param test_size: The proportion of the data to use as test data.
        :param batch_size: The batch size for training.
        """
        self.filepath = filepath
        self.num_participants = num_participants
        self.sample_freq = 200
        self.target_sample_freq = 128
        self.window_size = window_size * self.target_sample_freq  # Convert window size from seconds to samples
        self.stride = stride * self.target_sample_freq  # Convert stride from seconds to samples
        self.batch_size = 16
        self.eeg_data = {}

    def normalize_channels(self, trials):
        """
        Normalize each channel in the trials using StandardScaler.

        Parameters:
        trials (numpy.ndarray): Input data of shape (n_trials, n_channels, n_timepoints).
                                Each channel of each trial will be normalized independently.

        Returns:
        numpy.ndarray: The normalized trials with the same shape as the input.
        """
        for j in range(trials.shape[1]):  # Iterate over channels
            scaler = StandardScaler()
            # Fit the scaler to the data of the current channel across all trials
            scaler.fit(trials[:, j, :])
            # Transform the data for the current channel
            trials[:, j, :] = scaler.transform(trials[:, j, :])
        return trials
            
    def preprocess_eeg(self, raw_eeg):
        """
        Preprocess the EEG data using CAR, filtering, and resampling to 128 Hz.
        """

        # Resample the EEG data from 200 Hz to 128 Hz
        num_samples = int(raw_eeg.shape[1] * self.target_sample_freq / self.sample_freq)
        resampled_eeg = signal.resample(raw_eeg, num_samples, axis=1)
        # print(f"Resampled EEG shape: {resampled_eeg.shape} at {self.target_sample_freq} Hz")

        return resampled_eeg

    def sliding_windows(self, eeg_data):
        """
        Generate sliding window segments from EEG data.
        """
        # logging.info(f"Generating sliding windows with window size {self.window_size} and stride {self.stride}")

        windows = []
        num_channels, total_length = eeg_data.shape
        num_windows = (total_length - self.window_size) // self.stride + 1

        if num_windows <= 0:
            logging.warning(f"Not enough data to generate windows. Total length: {total_length}, window size: {self.window_size}")
            return np.array([])

        # logging.info(f"Number of windows to generate: {num_windows}")

        for start in range(0, total_length - self.window_size + 1, self.stride):
            end = start + self.window_size
            segment = eeg_data[:, start:end]
            windows.append(segment)

        windows = np.array(windows)

        # Shuffle windows within this trial
        np.random.shuffle(windows)

        return windows

    def process_file(self, file_name, labels):
        """
        Processes a single .mat file, preprocesses the EEG data, and generates sliding windows.
        :param file_name: Path to the .mat file.
        :param labels: The list of labels for each trial.
        :return: Sliding windows and labels.
        """
        keys_to_ignore = ['__header__', '__version__', '__globals__']
        mat = io.loadmat(file_name)

        # Inspect the keys to find the correct trial keys
        trial_keys = [key for key in mat.keys() if key not in keys_to_ignore]
        
        # This assumes the keys are structured as 'prefix_eeg{trial}'.
        # Extract the prefix from the first available key
        first_trial_key = trial_keys[0]  # Assumes there's at least one trial
        prefix = first_trial_key.split('_eeg')[0]  # Get the prefix (e.g., 'cz' or 'ha')
    
        # logging.info(f"Detected prefix: {prefix} for file: {file_name}")

        windows_list = []
        label_list = []
        labelcount = 0

        for trial in range(1, 25):
            trial_name = f'{prefix}_eeg{trial}'  # Construct the dynamic trial name
            if trial_name in mat:
                trial_data = mat[trial_name]
                # Preprocess the EEG data (including downsampling)
                preprocessed_eeg = self.preprocess_eeg(trial_data)
                windows = self.sliding_windows(preprocessed_eeg)

                if windows.size > 0:  # Check if windows were generated
                    windows_list.append(windows)
                    label_list += [labels[labelcount]] * windows.shape[0]
                    # logging.info(f"Trial {trial_name}: Generated {windows.shape[0]} windows with label {labels[labelcount]}")
                else:
                    logging.warning(f"Trial {trial_name} did not generate any valid windows.")
                labelcount += 1

        if windows_list:
            windows_array = np.concatenate(windows_list, axis=0)
            label_array = np.array(label_list)
            # print(f"Concatenated windows shape: {windows_array.shape}, labels shape: {label_array.shape}")

            # Convert labels to one-hot encoded format
            label_array = self.get_labels(label_array)

            return windows_array, label_array
        else:
            print(f"No valid data for this subject.")
            return None, None

    def get_labels(self, label_array):
        """
        Converts a list of labels into one-hot encoded format.
        :param label_array: Array of labels to convert.
        :return: One-hot encoded label array.
        """
        return tf.keras.utils.to_categorical(label_array)

    def get_session_labels(self):
        """
        Returns session-specific labels for 3 sessions.
        """
        session1_label = np.array([1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3])
        session2_label = np.array([2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1])
        session3_label = np.array([1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0])
        return session1_label, session2_label, session3_label

    def get_participant_files(self, participant_num):
        """
        Retrieves the .mat files for a specific participant.
        """
        participant_prefix = f"{participant_num}_"
        participant_files = []
        for root, _, files in os.walk(self.filepath):
            for file_name in files:
                if file_name.startswith(participant_prefix):
                    participant_files.append(os.path.join(root, file_name))
        return participant_files

    def get_session_number(self, file_path):
        """
        Extracts session number from the file path (based on folder structure).
        """
        session_number = int(file_path.split('/')[-2])  # Assumes session number is in the directory name
        return session_number

    def create_tf_datasets(self, windows_array, label_array):
        """
        Create TensorFlow datasets from the windows and labels.
        :param windows_array: Numpy array of EEG windows.
        :param label_array: Numpy array of corresponding labels.
        :return: Train and test datasets as TensorFlow Dataset objects.
        """
        print(f"Splitting data into train and test datasets...")

        # Split the windows and labels into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            windows_array, label_array, test_size=0.2, random_state=None)
        
        # Normalize the channels in the resampled EEG data
        X_train = self.normalize_channels(X_train)
        X_test = self.normalize_channels(X_test)

        logging.info(f"Train dataset shape: {X_train.shape}, Test dataset shape: {X_test.shape}")
        logging.info(f"Train labels shape: {y_train.shape}, Test labels shape: {y_test.shape}")

        # Create TensorFlow datasets from the arrays
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

        # Shuffle and batch the datasets
        train_dataset = train_dataset.shuffle(buffer_size=10000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        test_dataset = test_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        # print(f"Train dataset shape: {X_train.shape}, Test dataset shape: {X_test.shape}")
        return train_dataset, test_dataset

    def process_participant(self, participant_num):
        """
        Processes the data for a single participant across 3 sessions.
        :param participant_num: The participant number to process.
        :return: Processed TensorFlow datasets (train and test) for the participant.
        """
        session1_label, session2_label, session3_label = self.get_session_labels()
        participant_files = self.get_participant_files(participant_num)
        sorted_files = sorted(participant_files, key=self.get_session_number)

        # Process session 1, 2, and 3 for the participant
        windows_list, labels_list = [], []
        for session_num in range(1, 4):
            file_path = sorted_files[session_num - 1]
            
            # Process each session's trials based on session-specific labels
            if session_num == 1:
                windows, labels = self.process_file(file_path, session1_label)
            elif session_num == 2:
                windows, labels = self.process_file(file_path, session2_label)
            elif session_num == 3:
                windows, labels = self.process_file(file_path, session3_label)

            # Split windows and labels into trials for shuffling
            num_trials = len(session1_label)  # Assumes each session has the same number of trials (24)
            trial_size = windows.shape[0] // num_trials  # Number of windows per trial
            
            # Separate windows and labels into individual trials
            trial_windows = [windows[i * trial_size:(i + 1) * trial_size] for i in range(num_trials)]
            trial_labels = [labels[i * trial_size:(i + 1) * trial_size] for i in range(num_trials)]
            
            # Shuffle trials within this session
            trial_indices = np.arange(num_trials)
            np.random.shuffle(trial_indices)
            
            # Apply the shuffle to windows and labels
            trial_windows = [trial_windows[i] for i in trial_indices]
            trial_labels = [trial_labels[i] for i in trial_indices]
            
            # Concatenate shuffled trials back into session-level arrays
            session_windows = np.concatenate(trial_windows, axis=0)
            session_labels = np.concatenate(trial_labels, axis=0)
            
            # Append session data to participant-level lists
            windows_list.append(session_windows)
            labels_list.append(session_labels)

        # Combine all session windows and labels
        windows_array = np.vstack(windows_list)
        labels_array = np.concatenate(labels_list)

        logging.info(f"Final concatenated windows shape for participant {participant_num}: {windows_array.shape}")
        logging.info(f"Final concatenated labels shape for participant {participant_num}: {labels_array.shape}")

        # Ensure the length of windows and labels match
        assert len(windows_array) == len(labels_array), "Mismatch between windows and labels length."

        # Create TensorFlow datasets (train and test)
        train_dataset, test_dataset = self.create_tf_datasets(windows_array, labels_array)

        return train_dataset, test_dataset

    def load_dataset(self):
        """
        Loads and processes the dataset for all participants and stores the TensorFlow datasets.
        """
        print("Starting dataset loading and preprocessing...")

        for participant_num in range(1, self.num_participants + 1):
            print(f"Processing Participant {participant_num}")
            train_dataset, test_dataset = self.process_participant(participant_num)

            self.eeg_data[participant_num] = {
                'train_ds': train_dataset,
                'test_ds': test_dataset
            }

            print(f"Completed Participant {participant_num}")

        print("All participants processed.")
        return self.eeg_data

class CHBMITLoader:
    def __init__(self, filepath):
        self.base_path = filepath
        self.eeg_data = {}
        self.batch_size = 16
        self.sfreq = 128

    def extract_data_and_labels(self, edf_filename, summary_text):
        folder, basename = os.path.split(edf_filename)

        # Load EDF file with MNE
        try:
            edf = mne.io.read_raw_edf(edf_filename, stim_channel=None, preload=True, verbose='ERROR')
        except ValueError as e:
            print(f"Error reading {edf_filename} with MNE: {e}")
            return None, None, None
        
        sfreq = edf.info['sfreq']
        edf.resample(sfreq=self.sfreq, verbose=False)

        selected_channels = [
            'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
            'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
            'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
            'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
            'FZ-CZ', 'CZ-PZ'
        ]

        # Ensure selected channels are present in the data
        missing_channels = [ch for ch in selected_channels if ch not in edf.ch_names]
        if missing_channels:
            print(f"Missing channels in {edf_filename}: {missing_channels}. Skipping...")
            return None, None, None
        
        edf.pick_channels(selected_channels)
        X = edf.get_data().astype(np.float32) * 1e6  # Convert to µV
        y = np.zeros(X.shape[1], dtype=np.int64)

        # Extract information from summary_text
        print(f"Processing file: {basename}, Sampling frequency: {self.sfreq} Hz")

        # Extract seizure details for the current file
        seizure_start_times = []
        seizure_end_times = []

        lines = summary_text.splitlines()
        is_target_file = False

        for line in lines:
            # Detect the target file section in the summary
            if line.startswith(f"File Name: {basename}"):
                is_target_file = True
            elif line.startswith("File Name:") and is_target_file:
                # If we reach a new file name and were in the target section, stop parsing
                break

            if is_target_file:
                # Extract seizure start and end times
                start_times = re.findall(r"Seizure\s*\d*\s*Start Time:\s*([\d]+)\s*seconds", line)
                end_times = re.findall(r"Seizure\s*\d*\s*End Time:\s*([\d]+)\s*seconds", line)
                seizure_start_times.extend(start_times)
                seizure_end_times.extend(end_times)

        # Ensure there are matching start and end times
        if len(seizure_start_times) != len(seizure_end_times):
            print(f"Mismatched seizure start and end times in {basename}. Skipping...")
            return None, None, None

        # Process each seizure and update the label array
        for start_str, end_str in zip(seizure_start_times, seizure_end_times):
            start_sec = int(start_str)
            end_sec = int(end_str)
            print(f"Seizure detected from {start_sec}s to {end_sec}s in file {basename}.")

            i_seizure_start = int(round(start_sec * sfreq))
            i_seizure_stop = int(round((end_sec + 1) * sfreq))
            y[i_seizure_start:min(i_seizure_stop, len(y))] = 1

        if not seizure_start_times:
            print(f"No seizures detected in file: {basename}.")

        assert X.shape[1] == len(y)
        return X, y, sfreq

    def epoch_and_segment(self, X, y, epoch_length=5, overlap=4, is_test=False):
        """
        Segments EEG data into overlapping or non-overlapping epochs.

        Parameters:
        - X: EEG data array (channels x time samples)
        - y: Label array (1 for seizure, 0 for non-seizure)
        - epoch_length: Length of each segment in seconds
        - overlap: Overlap in seconds between consecutive windows
        - is_test: If True, segments all data into non-overlapping windows

        Returns:
        - X_final: Segmented EEG data (epochs x channels x time samples)
        - y_final: Corresponding labels (epochs,)
        """
        samples_per_epoch = int(epoch_length * self.sfreq)
        step_size = samples_per_epoch if is_test else int(epoch_length * self.sfreq - overlap * self.sfreq)

        X_final = []
        y_final = []

        for start_idx in range(0, X.shape[1] - samples_per_epoch + 1, step_size):
            end_idx = start_idx + samples_per_epoch
            X_epoch = X[:, start_idx:end_idx]
            y_epoch = y[start_idx:end_idx]

            if is_test:
                # For test, append all segments (seizure or non-seizure)
                X_final.append(X_epoch)
                y_final.append(1 if np.any(y_epoch) else 0)
            else:
                # For train, append seizure and non-seizure segments
                if np.any(y_epoch):  # Seizure-related
                    X_final.append(X_epoch)
                    y_final.append(1)
                elif not np.any(y_epoch):  # Non-seizure
                    X_final.append(X_epoch)
                    y_final.append(0)

        if not is_test:
            # Shuffle the training segments
            combined = list(zip(X_final, y_final))
            random.shuffle(combined)
            X_final, y_final = zip(*combined)

        return np.array(X_final), np.array(y_final)



    def create_tf_datasets(self, windows_array, label_array):
        print(f"Splitting data into train and test datasets...")
        train_dataset = tf.data.Dataset.from_tensor_slices((windows_array['train'], label_array['train']))
        test_dataset = tf.data.Dataset.from_tensor_slices((windows_array['test'], label_array['test']))

        train_dataset = train_dataset.shuffle(10000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        test_dataset = test_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        return train_dataset, test_dataset

    def load_dataset(self):
        for subject_folder in sorted(glob(os.path.join(self.base_path, "chb*"))):
            filename = os.path.basename(subject_folder)
            subject_id = filename[-2:]

            edf_file_names = sorted(glob(os.path.join(subject_folder, "*.edf")))
            summary_file = os.path.join(subject_folder, f"{filename}-summary.txt")

            if not os.path.exists(summary_file):
                print(f"Warning: Summary file missing for {filename}. Skipping...")
                continue

            summary_content = open(summary_file, 'r').read()

            # Reserve one file for testing, process others for training
            test_file_name = random.choice(edf_file_names)
            train_file_names = [f for f in edf_file_names if f != test_file_name]

            windows_train, labels_train = [], []
            windows_test, labels_test = [], []

            for edf_file_name in edf_file_names:
                X, y, sfreq = self.extract_data_and_labels(edf_file_name, summary_content)
                if X is None or y is None or sfreq is None:
                    continue

                X_epochs, y_epochs = self.epoch_and_segment(X, y)

                if edf_file_name == test_file_name:
                    windows_test.append(X_epochs)
                    labels_test.append(y_epochs)
                else:
                    windows_train.append(X_epochs)
                    labels_train.append(y_epochs)

            if not windows_train or not windows_test:
                print(f"Insufficient data for subject {subject_id}. Skipping...")
                continue

            windows_train = np.vstack(windows_train)
            labels_train = np.hstack(labels_train)
            windows_test = np.vstack(windows_test)
            labels_test = np.hstack(labels_test)

            train_ds, test_ds = self.create_tf_datasets(
                {"train": windows_train, "test": windows_test},
                {"train": labels_train, "test": labels_test}
            )

            self.eeg_data[subject_id] = {
                'train_ds': train_ds,
                'test_ds': test_ds
            }

        return self.eeg_data

class SienaLoader:
    def __init__(self, filepath):
        self.base_path = filepath
        self.batch_size = 16
        self.eeg_data = {}
    
    def parse_timestamp(self, string: str) -> float:
        match = re.search(r"([0-9]{2}[:.][0-9]{2}[:.][0-9]{2})", string)
        if not match:
            raise ValueError("No valid timestamp found in the string.")
        time_str = match.group(1).replace(":", ".")
        return time.mktime(time.strptime(time_str, "%H.%M.%S"))

    def subtract_timestamps(self, time1: float, time2: float) -> float:
        difference = time1 - time2
        if difference < 0:
            difference += 24 * 60 * 60  # Add a full day in seconds
        return difference

    def calculate_seconds(self, start_time_str, event_time_str):
        start_time = self.parse_timestamp(start_time_str)
        event_time = self.parse_timestamp(event_time_str)
        return self.subtract_timestamps(event_time, start_time)

    def extract_data_and_labels(self, edf_filename, summary_content):
        folder, basename = os.path.split(edf_filename)
        
        # Read the EDF file using mne
        edf = mne.io.read_raw_edf(edf_filename, preload=True, stim_channel=None)
        
        # Resample to reduce the computational load if needed
        edf.resample(sfreq=128)

        # Select only EEG channels
        eeg_channels = ['EEG Fp1', 'EEG F3', 'EEG C3', 'EEG P3', 'EEG O1', 'EEG F7', 'EEG T3', 'EEG T5', 'EEG Fc1', 'EEG Fc5', 'EEG Cp1', 'EEG Cp5', 'EEG F9', 'EEG Fz', 'EEG Pz', 'EEG F4', 'EEG C4', 'EEG P4', 'EEG O2', 'EEG F8', 'EEG T4', 'EEG T6', 'EEG Fc2', 'EEG Fc6', 'EEG Cp2', 'EEG Cp6', 'EEG F10'] #'EEG Cz', 
        edf.pick_channels(eeg_channels)

        # Get the data as a numpy array and convert units to microvolts
        X = edf.get_data().astype(np.float32) * 1e6  # to µV
        y = np.zeros(X.shape[1], dtype=np.int64)
        sfreq = edf.info['sfreq']  # Get the sampling frequency

        lines = summary_content.splitlines()

        # Variables to track the current filename and registration start time
        current_file = None
        registration_start_time_str = None
        seizure_start_times = []
        seizure_end_times = []

        # Iterate over each line
        for line in lines:
            # Check if the line is specifying a file name
            match_file = re.search(r"File name:\s*(.*\.edf)", line)
            if match_file:
                current_file = match_file.group(1).strip()

            # If we are on the target file, process registration and seizures
            if current_file == basename:
                # Extract registration start time
                if "Registration start time" in line:
                    registration_start_time_str = re.search(r"Registration start time:\s*(\d{2}[:.]\d{2}[:.]\d{2})", line).group(1)

                # Extract seizure start and end times
                if "Seizure start time" in line or "Start time" in line:
                    seizure_start_times = re.findall(r"(\d{2}[:.]\d{2}[:.]\d{2})", line)

                if "Seizure end time" in line or "End time" in line:
                    seizure_end_times = re.findall(r"(\d{2}[:.]\d{2}[:.]\d{2})", line)
                    print("Found Seizure end time")

                    if seizure_start_times and seizure_end_times:
                        for start_str, end_str in zip(seizure_start_times, seizure_end_times):
                            print(f"start_str: {start_str}, end_str: {end_str}")
                            # Calculate start and end times in seconds relative to the registration start time
                            start_sec = self.calculate_seconds(registration_start_time_str, start_str)
                            end_sec = self.calculate_seconds(registration_start_time_str, end_str)
                            print(f"Seizure detected from {start_sec}s to {end_sec}s.")

                            # Calculate seizure indices
                            i_seizure_start = int(round(start_sec * sfreq))
                            i_seizure_stop = int(round((end_sec + 1) * sfreq))

                            y[i_seizure_start:i_seizure_stop] = 1         
                    else:
                        print("No seizure detected in this file.")

        assert X.shape[1] == len(y)
        return X, y, sfreq

    def epoch_split(self,X, y, sfreq, epoch_length=1, overlap=0.5): 
        """
        Split EEG data into overlapping epochs.
        
        Args:
            X (np.array): The EEG data of shape (n_channels, n_samples).
            y (np.array): The labels of shape (n_samples,).
            sfreq (float): Sampling frequency in Hz.
            epoch_length (int or float): Length of each epoch in seconds.
            overlap (float): Fractional overlap between consecutive epochs (e.g., 0.5 for 50% overlap).
        
        Returns:
            np.array: Array of shape (n_epochs, n_channels, samples_per_epoch) representing the segmented epochs.
            np.array: Array of shape (n_epochs, 2) representing the one-hot encoded labels for each epoch.
        """
        # Calculate the number of samples per epoch
        samples_per_epoch = int(epoch_length * sfreq)
        # Calculate step size between epochs based on overlap
        step_size = int(samples_per_epoch * (1 - overlap))
        
        X_final = []
        y_final = []
        
        # Loop through the data to extract epochs
        for start_idx in range(0, X.shape[1] - samples_per_epoch + 1, step_size):
            end_idx = start_idx + samples_per_epoch
            X_epoch = X[:, start_idx:end_idx]
            y_epoch = y[start_idx:end_idx]
            
            # Append the epoch data to the final list
            X_final.append(X_epoch)
            # Assign label [1, 0] if seizure (y contains at least one 1), otherwise [0, 1]
            y_final.append([1, 0] if np.any(y_epoch) else [0, 1])

            # Print out details of the epochs for debugging purposes
            # if np.any(y_epoch):
            #     print(f"Epoch {start_idx // step_size + 1}: X_epoch shape: {X_epoch.shape}, y_epoch label: {y_final[-1]}")
        
        return np.array(X_final), np.array(y_final)


    def create_tf_datasets(self, windows_array, label_array):
        X_train, X_test, y_train, y_test = train_test_split(
            windows_array, label_array, test_size=0.2, random_state=42)

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

        train_dataset = train_dataset.shuffle(buffer_size=10000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        test_dataset = test_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        print(f"Train dataset shape: {X_train.shape}, Test dataset shape: {X_test.shape}")
        return train_dataset, test_dataset

    def load_dataset(self):
        for subject_folder in sorted(glob(os.path.join(self.base_path, "PN*"))):
            subject_id = os.path.basename(subject_folder)
            print(f"\nProcessing subject folder: {subject_folder}")
            print(f"Subject ID: {subject_id}")

            edf_file_names = sorted(glob(os.path.join(subject_folder, "*.edf")))
            if not edf_file_names:
                print(f"Warning: No EDF files found for subject {subject_id}. Skipping...")
                continue

            print(f"Found {len(edf_file_names)} EDF files for subject {subject_id}")
                    
            summary_file = os.path.join(subject_folder, f"Seizures-list-{subject_id}.txt")

            if not os.path.exists(summary_file):
                print(f"Warning: Summary file missing for {subject_id}. Skipping...")
                continue

            print(f"Summary file found for subject {subject_id}: {summary_file}")

            with open(summary_file, 'r') as f:
                summary_content = f.read()

            # Apply regex correction for subject 6
            if subject_id == "PN06":
                print(f"Applying correction to summary content for subject {subject_id}")
                summary_content = re.sub(r'PNO(\d+)-', r'PN0\1-', summary_content)

            if subject_id == "PN11":
                print(f"Applying correction to summary content for subject {subject_id}")
                summary_content = re.sub(r'(PN11)-\.edf', r'\1-1.edf', summary_content)

            if subject_id == "PN01":
                summary_content = re.sub(r'(PN01)\.edf', r'\1-1.edf', summary_content)

            if subject_id == "PN10":
                summary_content = re.sub(r'1\s6[.:]49[.:]25', '16.49.25', summary_content)

            all_X = []
            all_y = []
            overlap = 0.5
            epoch_length = 1

            for edf_file_name in edf_file_names:
                X, y, sfreq = self.extract_data_and_labels(edf_file_name, summary_content)
                X_epochs, y_epochs = self.epoch_split(X, y, sfreq, epoch_length, overlap)
                all_X.append(X_epochs)
                all_y.append(y_epochs)

            # Combine all epochs into final arrays
            all_X = np.vstack(all_X)
            all_y = np.vstack(all_y)

            print(f"Final X shape for subject {subject_id}: {all_X.shape}, Final y shape: {all_y.shape}")

            train_dataset, test_dataset = self.create_tf_datasets(all_X, all_y)

            self.eeg_data[subject_id] = {
                'train_ds': train_dataset,
                'test_ds': test_dataset
            }

        return self.eeg_data

class EEGMATLoader:
    def __init__(self, filepath):
        self.base_path = filepath
        self.original_fs = 500
        self.target_fs = 128
        self.segment_length_sec = 2
        self.overlap_sec = 1
        self.batch_size = 16
        self.eeg_data = {}
        self.n_splits = 3

    def load_edf(self, file_path):
        """Load an EDF file and return the signal data."""
        f = pyedflib.EdfReader(file_path)
        n = f.signals_in_file
        sigbufs = np.zeros((n, f.getNSamples()[0]))
        for i in np.arange(n):
            sigbufs[i, :] = f.readSignal(i)
        f._close()
        del f
        return sigbufs

    def downsample(self, data):
        """Downsample the data from the original frequency to the target frequency."""
        num_samples = int(data.shape[1] * self.target_fs / self.original_fs)
        downsampled_data = resample(data, num_samples, axis=1)
        return downsampled_data

    def normalize_channels(self, trials):
        """
        Normalize each channel in the trials using StandardScaler.

        Parameters:
        trials (numpy.ndarray): Input data of shape (n_trials, n_channels, n_timepoints).
                                Each channel of each trial will be normalized independently.

        Returns:
        numpy.ndarray: The normalized trials with the same shape as the input.
        """
        for j in range(trials.shape[1]):  # Iterate over channels
            scaler = StandardScaler()
            # Fit the scaler to the data of the current channel across all trials
            scaler.fit(trials[:, j, :])
            # Transform the data for the current channel
            trials[:, j, :] = scaler.transform(trials[:, j, :])
        return trials
    
    def segment_data(self, data):
        """Segment the data into overlapping windows."""
        segment_length = int(self.segment_length_sec * self.target_fs)
        overlap = int(self.overlap_sec * self.target_fs)
        step = segment_length - overlap

        segments = []
        for start in range(0, data.shape[1] - segment_length + 1, step):
            segments.append(data[:, start:start + segment_length])
        return np.array(segments)

    def load_dataset(self):
        """Load and process EEG data for all subjects, storing data for k-fold cross-validation."""
        # Identify all subjects based on the directory contents
        subject_files = sorted([f for f in os.listdir(self.base_path) if f.endswith('.edf')])
        subjects = set([f.split('_')[0] for f in subject_files])  # Extract unique subject IDs

        all_data = []
        all_labels = []
        subject_ids = []  # Used for tracking subjects in GroupKFold

        for subject in subjects:
            print(f"Processing Subject: {subject}")

            resting_path = os.path.join(self.base_path, f"{subject}_1.edf")
            task_path = os.path.join(self.base_path, f"{subject}_2.edf")

            if not os.path.exists(resting_path) or not os.path.exists(task_path):
                print(f"Warning: Missing files for {subject}. Skipping this subject...")
                continue

            eeg_data_resting = self.load_edf(resting_path)
            eeg_data_task = self.load_edf(task_path)
            print(f"Loaded data: Resting state: {eeg_data_resting.shape}, Task state: {eeg_data_task.shape}")

            # Downsample the data
            downsampled_eeg_resting = self.downsample(eeg_data_resting)
            downsampled_eeg_task = self.downsample(eeg_data_task)

            # Trim data for segmentation
            resting_trimmed = downsampled_eeg_resting[:, :23040]
            task_trimmed = downsampled_eeg_task[:, :7680]

            # Segment data
            resting_segments = self.segment_data(resting_trimmed)
            task_segments = self.segment_data(task_trimmed)
            print(f"Segmented data: Resting state: {resting_segments.shape}, Task state: {task_segments.shape}")

            # Concatenate resting and task data and create labels
            X = np.concatenate((resting_segments, task_segments), axis=0)

            # Normalize channels across all trials for the subject
            X = self.normalize_channels(X)  # Apply normalization here

            y = np.array([0] * resting_segments.shape[0] + [1] * task_segments.shape[0])

            # Convert labels to one-hot encoded format
            y = np_utils.to_categorical(y, num_classes=2)

            # Append data to the global arrays
            all_data.append(X)
            all_labels.append(y)
            subject_ids.extend([subject] * X.shape[0])  # Associate each segment with the subject ID

        # Concatenate all subject data into one array
        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Store the full dataset in 'all' for cross-validation access
        self.eeg_data['all'] = {
            'trials': all_data,
            'labels': all_labels
        }
        print(f"\nFull dataset shape - Data: {all_data.shape}, Labels: {all_labels.shape}")

        # Use GroupKFold to split by subject
        gkf = GroupKFold(n_splits=self.n_splits)
        for fold, (train_indices, test_indices) in enumerate(gkf.split(all_data, groups=subject_ids), start=1):
            train_subjects = set(np.array(subject_ids)[train_indices])
            test_subjects = set(np.array(subject_ids)[test_indices])

            # Store only indices for each fold
            self.eeg_data[fold] = {
                'train_indices': train_indices,
                'test_indices': test_indices
            }

            # Print fold information for verification
            print(f"\nFold {fold}")
            print(f"Training subjects: {sorted(train_subjects)}")
            print(f"Testing subjects: {sorted(test_subjects)}")
            print(f"Number of training samples: {len(train_indices)}, Number of testing samples: {len(test_indices)}")

        return self.eeg_data

class BCICIII2Loader:
    def __init__(self, filepath):
        self.base_path = filepath
        self.subjects = ['A', 'B']
        self.original_fs = 240
        self.target_fs = 128
        self.epoch_duration_sec = 0.667
        self.batch_size = 16
        self.eeg_data = {}

    def _load_data(self, data, true_labels, is_train=True):
        logging.info("Loading %s data", "training" if is_train else "testing")
        
        signals = data['Signal']
        flashing = data['Flashing']
        stimulus_code = data['StimulusCode']
        word = data['TargetChar'] if 'TargetChar' in data.keys() else true_labels
        stimulus_type = data['StimulusType'] if is_train and 'StimulusType' in data.keys() else ""

        # Downsample signals to the target frequency (128 Hz)
        signals = self._downsample(signals)

        signal_duration = len(signals) * len(signals[0]) / (self.original_fs * 60)
        trials = len(word[0])
        samples_per_trial = len(signals[0])

        logging.info("Sampling Frequency: %d Hz", self.original_fs)
        logging.info("Session duration: %.2f minutes", signal_duration)
        logging.info("Number of letters: %d", trials)

        return signals, flashing, word, stimulus_type, stimulus_code, signal_duration, trials, samples_per_trial

    def _downsample(self, signals):
        downsample_factor = self.original_fs // self.target_fs
        return signals[:, ::downsample_factor, :]

    def split_epochs_train(self, signal_train, flashing_train, stimulus_type_train, trials_train, samples_per_trial_train):
        """Split and filter epochs from the training dataset and return various epoch groups."""
        num_channels = signal_train.shape[2]
        epoch_duration_samples = int(self.epoch_duration_sec * self.target_fs)

        # Prepare Butterworth filter
        butter_order = 8
        f_cut_low = 0.1
        f_cut_high = 20
        sos = signal.butter(butter_order, [f_cut_low, f_cut_high], 'bandpass', fs=self.target_fs, output='sos')

        epochs_train = []     # Filtered epochs
        labels_train = []     # Corresponding labels
        epochs_P300 = []      # P300 epochs (filtered)
        epochs_noP300 = []    # No P300 epochs (filtered)
        epochs_P300_unF = []  # P300 epochs (unfiltered)

        positive_epochs = 0   # Counter for P300 epochs
        negative_epochs = 0   # Counter for no P300 epochs

        # Iterate over the entire signal
        for trial in tqdm(range(trials_train), desc='Splitting Train Epochs'):
            for sample in range(samples_per_trial_train):
                if sample == 0 or (flashing_train[trial, sample - 1] == 0 and flashing_train[trial, sample] == 1):
                    lower_sample = sample
                    upper_sample = sample + epoch_duration_samples

                    if upper_sample <= signal_train.shape[1]:
                        # Extract epoch and apply filtering
                        epoch = signal_train[trial, lower_sample:upper_sample, :]
                        epochs_P300_unF.append(epoch)  # Save the unfiltered version of the signal
                        filtered_epoch = signal.sosfiltfilt(sos, epoch, axis=0)  # Apply filtering
                        epochs_train.append(filtered_epoch)
                        # epochs_train.append(epoch)

                        # Label and save epochs
                        if stimulus_type_train[trial, sample] == 1:
                            positive_epochs += 1
                            labels_train.append(1)
                            epochs_P300.append(filtered_epoch)
                            # epochs_P300.append(epoch)

                            # Replicate P300 epoch 4 times to maintain balance
                            for _ in range(4):
                                positive_epochs += 1
                                labels_train.append(1)
                                epochs_train.append(filtered_epoch)
                                # epochs_train.append(epoch)
                        else:
                            negative_epochs += 1
                            labels_train.append(0)
                            epochs_noP300.append(filtered_epoch)
                            # epochs_noP300.append(epoch)
                            

        # Compute average positive and negative epochs
        average_positive_epoch = np.mean(epochs_P300, axis=0) if len(epochs_P300) > 0 else None
        average_negative_epoch = np.mean(epochs_noP300, axis=0) if len(epochs_noP300) > 0 else None

        # Convert lists to numpy arrays for further analysis
        epochs_train = np.array(epochs_train).transpose(0, 2, 1)
        labels_train = np_utils.to_categorical(labels_train, 2)
        epochs_P300 = np.array(epochs_P300)
        epochs_noP300 = np.array(epochs_noP300)
        epochs_P300_unF = np.array(epochs_P300_unF)

        # Print unbalance ratio
        unbalance_ratio = negative_epochs / positive_epochs if positive_epochs != 0 else float('inf')
        print(f'Unbalance ratio {round(unbalance_ratio, 2)}')

        return epochs_train, labels_train, epochs_P300, epochs_noP300, epochs_P300_unF

    def split_epochs_test(self, signals, flashing, trials, samples_per_trial):
        """Split the testing data into epochs with filtering."""
        num_channels = signals.shape[2]
        epoch_duration_samples = int(self.epoch_duration_sec * self.target_fs)

        epochs_test = []
        onset_test = []

        for trial in tqdm(range(trials), desc='Splitting Test Epochs'):
            for sample in range(samples_per_trial):
                if sample == 0 or (flashing[trial, sample - 1] == 0 and flashing[trial, sample] == 1):
                    lower_sample = sample
                    upper_sample = sample + epoch_duration_samples

                    if upper_sample <= signals.shape[1]:
                        onset_test.append(sample)
                        epoch = signals[trial, lower_sample:upper_sample, :]
                        epochs_test.append(epoch)

        return np.array(epochs_test).transpose(0, 2, 1)

    def stimCode_reduced(self, stim_code):
        """Reduce stimulus codes by removing consecutive duplicates and zeros."""
        new_stim_code = []
        for i in range(len(stim_code[:, 0])):  # Iterate for all trials (letters) in the session
            index = np.where(np.diff(stim_code[i]) == 0)
            new_temp_stimcode = np.delete(stim_code[i], index)  # Remove consecutive duplicates
            index = np.where(new_temp_stimcode == 0)
            new_temp_stimcode = np.delete(new_temp_stimcode, index)  # Remove zeros
            new_stim_code.append(new_temp_stimcode)
        return np.array(new_stim_code)

    def target_test(self, stimulus_code_test, word_test):
        """Transform target characters into stimulus type array for the test set."""
        logging.info("Transforming target characters into stimulus type array")

        # Experimental grid
        MA = [['A', 'B', 'C', 'D', 'E', 'F'],
            ['G', 'H', 'I', 'J', 'K', 'L'],
            ['M', 'N', 'O', 'P', 'Q', 'R'],
            ['S', 'T', 'U', 'V', 'W', 'X'],
            ['Y', 'Z', '1', '2', '3', '4'],
            ['5', '6', '7', '8', '9', '_']]
        
        MA_array = np.array(MA)
        stimulus_code_reduced = self.stimCode_reduced(stimulus_code_test)

        target_test_labels = []

        # Iterate over each trial (letter) in the test set
        for k in range(len(word_test)):
            i = np.where(MA_array == word_test[k])[0][0] + 7
            j = np.where(MA_array == word_test[k])[1][0] + 1

            # Iterate over the reduced stimulus codes for each trial
            for epoch_index in range(len(stimulus_code_reduced[k])):
                if stimulus_code_reduced[k][epoch_index] == i or stimulus_code_reduced[k][epoch_index] == j:
                    target_test_labels.append(1)  # P300 (target stimulus)
                else:
                    target_test_labels.append(0)  # No P300 (non-target stimulus)

        # Convert to categorical to match model output requirements
        return np_utils.to_categorical(np.array(target_test_labels), 2)


    def _to_tf_dataset(self, epochs, labels, train=True):
        """Convert epochs and labels to TensorFlow dataset."""
        dataset = tf.data.Dataset.from_tensor_slices((epochs, labels))

        if train:
            # Shuffle the dataset with reshuffle each iteration for training
            dataset = dataset.shuffle(buffer_size=10000, reshuffle_each_iteration=True).batch(self.batch_size)
        else:
            # Batch the dataset without shuffling for evaluation
            dataset = dataset.batch(self.batch_size)

        # Prefetch to improve pipeline performance
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset    
    
    def normalize_channels(self, trials):
        """
        Normalize each channel in the trials using StandardScaler.

        Parameters:
        trials (numpy.ndarray): Input data of shape (n_trials, n_channels, n_timepoints).
                                Each channel of each trial will be normalized independently.

        Returns:
        numpy.ndarray: The normalized trials with the same shape as the input.
        """
        for j in range(trials.shape[1]):  # Iterate over channels
            scaler = StandardScaler()
            # Fit the scaler to the data of the current channel across all trials
            scaler.fit(trials[:, j, :])
            # Transform the data for the current channel
            trials[:, j, :] = scaler.transform(trials[:, j, :])
        return trials    

    def load_dataset(self):
        for subject in self.subjects:
            print(f"Processing subject {subject}")
            logging.info("Loading datasets for subject %s", subject)

            train_path = os.path.join(self.base_path, f'Subject_{subject}_Train.mat')
            test_path = os.path.join(self.base_path, f'Subject_{subject}_Test.mat')
            true_labels_path = os.path.join(self.base_path, f'true_labels_{subject}.txt')

            if not (os.path.exists(train_path) and os.path.exists(test_path) and os.path.exists(true_labels_path)):
                raise FileNotFoundError(f"One or more files are missing for subject {subject}")

            # Load the data from the specified paths
            train_data = io.loadmat(train_path)
            test_data = io.loadmat(test_path)
            test_target = pd.read_csv(true_labels_path, header=None).values.flatten()

            # Load and process training data
            signal_train, flashing_train, word_train, stimulus_type_train, _, _, trials_train, samples_per_trial_train = self._load_data(train_data, test_target, is_train=True)

            # Load and process testing data
            signal_test, flashing_test, word_test, _, stimulus_code_test, _, trials_test, samples_per_trial_test = self._load_data(test_data, test_target, is_train=False)

            # Split and filter training data into epochs
            epochs_train, labels_train, epochs_P300, epochs_noP300, epochs_P300_unF = self.split_epochs_train(
                signal_train, flashing_train, stimulus_type_train, trials_train, samples_per_trial_train
            )
            logging.info(f"Train epochs shape: {epochs_train.shape}, Train labels shape: {labels_train.shape}")
            print(f"TRAINING EPOCHS SUBJECT {subject}")
            print(f"Signal windows tensor shape: {epochs_train.shape}")
            print(f"Labels shape: {labels_train.shape}")
            print(f"p300 shape: {epochs_P300.shape}, noP300 shape: {epochs_noP300.shape}, unfiltered shape: {epochs_P300_unF.shape}")

            # Split and filter testing data into epochs
            epochs_test = self.split_epochs_test(signal_test, flashing_test, trials_test, samples_per_trial_test)
            labels_test = self.target_test(stimulus_code_test, word_test[0])
            logging.info(f"Test epochs shape: {epochs_test.shape}, Test labels shape: {labels_test.shape}")
            print(f"TEST SUBJECT {subject}")
            print(f"Signal windows tensor shape: {epochs_test.shape}")
            print(f"Target test shape: {labels_test.shape}")
            print(f"Target test unique values: {np.unique(labels_test, axis=0)}")

            # Normalize the channels in the training and testing data
            epochs_train = self.normalize_channels(epochs_train)
            epochs_test = self.normalize_channels(epochs_test)

            # Convert to TensorFlow datasets
            self.train_dataset = self._to_tf_dataset(epochs_train, labels_train, train=True)
            self.test_dataset = self._to_tf_dataset(epochs_test, labels_test, train=False)

            # Store the datasets in the dictionary
            self.eeg_data[subject] = {
                'train_ds': self.train_dataset,
                'test_ds': self.test_dataset
            }
            logging.info("Datasets loaded successfully for subject %s", subject)

        return self.eeg_data

class TUHAbnormalLoader:
    def __init__(self, filepath):
        """
        Initializes the TUHAbnormalLoader class with necessary configuration parameters.

        Parameters:
        - filepath (str): Path to the directory containing EEG dataset files.
        """
        self.filepath = filepath
        self.batch_size = 16  # Fixed typo from batch_siie to batch_size
        self.target_freq = 128  # Target frequency to resample the data
        self.sec_to_cut = 60  # Seconds to cut from start and end
        self.selected_ch_names = np.array([
            'A1', 'A2', 'C3', 'C4', 'CZ', 'F3', 'F4', 'F7', 'F8', 'FP1', 'FP2', 'FZ',
            'O1', 'O2', 'P3', 'P4', 'PZ', 'T3', 'T4', 'T5', 'T6'
        ])
        self.eeg_data = {}
        self.eeg_data['all'] = {}

    def segment_epochs(self, data, minutes_to_keep, segment_into_epochs=False):
        """
        Helper function to preprocess data for training or evaluation.

        Parameters:
        - data (np.array): EEG data after channel selection and resampling.
        - minutes_to_keep (int): Number of minutes of data to keep.
        - segment_into_epochs (bool): Whether to segment data into non-overlapping 1-minute epochs.

        Returns:
        - data (np.array): Preprocessed data after cutting and segmentation.
        """
        # Remove the first minute (artifact-prone region)
        if data.shape[1] > int(2 * 60 * self.target_freq):
            data = data[:, int(60 * self.target_freq):]

        # Ensure there is enough data for the specified minutes
        max_timepoints = min(data.shape[1], int(minutes_to_keep * 60 * self.target_freq))
        data = data[:, :max_timepoints]

        if segment_into_epochs:
            # Segment the data into non-overlapping 1-minute epochs
            epoch_length = int(60 * self.target_freq)  # 1 minute in samples
            num_epochs = max_timepoints // epoch_length

            if num_epochs > 0:
                data = np.stack([
                    data[:, i * epoch_length:(i + 1) * epoch_length] for i in range(num_epochs)
                ], axis=0)  # Shape: (num_epochs, n_channels, epoch_length)

                print(f"Data shape after segmenting into non-overlapping 1-minute epochs: {data.shape}")
            else:
                # Not enough data for even one epoch
                print(f"Not enough data to create at least one epoch. Skipping file.")
                return None

        return data

    def preprocess_one_file(self, filename):
        """
        Preprocess an individual EEG file.

        Parameters:
        - filename (str): Path to the EEG file.

        Returns:
        - data (np.array): Preprocessed EEG data.
        - label (np.array): One-hot encoded label (normal or abnormal).
        """
        try:
            print(f"\nProcessing file: {filename}")

            # Load the raw EEG file
            raw = mne.io.read_raw_edf(filename, preload=True, verbose='error')
            self.original_freq = raw.info['sfreq']
            print(f"Original frequency: {self.original_freq} Hz")

            # Clean channel names and select the relevant channels
            cleaned_ch_names = np.array([
                c.replace('EEG ', '').replace('-REF', '') for c in raw.ch_names
            ])
            ch_idxs = np.array([
                np.where(cleaned_ch_names == ch)[0][0] for ch in self.selected_ch_names
            ])
            print(f"Selected {len(ch_idxs)} channels")

            # Resample the data to the target frequency
            raw = raw.load_data().copy().resample(self.target_freq, npad='auto')
            data = raw.get_data()[ch_idxs, :]
            print(f"Data shape after channel selection and resampling to {self.target_freq}: {data.shape}")

            # Process data for training or evaluation
            if '/train/' in filename:
                # Preprocess for training: Keep 5 minutes and segment into 1-minute epochs
                data = self.segment_epochs(data, minutes_to_keep=4, segment_into_epochs=True)
            elif '/eval/' in filename:
                # Preprocess for evaluation: Keep 2 minutes and segment into 1-minute epochs
                data = self.segment_epochs(data, minutes_to_keep=2, segment_into_epochs=True)

            if data is None:
                return None, None

            # Normalize the channels
            data = self.normalize_channels(data)

            # Create labels: 1 for abnormal, 0 for normal (based on filename)
            label = np.repeat(int('abnormal' in filename), data.shape[0])
            label = np_utils.to_categorical(label, 2)
            print(f"Label shape: {label.shape}, Labels: {label}")

            return data, label

        except (IndexError, ValueError):
            print(f"Skipping file {filename} due to missing channels/ lack of data.")
            return None, None
    
    def normalize_channels(self, trials):
        """Normalize each channel in the trials using StandardScaler."""
        for j in range(trials.shape[1]):  # Iterate over channels
            scaler = StandardScaler()
            scaler.fit(trials[:, j, :])
            trials[:, j, :] = scaler.transform(trials[:, j, :])
        return trials
    
    def load_dataset(self):
        """
        Load and preprocess all EEG files, stacking data and labels for training and evaluation sets.

        This function reads all `.edf` files in the specified path and separates them into 
        training and evaluation datasets.

        Returns:
        - eeg_data (dict): Contains 'train_ds' and 'test_ds' datasets for training and evaluation.
        """
        train_data = []
        train_labels = []
        test_data = []
        test_labels = []

        # Read all files in the dataset directory
        all_files = glob(os.path.join(self.filepath, '**/*.edf'), recursive=True)
        print(f"Found {len(all_files)} .edf files in the dataset.")

        # Iterate over all files and preprocess them
        for filename in all_files:
            x, y = self.preprocess_one_file(filename)
            if x is None or y is None:
                continue

            base_filename = os.path.basename(filename)

            if '/train/' in filename:
                print(f"Training file: {base_filename} - Label: {'Abnormal' if 'abnormal' in filename else 'Normal'}")
                train_data.append(x)
                train_labels.append(y)
            elif '/eval/' in filename:
                print(f"Evaluation file: {base_filename} - Label: {'Abnormal' if 'abnormal' in filename else 'Normal'}")
                test_data.append(x)
                test_labels.append(y)

        # Stack all data and labels for train and evaluation sets
        if train_data:
            train_data = np.vstack(train_data)
            train_labels = np.vstack(train_labels)

            # Shuffle training data and labels together
            indices = np.arange(train_data.shape[0])
            np.random.shuffle(indices)
            train_data = train_data[indices]
            train_labels = train_labels[indices]

            print(f"Final training data shape: {train_data.shape}")
            print(f"Final training labels shape: {train_labels.shape}")
        else:
            train_data, train_labels = np.array([]), np.array([])

        if test_data:
            test_data = np.vstack(test_data)
            test_labels = np.vstack(test_labels)
            print(f"Final evaluation data shape: {test_data.shape}")
            print(f"Final evaluation labels shape: {test_labels.shape}")
        else:
            test_data, test_labels = np.array([]), np.array([])

        # Convert the data into TensorFlow datasets
        if train_data.size > 0:
            train_ds = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
            train_ds = train_ds.shuffle(buffer_size=10000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
            print("Train dataset created successfully.")
        else:
            train_ds = None
            print("No training data found.")

        if test_data.size > 0:
            test_ds = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
            test_ds = test_ds.shuffle(len(test_data)).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
            print("Test dataset created successfully.")
        else:
            test_ds = None
            print("No evaluation data found.")

        # Store the datasets in the 'subject' key of the eeg_data dictionary
        self.eeg_data['all']['train_ds'] = train_ds
        self.eeg_data['all']['test_ds'] = test_ds

        return self.eeg_data

class HighGammaLoader:
    def __init__(self):
        """
        Initialize the HighGammaLoader with the target sampling frequency (128Hz) and batch size.
        """
        self.target_fs = 128
        self.batch_size = 16
        self.eeg_data = {}
        
        # Create an instance of the dataset and paradigm classes
        self.dataset = Schirrmeister2017()
        self.paradigm = MotorImagery()
        self.subjects = self.dataset.subject_list  # Load all available subjects

        # List of sensors to keep
        self.sensors_to_keep = ['FC5', 'FC1', 'FC2', 'FC6', 'C3', 'C4', 'CP5', 'CP1', 'CP2', 'CP6', 
                                'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4',
                                'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h', 'FCC5h', 'FCC3h', 'FCC4h', 
                                'FCC6h', 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h', 'CPP5h', 'CPP3h', 
                                'CPP4h', 'CPP6h', 'FFC1h', 'FFC2h', 'FCC1h', 'FCC2h', 'CCP1h', 
                                'CCP2h', 'CPP1h', 'CPP2h']        

    def normalize_channels(self, trials):
        """
        Normalize each channel in the trials using StandardScaler.

        Parameters:
        trials (numpy.ndarray): Input data of shape (n_trials, n_channels, n_timepoints).
                                Each channel of each trial will be normalized independently.

        Returns:
        numpy.ndarray: The normalized trials with the same shape as the input.
        """
        for j in range(trials.shape[1]):  # Iterate over channels
            scaler = StandardScaler()
            # Fit the scaler to the data of the current channel across all trials
            scaler.fit(trials[:, j, :])
            # Transform the data for the current channel
            trials[:, j, :] = scaler.transform(trials[:, j, :])
        return trials
    
    def _downsample(self, X, original_fs):
        """
        Downsample the data to the target frequency (128Hz).
        """
        num_samples = int(X.shape[2] * self.target_fs / original_fs)
        return resample(X, num_samples, axis=2)

    def _prepare_data(self, X, labels, meta_df):
        """
        Filter the channels, downsample the data, split into train/test, and one-hot encode labels.
        """
        # Dynamically fetch raw info from the actual data being processed
        first_subject = self.subjects[0]
        raw = self.dataset._get_single_subject_data(first_subject)['0']['0train']  # Access the first subject for channel information
        # original sample frequency
        original_fs = raw.info['sfreq']
        print(f"Original sampling frequency: {original_fs} Hz")
        # Find indices of sensors to keep
        sensor_indices = [raw.info['ch_names'].index(sensor) for sensor in self.sensors_to_keep if sensor in raw.info['ch_names']]
        
        # Filter data to keep only selected sensors
        X_filtered = X[:, sensor_indices, :]
        print(f"After filtering sensors, data shape: {X_filtered.shape}")
        
        # Downsample the data from the original frequency of 250Hz to 128Hz
        logging.info("Downsampling data to 128Hz")
        X_filtered_downsampled = self._downsample(X_filtered, original_fs)
        print(f"After downsampling, data shape: {X_filtered_downsampled.shape}")
        
        # Check for 'run' column in meta_df before splitting the data
        if 'run' not in meta_df.columns:
            raise ValueError("Expected 'run' column in meta data for splitting into train and test sets.")
        
        # Split the data into training and testing sets
        train_indices = meta_df['run'].str.contains('train')
        test_indices = meta_df['run'].str.contains('test')
        
        X_train, X_test = X_filtered_downsampled[train_indices], X_filtered_downsampled[test_indices]
        y_train, y_test = labels[train_indices], labels[test_indices]

        # Normalize the channels across all trials
        X_train = self.normalize_channels(X_train)
        X_test = self.normalize_channels(X_test)

        # Encode the labels and one-hot encode them
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)

        print(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")
        print(f"Testing data shape: {X_test.shape}, Testing labels shape: {y_test.shape}")
        
        return X_train, X_test, y_train, y_test

    def _create_tf_datasets(self, X_train, y_train, X_test, y_test):
        """
        Create TensorFlow datasets from the train and test data.
        """
        # Create TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

        # Shuffle, batch, and prefetch the training dataset; batch and prefetch the testing dataset
        train_dataset = train_dataset.shuffle(buffer_size=10000).batch(self.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        test_dataset = test_dataset.batch(self.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return train_dataset, test_dataset

    def load_dataset(self):
        """
        Load and process the dataset for all subjects.
        """
        logging.info(f"Loading dataset for {len(self.subjects)} subjects")

        for subject_id in self.subjects:
            logging.info(f"Processing subject {subject_id}")

            # Use the paradigm instance to load the data for the subject
            X, labels, meta = self.paradigm.get_data(dataset=self.dataset, subjects=[subject_id])

            # Print shapes before processing
            print(f"Raw data shape before processing: {X.shape}")
            print(f"Raw labels shape before processing: {labels.shape}")

            # Convert metadata into a DataFrame
            meta_df = pd.DataFrame(meta)

            # Prepare data (filter, downsample, split into train/test, and encode labels)
            X_train, X_test, y_train, y_test = self._prepare_data(X, labels, meta_df)

            # Create TensorFlow datasets
            train_dataset, test_dataset = self._create_tf_datasets(X_train, y_train, X_test, y_test)

            # Store the prepared datasets in the dictionary
            self.eeg_data[subject_id] = {
                'train_ds': train_dataset,
                'test_ds': test_dataset
            }
            logging.info(f"Data for subject {subject_id} loaded successfully")

        logging.info("All datasets loaded and prepared")
        return self.eeg_data

class SleepEDFLoader:
    def __init__(self, filepath, batch_size=16, n_splits=3):
        self.path = filepath
        self.batch_size = batch_size
        self.n_splits = n_splits
        self.eeg_data = {}  # Dictionary to hold datasets
        self.annotation_files = sorted(glob(os.path.join(filepath, '*Hypnogram.edf')))
        self.signal_files = sorted(glob(os.path.join(filepath, '*PSG.edf')))
        self.len_annot = len(self.annotation_files)
        self.len_signal = len(self.signal_files)
        
        logging.basicConfig(level=logging.INFO)
        
        # Define the annotation to label mapping within the class
        self.ann2label = {
            "Sleep stage W": 0,  # Awake
            "Sleep stage 1": 1,  # N1
            "Sleep stage 2": 2,  # N2
            "Sleep stage 3": 3,  # N3
            "Sleep stage 4": 3,  # N3 (merge N3 and N4 as described)
            "Sleep stage R": 4,  # REM
            "Sleep stage ?": 5,  # Unknown
            "Movement time": 5   # Movement
        }
        
        # Define epoch size within the class
        self.EPOCH_SEC_SIZE = 30  # 30-second epochs

    def _read_signals_and_annotations(self, signal_file, annotation_file, select_ch="EEG Fpz-Cz"):
        """Read signals and annotations from EDF files using mne."""
        # Read the signal file using mne
        raw = read_raw_edf(signal_file, preload=True)
        sampling_rate = raw.info['sfreq']
        
        # Extract the selected channel data
        raw_ch = raw.to_data_frame()[select_ch].values
        
        # print(f"Signal data shape (raw EEG): {raw_ch.shape}")
        
        # Read the annotations (hypnogram) file using mne
        annotations = mne.read_annotations(annotation_file)
        
        # print(f"Annotations: {len(annotations)} total")
        
        # Align annotations to raw data
        raw.set_annotations(annotations)
        
        # Extract events from the annotations (start time, duration, description)
        annots = annotations.onset, annotations.duration, annotations.description
        
        return raw_ch, annots, sampling_rate
    
    def _process_annotations(self, annotations, sampling_rate):
        """Convert annotations to integer classes and segment the data."""
        labels = []
        label_idx = []

        onset, duration, description = annotations

        wake_epochs_at_start = 0
        wake_epochs_at_end = 0
        sleep_found = False
        end_found = False

        for i in range(len(description)):
            ann_str = description[i]
            onset_sec = onset[i]
            duration_sec = duration[i]
            label = self.ann2label.get(ann_str, 5)  # Default to 5 if not found

            if label != 5:  # Ignore unknown and movement stages (label == 5)
                if duration_sec % self.EPOCH_SEC_SIZE != 0:
                    raise Exception("Duration not a multiple of epoch size")
                duration_epoch = int(duration_sec / self.EPOCH_SEC_SIZE)

                if label == 0:  # Wake stage W
                    if not sleep_found:
                        wake_epochs_at_start += duration_epoch
                        if wake_epochs_at_start > 60:
                            excess_epochs = wake_epochs_at_start - 60
                            duration_epoch -= excess_epochs
                            wake_epochs_at_start = 60
                        label_epoch = np.ones(duration_epoch, dtype=int) * label

                    elif end_found:
                        wake_epochs_at_end += duration_epoch
                        if wake_epochs_at_end > 60:
                            excess_epochs = wake_epochs_at_end - 60
                            duration_epoch -= excess_epochs
                            wake_epochs_at_end = 60
                        label_epoch = np.ones(duration_epoch, dtype=int) * label

                    else:
                        continue  # Skip wake periods during sleep

                else:
                    sleep_found = True
                    label_epoch = np.ones(duration_epoch, dtype=int) * label

                labels.append(label_epoch)

                idx = int(onset_sec * sampling_rate) + np.arange(duration_epoch * self.EPOCH_SEC_SIZE * sampling_rate, dtype=int)
                label_idx.append(idx)

            # Mark the end after sleep stages
            if sleep_found and label == 0:
                end_found = True

        labels_array = np.hstack(labels)
        label_idx_array = np.hstack(label_idx)

        # print(f"Processed annotations shape (labels): {labels_array.shape}")
        # print(f"Processed label indices shape: {label_idx_array.shape}")

        return labels_array, label_idx_array


    def _segment_data(self, raw_data, labels, label_idx, sampling_rate):
        """Segment data into 30s epochs and assign labels."""
        # Filter the raw data by valid label indices
        raw_data = raw_data[label_idx]

        # print(f"Raw data shape after filtering by label indices: {raw_data.shape}")

        # Verify that we can split into 30-s epochs
        if len(raw_data) % (self.EPOCH_SEC_SIZE * sampling_rate) != 0:
            raise Exception("Data length is not a multiple of epoch size")
        
        n_epochs = int(len(raw_data) // (self.EPOCH_SEC_SIZE * sampling_rate)) 
        
        # Split the raw data into 30-s epochs
        X_data = np.asarray(np.split(raw_data, n_epochs)).astype(np.float32)

        # print(f"X_data shape after segmentation (epochs): {X_data.shape}")

        # Add channel dimension (assuming 1 channel, adjust if necessary)
        X_data = np.expand_dims(X_data, axis=1)  # Shape becomes (n_epochs, 1, trial_length)

        # Trim or pad the labels to match the number of epochs
        if len(labels) > n_epochs:
            Y_data = labels[:n_epochs]  # Trim labels if there are more labels than epochs
        else:
            Y_data = np.pad(labels, (0, n_epochs - len(labels)), 'constant', constant_values=0)  # Pad with zeros if fewer

        # print(f"X_data shape after adding channel dimension: {X_data.shape}")
        # print(f"Y_data shape after segmentation (epochs): {Y_data.shape}")
        
        return X_data, Y_data
    
    def _remove_unwanted_classes(self, X_data, Y_data, unwanted_classes=[5]):
        """Remove unwanted classes (like 'awake' and 'movement')."""
        for unwanted_class in unwanted_classes:
            mask = Y_data != unwanted_class
            X_data = X_data[mask]
            Y_data = Y_data[mask]
        
        # print(f"X_data shape after removing unwanted classes: {X_data.shape}")
        # print(f"Y_data shape after removing unwanted classes: {Y_data.shape}")
        
        return X_data, Y_data
       
    def normalize_channels(self, trials):
        """Normalize each channel in the trials using StandardScaler."""
        for j in range(trials.shape[1]):  # Iterate over channels
            scaler = StandardScaler()
            scaler.fit(trials[:, j, :])
            trials[:, j, :] = scaler.transform(trials[:, j, :])
        return trials
    
    def load_dataset(self):
        """Load and process the entire dataset, preparing for k-fold cross-validation."""
        X_data_total = []
        Y_data_total = []
        subject_ids = []

        for i in range(self.len_annot):
            logging.info(f"\nProcessing file set {i+1}/{self.len_annot}")
            signal_file = self.signal_files[i]
            annotation_file = self.annotation_files[i]
            logging.info(f"Signal file: {signal_file}")
            logging.info(f"Annotation file: {annotation_file}")

            # Read signals and annotations
            raw_data, annotations, sampling_rate = self._read_signals_and_annotations(signal_file, annotation_file)
            labels, label_idx = self._process_annotations(annotations, sampling_rate)
            X_data, Y_data = self._segment_data(raw_data, labels, label_idx, sampling_rate)
            logging.info(f"Signals shape: {raw_data.shape}")
            logging.info(f"X_data shape: {X_data.shape}, Y_data shape: {Y_data.shape}")
            
            # Append data and labels
            X_data_total.append(X_data)
            Y_data_total.append(Y_data)
            subject_ids.extend([i] * X_data.shape[0])  # Assign subject ID based on file index

        # Concatenate all subjects' data and labels
        X_data_total = np.concatenate(X_data_total, axis=0)
        Y_data_total = np.concatenate(Y_data_total, axis=0)
        Y_data_total = np_utils.to_categorical(Y_data_total)
        logging.info(f"Data shape after processing all files: {X_data_total.shape}, {Y_data_total.shape}")

        # Normalize each channel independently
        X_data_total = self.normalize_channels(X_data_total)
        
        self.eeg_data['all'] = {'trials': X_data_total, 'labels': Y_data_total}

        # Prepare k-fold cross-validation splits by subject
        gkf = GroupKFold(n_splits=self.n_splits)
        for fold, (train_indices, test_indices) in enumerate(gkf.split(X_data_total, groups=subject_ids), start=1):
            train_subjects = set(np.array(subject_ids)[train_indices])
            test_subjects = set(np.array(subject_ids)[test_indices])

            self.eeg_data[fold] = {'train_indices': train_indices, 'test_indices': test_indices}

            # Print debugging information for each fold
            print(f"\nFold {fold}")
            print(f"Training subjects: {sorted(train_subjects)}")
            print(f"Testing subjects: {sorted(test_subjects)}")
            print(f"Train data shape: {len(train_indices)}, Test data shape: {len(test_indices)}")

        logging.info("Dataset loaded and k-fold indices prepared successfully.")
        return self.eeg_data

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

class SimulatedEEGLoader:
    def __init__(self, filepath, n_channels=14, trial_length=128*2, test_size=0.2, batch_size=16):
        self.file_path = filepath
        self.n_channels = n_channels
        self.trial_length = trial_length
        self.test_size = test_size
        self.batch_size = batch_size
        self.eeg_data = {'all': None}
    
    def load_data(self):
        """Loads CSV data from the file and reshapes it into trials."""
        df = pd.read_csv(self.file_path)
        data = df.iloc[:, :self.n_channels].values
        labels = df['Label'].values
        
        # Reshape the data into (trials, n_channels, trial_length)
        n_trials = data.shape[0] // self.trial_length
        reshaped_data = data.reshape(n_trials, self.trial_length, self.n_channels)
        reshaped_data = np.transpose(reshaped_data, (0, 2, 1))  # Convert to (trials, n_channels, trial_length)
        return reshaped_data, labels[::self.trial_length]
    
    def create_dataset(self, data, labels):
        """Creates a TensorFlow dataset from the data and labels."""
        labels = tf.keras.utils.to_categorical(labels, num_classes=len(np.unique(labels)))
        dataset = tf.data.Dataset.from_tensor_slices((data, labels))
        dataset = dataset.shuffle(buffer_size=100).batch(self.batch_size)
        return dataset
    
    def split_data(self, data, labels):
        """Splits the data into training and testing datasets."""
        train_data, test_data, train_labels, test_labels = train_test_split(
            data, labels, test_size=self.test_size, stratify=labels)
        print(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")
        print(f"Train labels shape: {train_labels.shape}, Test labels shape: {test_labels.shape}")
                
        self.eeg_data['all'] = {
            'train_ds': self.create_dataset(train_data, train_labels),
            'test_ds': self.create_dataset(test_data, test_labels)
        }
    
    def load_dataset(self):
        """Main function to load, split, and create train/test datasets."""
        data, labels = self.load_data()
        self.split_data(data, labels)
        print(f"Preprocessing complete: Dataset shape: {data.shape}")
        return self.eeg_data

class SimulatedCVLoader:
    def __init__(self, filepath, n_splits=5):
        self.data_file_dir = filepath
        self.n_splits = n_splits
        self.eeg_data = {}

    def load_dataset(self):
        subject_files = [f for f in os.listdir(self.data_file_dir) if f.endswith('.csv')]
        
        all_data = []        # Store concatenated trial data for all subjects
        all_labels = []      # Store concatenated labels for all subjects
        subject_ids = []     # Track subject ID for each trial

        # Load data for each subject from CSV files
        for subject_file in subject_files:
            subject_id = int(subject_file.split('_')[1].split('.')[0])  # Extract subject ID from filename
            subject_path = os.path.join(self.data_file_dir, subject_file)
            
            # Read CSV file
            df = pd.read_csv(subject_path)
            n_channels = 14  # Set according to generated data parameters
            n_time_points = 256  # Set according to generated data parameters
            
            # Separate channel data, labels, and subjects
            subject_data = df.iloc[:, :n_channels].values  # Channel data
            labels = df['Label'].values[::n_time_points]   # Class labels for each trial
            trials = subject_data.reshape(-1, n_time_points, n_channels).transpose(0, 2, 1)  # Reshape to (n_trials, n_channels, n_time_points)

            # Append trials and labels to overall dataset lists
            all_data.append(trials)
            all_labels.append(labels)
            subject_ids.extend([subject_id] * len(trials))  # Extend subject IDs for each trial

        # Concatenate all subject trials and labels
        all_trials = np.concatenate(all_data, axis=0)      # Shape: (total_trials, n_channels, n_time_points)
        all_labels = np.concatenate(all_labels, axis=0)    # Shape: (total_trials,)

        # one-hot encoded for labels
        all_labels = np_utils.to_categorical(all_labels)

        # Normalize channels (StandardScaler by channel)
        all_trials = self.normalize_channels(all_trials)

        # Store the entire dataset in eeg_data['all']
        self.eeg_data['all'] = {'trials': all_trials, 'labels': all_labels}

        # Perform GroupKFold cross-validation
        gkf = GroupKFold(n_splits=self.n_splits)
        for fold, (train_indices, test_indices) in enumerate(gkf.split(all_trials, groups=subject_ids)):
            self.eeg_data[fold + 1] = {'train_indices': train_indices, 'test_indices': test_indices}

            # Debug output for verification
            train_subjects = set(subject_ids[i] for i in train_indices)
            test_subjects = set(subject_ids[i] for i in test_indices)
            print(f"\nFold {fold + 1}")
            print(f"Training subjects: {sorted(train_subjects)}")
            print(f"Testing subjects: {sorted(test_subjects)}")
            print(f"Number of training samples: {len(train_indices)}")
            print(f"Number of testing samples: {len(test_indices)}")

        print("Data loaded successfully with subject-level fold indices saved in eeg_data.")
        return self.eeg_data

    def normalize_channels(self, trials):
        # Standardize each channel independently across all trials
        for ch in range(trials.shape[1]):
            scaler = StandardScaler()
            trials[:, ch, :] = scaler.fit_transform(trials[:, ch, :].T).T
        return trials

class SEEDIVLoader:
    def __init__(self, filepath, num_participants=15, window_size=8, stride=4):
        """
        Initializes the SEEDIVLoader class with paths and parameters.
        """
        self.filepath = filepath
        self.num_participants = num_participants
        self.sample_freq = 200
        self.target_sample_freq = 128
        self.window_size = window_size * self.target_sample_freq
        self.stride = stride * self.target_sample_freq
        self.batch_size = 16
        self.eeg_data = {}

        # Setting up the logger
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("SEEDIVLoader")

    def preprocess_eeg(self, raw_eeg):
        """
        Preprocess the EEG data using CAR, filtering, and resampling to 128 Hz.
        """
        average_reference = np.mean(raw_eeg, axis=0)
        car_eeg = raw_eeg - average_reference
        b, a = signal.butter(4, [0.15, 40], btype='bandpass', fs=self.sample_freq)
        filtered_eeg = signal.filtfilt(b, a, car_eeg, axis=1)
        num_samples = int(filtered_eeg.shape[1] * self.target_sample_freq / self.sample_freq)
        return signal.resample(filtered_eeg, num_samples, axis=1)

    def sliding_windows(self, eeg_data):
        """
        Generate sliding window segments from EEG data.
        """
        windows = []
        num_channels, total_length = eeg_data.shape
        num_windows = (total_length - self.window_size) // self.stride + 1

        if num_windows <= 0:
            self.logger.warning(f"Not enough data to generate windows. Total length: {total_length}, window size: {self.window_size}")
            return np.array([])

        for start in range(0, total_length - self.window_size + 1, self.stride):
            end = start + self.window_size
            segment = eeg_data[:, start:end]
            windows.append(segment)

        return np.array(windows)

    def process_file(self, file_name, labels):
        """
        Processes a single .mat file, preprocesses the EEG data, and generates sliding windows.
        """
        keys_to_ignore = ['__header__', '__version__', '__globals__']
        mat = io.loadmat(file_name)
        trial_keys = [key for key in mat.keys() if key not in keys_to_ignore]
        prefix = trial_keys[0].split('_eeg')[0]
        
        windows_list, label_list = [], []
        labelcount = 0

        for trial in range(1, 25):
            trial_name = f'{prefix}_eeg{trial}'
            if trial_name in mat:
                trial_data = mat[trial_name]
                preprocessed_eeg = self.preprocess_eeg(trial_data)
                windows = self.sliding_windows(preprocessed_eeg)

                if windows.size > 0:
                    np.random.shuffle(windows)
                    windows_list.append(windows)
                    label_list += [labels[labelcount]] * windows.shape[0]
                labelcount += 1

        if windows_list:
            windows_array = np.concatenate(windows_list, axis=0)
            label_array = np.array(label_list)
            # Convert labels to one-hot encoded format
            label_array = self.get_labels(label_array)

            return windows_array, label_array
        else:
            print(f"No valid data for this subject.")
            return None, None

    def get_labels(self, label_array):
        return tf.keras.utils.to_categorical(label_array)

    def get_session_labels(self):
        session1_label = np.array([1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3])
        session2_label = np.array([2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1])
        session3_label = np.array([1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0])
        return session1_label, session2_label, session3_label

    def get_session_number(self, file_path):
        """
        Extracts session number from the file path (based on folder structure).
        """
        session_number = int(file_path.split('/')[-2])  # Assumes session number is in the directory name
        return session_number
    
    def get_participant_files(self, participant_num):
        participant_prefix = f"{participant_num}_"
        participant_files = []
        for root, _, files in os.walk(self.filepath):
            for file_name in files:
                if file_name.startswith(participant_prefix):
                    participant_files.append(os.path.join(root, file_name))
        return participant_files

    def process_participant(self, participant_num):
        """
        Processes the data for a single participant, creating an 80-20 train-test split on a trial basis.
        """
        session1_label, session2_label, session3_label = self.get_session_labels()
        participant_files = self.get_participant_files(participant_num)
        sorted_files = sorted(participant_files, key=self.get_session_number)

        trial_windows, trial_labels = [], []
        for session_num, labels in zip(range(1, 4), [session1_label, session2_label, session3_label]):
            file_path = sorted_files[session_num - 1]
            windows, labels = self.process_file(file_path, labels)
            num_trials = len(labels)
            trial_size = windows.shape[0] // num_trials

            for i in range(num_trials):
                trial_windows.append(windows[i * trial_size:(i + 1) * trial_size])
                trial_labels.append(labels[i * trial_size:(i + 1) * trial_size])

        trials_by_class = defaultdict(list)
        for i, trial in enumerate(trial_labels):
            majority_label = np.argmax(np.bincount(trial.argmax(axis=1)))
            trials_by_class[majority_label].append(i)

        train_indices, test_indices = [], []
        for class_trials in trials_by_class.values():
            train_class, test_class = train_test_split(class_trials, test_size=0.2, random_state=None)
            train_indices.extend(train_class)
            test_indices.extend(test_class)

        X_train = np.concatenate([trial_windows[i] for i in train_indices], axis=0)
        X_test = np.concatenate([trial_windows[i] for i in test_indices], axis=0)
        y_train = np.concatenate([trial_labels[i] for i in train_indices], axis=0)
        y_test = np.concatenate([trial_labels[i] for i in test_indices], axis=0)

        print(f"Participant {participant_num}: Train data shape: {X_train.shape}, Train label shape: {y_train.shape}, Test data shape: {X_test.shape}, Test label shape: {y_test.shape}")

        X_train = self.normalize_channels(X_train)
        X_test = self.normalize_channels(X_test)

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

        train_dataset = train_dataset.shuffle(buffer_size=10000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        test_dataset = test_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        return train_dataset, test_dataset

    def normalize_channels(self, trials):
        for j in range(trials.shape[1]):
            scaler = StandardScaler()
            scaler.fit(trials[:, j, :])
            trials[:, j, :] = scaler.transform(trials[:, j, :])
        return trials

    def load_dataset(self):
        print("Starting dataset loading and preprocessing...")
        for participant_num in range(1, self.num_participants + 1):
            print(f"Processing Participant {participant_num}")
            train_dataset, test_dataset = self.process_participant(participant_num)
            self.eeg_data[participant_num] = {
                'train_ds': train_dataset,
                'test_ds': test_dataset
            }
            print(f"Completed Participant {participant_num}")
        print("All participants processed.")
        return self.eeg_data

