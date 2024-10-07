import os
import re
from glob import glob
import time
from tqdm import tqdm
import logging
import numpy as np
import pandas as pd
import mne
import pickle
import pyedflib
from scipy import io, signal
import tensorflow as tf
import tensorflow.keras.utils as np_utils # type: ignore
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from scipy.signal import resample, butter, filtfilt, lfilter, iirnotch 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class BCICIV2aLoader:
    def __init__(self, filepath, stimcodes):
        self.filepath = filepath
        self.stimcodes = stimcodes
        self.sample_freq = None
        self.logger = self._get_logger()
        self.batch_size = 16

    def _get_logger(self):
        # Set up logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        return logger

    def load_data(self, filename):
        gdf_name = filename.split(".")[0]
        raw_data = mne.io.read_raw_gdf(os.path.join(self.filepath, filename), preload=True, eog=['EOG-left', 'EOG-central', 'EOG-right'])
        raw_data.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])
        raw_data.filter(l_freq=4, h_freq=40)
        raw_data.resample(128)
        self.sample_freq = int(raw_data.info.get('sfreq'))
        before_trial = int(0.5 * self.sample_freq)
        data = raw_data.get_data() * 1e6  # Convert to microvolts

        self.logger.info(f"Loading data from {filename}...")
        
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
        print("t loaded successfully")
        return np.array(trials), labels

    def _process_evaluation_data(self, raw_data, data, gdf_name, before_trial):
        trials = []
        for annotation in raw_data.annotations:
            if annotation['description'] == "783":
                onset_idx = int(annotation['onset'] * self.sample_freq)
                trial = data[:, onset_idx - before_trial:onset_idx + int(4 * self.sample_freq)]
                trials.append(trial)
        labels = io.loadmat(os.path.join(self.filepath, "true_labels", gdf_name + ".mat"))["classlabel"][:, 0] - 1
        labels = np_utils.to_categorical(labels, num_classes=4)
        print("e loaded successfully")
        return np.array(trials), labels

    def _get_labels(self, labels):
        unique_labels = np.sort(np.unique(labels))
        label_map = {old: new for new, old in enumerate(unique_labels)}
        mapped_labels = np.vectorize(label_map.get)(labels)
        return np.eye(len(unique_labels))[mapped_labels.astype(int)]

    def extract_features(self, trials):
        features = np.array([[np.mean(channel), np.std(channel), np.min(channel), np.max(channel)] 
                             for trial in trials for channel in trial])
        self.logger.info("Features extracted successfully.")
        return features

    # Create the dataset
    def create_datasets(self, trials, labels, win_length, stride):
        print("Creating dataset....")
        crop_starts = []

        for trial_idx, trial in enumerate(trials):
            trial_len = trial.shape[1]
            for i in range(0, trial_len - win_length + 1, stride):
                crop_starts.append((trial_idx, i))

        crop_starts = np.array(crop_starts)
        np.random.shuffle(crop_starts)

        trial_indices = crop_starts[:, 0]
        start_indices = crop_starts[:, 1]

        trial_indices = tf.convert_to_tensor(trial_indices, dtype=tf.int32)
        start_indices = tf.convert_to_tensor(start_indices, dtype=tf.int32)
        trials_tensor = tf.convert_to_tensor(trials, dtype=tf.float32)
        labels_tensor = tf.convert_to_tensor(labels, dtype=tf.float32)

        # Print the shape of trials and labels tensors
        print(f"Trials shape: {trials_tensor.shape}")
        print(f"Labels shape: {labels_tensor.shape}")

        map_func = self._get_window(trials_tensor, labels_tensor,win_length)
        dataset = tf.data.Dataset.from_tensor_slices((trial_indices, start_indices))
        dataset = dataset.map(lambda trial_idx, start: map_func(trial_idx, start))
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        # Calculate the number of batches
        total_windows = len(crop_starts)  # Total number of windows created
        num_batches = (total_windows + self.batch_size - 1) // self.batch_size  # Total batches
        print(f"Total windows: {total_windows}")
        print(f"Batch size: {self.batch_size}")
        print(f"Number of batches: {num_batches}")
        

        self.logger.info("Tensor dataset created.")
        print("Dataset created successfully.")

        return dataset

    def _get_window(self, data, labels, win_length):
        def map_func(trial_idx, start):
            trial = tf.gather(data, trial_idx)
            window = trial[:, start:start + win_length]
            win_label = labels[trial_idx]
            return window, win_label
        return map_func

    def load_dataset(self):
        """
        Load and preprocess dataset from files for all subjects.
        
        Returns:
            dict: Dictionary containing training and testing datasets and labels for each subject.
        """
        filenames = [name for name in os.listdir(self.filepath) if name.endswith(".gdf")]
        eeg_data = {}

        for filename in filenames:
            self.logger.info(f"Loading data from {filename}...")
            gdf_name = filename.split(".")[0]
            subject = gdf_name[1:3]  # Extracts subject number from filename
            trials, labels = self.load_data(filename)
            print(f"trial shape: {trials.shape}, label shape: {labels.shape}")
            win_length = 2 * self.sample_freq  # Window length for crops (2 seconds)
            stride = 1 * self.sample_freq  # Define stride length here
            dataset = self.create_datasets(trials, labels, win_length, stride)

            if subject not in eeg_data:
                eeg_data[subject] = {}
            if "T" in gdf_name:
                eeg_data[subject]["train_ds"] = dataset
            elif "E" in gdf_name:
                eeg_data[subject]["test_ds"] = dataset

        return eeg_data

class BCICIV2bLoader:
    def __init__(self, data_file_dir):
        self.data_file_dir = data_file_dir
        self.sample_freq = None
        self.data = {}
        self.logger = self._get_logger()
        self.batch_size = 16
    
    def _get_logger(self):
        # Set up logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        return logger    

    def load_data(self, filename, gdf_name):
        filepath = self.data_file_dir
        raw_data = mne.io.read_raw_gdf(filepath + filename, preload=True,
                                       eog=['EOG:ch01', 'EOG:ch02', 'EOG:ch03'])
        raw_data.drop_channels(['EOG:ch01', 'EOG:ch02', 'EOG:ch03'])
        raw_data.filter(l_freq=4, h_freq=40)
        raw_data.resample(128)
        self.sample_freq = int(raw_data.info.get('sfreq'))
        before_trial = int(0.5 * self.sample_freq)
        data = raw_data.get_data() * 1e6  # Convert to microvolts

        self.logger.info(f"Loading data from {filename}...")

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
            trials = np.array(trials)
            labels = np_utils.to_categorical(labels, num_classes=2)

        return trials, labels

    def get_labels(self, labels):
        labels_unique = np.sort(np.unique(labels))
        for new_label, old_label in enumerate(labels_unique):
            labels[labels == old_label] = new_label

        labels = labels.astype(int)
        labels = np.eye(len(labels_unique))[labels]
        return labels

    def get_window(self, data, labels, win_length):
        def map_func(trial_idx, start):
            trial = tf.gather(data, trial_idx)
            window = trial[:, start:start + win_length]
            win_label = labels[trial_idx]
            return window, win_label
        return map_func

    def create_datasets(self, trials, labels, win_length, stride):
        self.logger.info("Creating dataset....")

        crop_starts = []

        for trial_idx, trial in enumerate(trials):
            trial_len = trial.shape[1]
            for i in range(0, trial_len - win_length + 1, stride):
                crop_starts.append((trial_idx, i))

        crop_starts = np.array(crop_starts)
        np.random.shuffle(crop_starts)

        trial_indices = crop_starts[:, 0]
        start_indices = crop_starts[:, 1]

        trial_indices = tf.convert_to_tensor(trial_indices, dtype=tf.int32)
        start_indices = tf.convert_to_tensor(start_indices, dtype=tf.int32)
        trials_tensor = tf.convert_to_tensor(trials, dtype=tf.float32)
        labels_tensor = tf.convert_to_tensor(labels, dtype=tf.float32)

        map_func = self.get_window(trials_tensor, labels_tensor, win_length)
        dataset = tf.data.Dataset.from_tensor_slices((trial_indices, start_indices))
        dataset = dataset.map(lambda trial_idx, start: map_func(trial_idx, start))
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        # Calculate the number of batches
        total_windows = len(crop_starts)  # Total number of windows created
        num_batches = (total_windows + self.batch_size - 1) // self.batch_size  # Total batches
        print(f"Total windows: {total_windows}")
        print(f"Batch size: {self.batch_size}")
        print(f"Number of batches: {num_batches}")

        self.logger.info("Tensor dataset created.")

        return dataset

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

            self.data[subject]['train_ds'] = self.create_datasets(all_train_trials, all_train_labels, win_length, stride)
            self.data[subject]['test_ds'] = self.create_datasets(all_test_trials, all_test_labels, win_length, stride)

        return self.data

class DREAMERLoader:
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
    
    def load_dreamer_data_by_subject(self):
        eeg_dataset = {}
        mat_data = io.loadmat(self.mat_path, verify_compressed_data_integrity=False)
        subject_len = len(mat_data['DREAMER'][0, 0]['Data'][0])  # 23 subjects
        
        overlap = int(self.chunk_size * self.overlap_rate)

        for subject in range(subject_len):
            eeg_data_list = []
            labels_list = []
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

                while end_at <= trial_samples.shape[1]:
                    clip_sample = trial_samples[:, start_at:end_at]

                    # Assuming binary classification based on label_type
                    if self.label_type == 'valence':
                        valence = mat_data['DREAMER'][0, 0]['Data'][0, subject]['ScoreValence'][0, 0][trial_id, 0]
                        label = 1 if valence >= 3.0 else 0
                    elif self.label_type == 'arousal':
                        arousal = mat_data['DREAMER'][0, 0]['Data'][0, subject]['ScoreArousal'][0, 0][trial_id, 0]
                        label = 1 if arousal >= 3.0 else 0

                    eeg_data_list.append(clip_sample)
                    labels_list.append(label)

                    start_at += step
                    end_at = start_at + dynamic_chunk_size

            # Store the data and labels in the dictionary for the current subject
            labels_list = np_utils.to_categorical(np.array(labels_list))

            eeg_dataset[subject] = {
                'eeg_data': np.array(eeg_data_list),
                'labels': np.array(labels_list)
            }
        return eeg_dataset
    
    def prepare_data_for_training(self, eeg_dataset):
        for subject in eeg_dataset:
            print("Preparing data for subject", subject)
            eeg_data = eeg_dataset[subject]['eeg_data']
            labels = eeg_dataset[subject]['labels']

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(eeg_data, labels, test_size=self.test_size, random_state=42, stratify=labels)

            print(f"Subject {subject} - Shape of X_train: {X_train.shape}, y_train: {y_train.shape}")
            print(f"Subject {subject} - Shape of X_test: {X_test.shape}, y_test: {y_test.shape}")

            # Convert to TensorFlow Datasets
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
            train_dataset = train_dataset.shuffle(buffer_size=10000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
            test_dataset = test_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

            # Store the datasets in the dictionary for the current subject
            eeg_dataset[subject]['train_ds'] = train_dataset
            eeg_dataset[subject]['test_ds'] = test_dataset

        return eeg_dataset

    def load_dataset(self):
        # Call functions to load the DREAMER dataset
        eeg_dataset = self.load_dreamer_data_by_subject()
        eeg_dataset = self.prepare_data_for_training(eeg_dataset)
        return eeg_dataset

class DEAPLoader:
    def __init__(self, filepath, label_type, win_length=8, stride=4, test_size=0.2):
        self.data_path = filepath
        self.sample_freq = 128
        self.win_length = win_length * self.sample_freq  # Convert window length to samples
        self.stride = stride * self.sample_freq  # Convert stride to samples
        self.test_size = test_size
        self.label_type = label_type
        self.batch_size = 16
        self.eeg_data = {}

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
        
        labels = np.where(labels <= 5, 0, labels)
        labels = np.where(labels > 5, 1, labels)
        labels = np_utils.to_categorical(labels)
        return labels

    def get_window(self, data, labels):
        """
        Defines a mapping function to generate windows of data and corresponding labels.
        """
        def map_func(trial_idx, start):
            trial = tf.gather(data, trial_idx)
            window = trial[:, start:start + self.win_length]
            win_label = labels[trial_idx]
            return window, win_label
        return map_func

    def create_datasets(self, trials, labels):
        """
        Creates datasets with sliding windows (crops) for train/test data.
        """
        crop_starts = []

        for trial_idx, trial in enumerate(trials):
            trial_len = trial.shape[1]
            for i in range(0, trial_len - self.win_length + 1, self.stride):
                crop_starts.append((trial_idx, i))

        crop_starts = np.array(crop_starts)
        np.random.shuffle(crop_starts)

        trial_indices = crop_starts[:, 0]
        start_indices = crop_starts[:, 1]

        trial_indices = tf.convert_to_tensor(trial_indices, dtype=tf.int32)
        start_indices = tf.convert_to_tensor(start_indices, dtype=tf.int32)
        trials_tensor = tf.convert_to_tensor(trials, dtype=tf.float32)
        labels_tensor = tf.convert_to_tensor(labels, dtype=tf.float32)

        map_func = self.get_window(trials_tensor, labels_tensor)
        dataset = tf.data.Dataset.from_tensor_slices((trial_indices, start_indices))
        dataset = dataset.map(lambda trial_idx, start: map_func(trial_idx, start))
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

    def load_dataset(self):
        """
        Main function to load and process the data for a subject, and split it into train/test datasets.
        """
        for subject_id in range(1, 33):
            data, labels = self.load_data_per_subject(subject_id)

            # Skip this subject if data could not be loaded
            if data is None or labels is None:
                continue  # Skip to the next subject

            labels = self.get_labels(labels)
            print(f"subject {subject_id} - data shape: {data.shape}, label shape: {labels.shape}")

            # Split the data and labels into train and test sets
            train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=self.test_size, random_state=42)

            # Create the train and test datasets
            train_dataset = self.create_datasets(train_data, train_labels)
            test_dataset = self.create_datasets(test_data, test_labels)

            # Store the datasets in a dictionary for the subject
            self.eeg_data[subject_id] = {
                'train_ds': train_dataset,
                'test_ds': test_dataset
            }
        return self.eeg_data    

class PhysionetMILoader:
    def __init__(self, data_file_dir, trial_length=4):
        self.data_file_dir = data_file_dir
        self.original_freq = None  # Original sample frequency (160 Hz)
        self.target_freq = 128  # Target sample frequency (128 Hz)
        self.trial_length = trial_length  # Trial length in seconds
        self.eeg_data = {}  # Store train/test datasets for each subject
        self.batch_size = 16

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

        print(f"Extracted {trials.shape} trials and {labels.shape} labels")
        return trials, labels

    def create_tf_datasets(self, trials, labels, test_size=0.2):
        """
        Creates train and test datasets using TensorFlow.
        """
        X_train, X_test, y_train, y_test = train_test_split(trials, labels, test_size=test_size, stratify=labels)
        
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        
        # Batch and shuffle the dataset
        train_dataset = train_dataset.shuffle(buffer_size=1000).batch(32).prefetch(tf.data.AUTOTUNE)
        test_dataset = test_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        return train_dataset, test_dataset

    def load_dataset(self):
        """
        Main function to load, process, and split data into train/test datasets.
        """
        subject_folders = [f for f in os.listdir(self.data_file_dir) if os.path.isdir(os.path.join(self.data_file_dir, f))]
        print(f"Found {len(subject_folders)} subject folders")

        for subject_folder in subject_folders:
            print(f"Loading data for subject {subject_folder}...")
            file_paths = self.extract_file_paths(subject_folder)
            signals, annotations = self.load_all_data(file_paths)
            if signals is None or annotations is None:
                print(f"Skipping {subject_folder} due to missing or incomplete data.")
                continue
            trials, labels = self.extract_trials_and_labels(signals, annotations)
            train_dataset, test_dataset = self.create_tf_datasets(trials, labels)

            # Assuming 'S001' is the subject (can be extended for multiple subjects)
            self.eeg_data[subject_folder] = {
                'train_ds': train_dataset,
                'test_ds': test_dataset
            }

        print("Data loaded successfully.")
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

                    windows_list.append(windows)
                    label_list += [self.labels[labelcount]] * windows.shape[0]
                    labelcount += 1

        if windows_list:
            windows_array = np.concatenate(windows_list, axis=0)
            label_array = np.array(label_list)
            print(f"Concatenated windows shape: {windows_array.shape}, labels shape: {label_array.shape}")

            # Shuffle the windows and labels together
            print("Shuffling the data...")
            indices = np.arange(windows_array.shape[0])
            np.random.shuffle(indices)
            windows_array = windows_array[indices]
            label_array = label_array[indices]

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

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

        # Shuffle and batch the datasets
        train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
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
    def __init__(self, filepath, window_size=512, overlap=128):
        self.folder_path = filepath  # Path containing EEG files
        self.channels = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]  # List of channel names
        self.sfreq = 128  # Sampling frequency
        self.window_size = window_size  # Window size in samples
        self.overlap = overlap  # Overlap in samples
        self.batch_size = 16  # Batch size for the datasets
        self.eeg_data = {}  # Dictionary to store train and test datasets for each subject

        # Get all file paths
        self.all_files_path = glob(os.path.join(filepath, '*.txt'))
        self.file_names = [x for x in os.listdir(filepath) if os.path.isfile(os.path.join(filepath, x)) and x.endswith(".txt") and x.startswith("sub")]
        logging.info(f"Found {len(self.all_files_path)} EEG files.")

    def butter_bandpass_filter(self, data, lowcut, highcut, order=4):
        nyquist = 0.5 * self.sfreq
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)

    def notch_filter(self, data, freq, quality_factor=30):
        nyquist = 0.5 * self.sfreq
        notch_freq = freq / nyquist
        b, a = butter(2, [notch_freq - notch_freq / quality_factor, notch_freq + notch_freq / quality_factor], btype='bandstop')
        return filtfilt(b, a, data)

    def create_windows(self, data, step):
        num_windows = (data.shape[0] - self.window_size) // step + 1
        return np.array([data[i * step:i * step + self.window_size] for i in range(num_windows)])

    def process_eeg_data(self, file_path):
        logging.info(f"Processing file: {file_path}")
        data = pd.read_csv(file_path, sep='\s+', header=None)
        data.columns = self.channels

        # Average Referencing
        data_avg_ref = data.sub(data.mean(axis=1), axis=0)

        # Bandpass Filter (1-40 Hz)
        for channel in self.channels:
            data_avg_ref[channel] = self.butter_bandpass_filter(data_avg_ref[channel], 1, 40)

        # Notch Filter to remove line noise at 50 Hz
        for channel in self.channels:
            data_avg_ref[channel] = self.notch_filter(data_avg_ref[channel], 50)

        # Windowing the Data
        step = self.window_size - self.overlap
        windowed_data = {ch: self.create_windows(data_avg_ref[ch], step) for ch in self.channels}

        # Convert to numpy array of shape (num_windows, num_channels, window_size)
        num_windows = windowed_data[self.channels[0]].shape[0]
        windowed_eeg = np.array([np.array([windowed_data[ch][i] for ch in self.channels]) for i in range(num_windows)])

        return windowed_eeg

    def load_subject_data(self, subject_id):
        """
        Load data for a specific subject, both hi and lo tasks.
        """
        subject_files = [file for file in self.all_files_path if f"sub{subject_id:02d}_" in file]
        logging.info(f"Found {len(subject_files)} files for subject {subject_id}")

        hi_files = [file for file in subject_files if '_hi' in file]
        lo_files = [file for file in subject_files if '_lo' in file]

        hi_epochs = [self.process_eeg_data(file) for file in hi_files]
        lo_epochs = [self.process_eeg_data(file) for file in lo_files]

        if not hi_epochs or not lo_epochs:
            logging.warning(f"Missing data for subject {subject_id}. Skipping.")
            return None, None

        # Concatenate data for high task and low task
        hi_data = np.vstack(hi_epochs)
        lo_data = np.vstack(lo_epochs)

        # Create labels for each type (0 for low task, 1 for high task)
        hi_labels = np.ones(hi_data.shape[0], dtype=int)
        lo_labels = np.zeros(lo_data.shape[0], dtype=int)

        # Combine data and labels
        data_array = np.concatenate((hi_data, lo_data), axis=0)
        label_array = np.concatenate((hi_labels, lo_labels), axis=0)

        # Shuffle the data and labels
        indices = np.arange(data_array.shape[0])
        np.random.shuffle(indices)
        data_array = data_array[indices]
        label_array = label_array[indices]

        # Convert labels to one-hot encoded format
        label_array = np_utils.to_categorical(label_array, num_classes=2)

        return data_array, label_array

    def create_tf_dataset(self, data_array, label_array):
        """
        Convert NumPy arrays to TensorFlow Dataset.
        """
        dataset = tf.data.Dataset.from_tensor_slices((data_array, label_array))
        dataset = dataset.shuffle(buffer_size=len(data_array), reshuffle_each_iteration=True).batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def load_dataset(self):
        """
        Load and process the dataset, creating train and test datasets for each subject.
        """
        # Get unique subject IDs from the file names
        unique_subject_ids = sorted(set([int(file.split('_')[0][3:]) for file in self.file_names]))

        for subject_id in unique_subject_ids:
            logging.info(f"Loading data for subject {subject_id}...")

            # Load the subject data (hi and lo task data)
            data_array, label_array = self.load_subject_data(subject_id)

            if data_array is None or label_array is None:
                continue

            # Split the subject's data into training and testing sets
            train_data, test_data, train_labels, test_labels = train_test_split(
                data_array, label_array, test_size=0.2, random_state=42
            )

            # Create TensorFlow datasets for training and testing
            train_dataset = self.create_tf_dataset(train_data, train_labels)
            test_dataset = self.create_tf_dataset(test_data, test_labels)

            # Store the train and test datasets for the subject
            self.eeg_data[f"subject_{subject_id:02d}"] = {
                'train_ds': train_dataset,
                'test_ds': test_dataset
            }

        logging.info("All subjects have been processed.")
        return self.eeg_data

class CHBMITLoader:
    def __init__(self, filepath):
        self.base_path = filepath
        self.eeg_data = {}
        self.batch_size = 16
        self.sfreq = None

    @staticmethod
    def butter_highpass(cutoff, fs, order=2):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    @staticmethod
    def highpass_filter(data, cutoff, fs, order=2):
        b, a = CHBMITLoader.butter_highpass(cutoff, fs, order=order)
        y = lfilter(b, a, data, axis=1)
        return y

    @staticmethod
    def notch_filter(data, fs, freq=60.0, quality=30.0):
        b, a = iirnotch(freq / (0.5 * fs), quality)
        y = lfilter(b, a, data, axis=1)
        return y

    def extract_data_and_labels(self, edf_filename, summary_text):
        folder, basename = os.path.split(edf_filename)

        # Load EDF file with MNE
        try:
            edf = mne.io.read_raw_edf(edf_filename, stim_channel=None, preload=True, verbose='ERROR')
        except ValueError as e:
            print(f"Error reading {edf_filename} with MNE: {e}")
            return None, None, None

        X = edf.get_data().astype(np.float32) * 1e6  # to mV
        sfreq = edf.info['sfreq']  # Get the sampling frequency

        # Apply the high-pass Butterworth filter
        X = self.highpass_filter(X, cutoff=0.5, fs=sfreq)

        # Apply the notch filter
        X = self.notch_filter(X, fs=sfreq, freq=60.0)

        y = np.zeros(X.shape[1], dtype=np.int64)

        # Extract information from summary_text
        i_text_start = summary_text.index(basename)
        print(f"Processing file: {basename}, Sampling frequency: {sfreq} Hz")

        if 'File Name' in summary_text[i_text_start:]:
            i_text_stop = summary_text.index('File Name', i_text_start)
        else:
            i_text_stop = len(summary_text)

        assert i_text_stop > i_text_start

        file_text = summary_text[i_text_start:i_text_stop]

        if 'Seizure Start' in file_text:
            start_sec = int(re.search(r"Seizure Start Time: ([0-9]*) seconds", file_text).group(1))
            end_sec = int(re.search(r"Seizure End Time: ([0-9]*) seconds", file_text).group(1))
            i_seizure_start = int(round(start_sec * sfreq))
            i_seizure_stop = int(round((end_sec + 1) * sfreq))
            y[i_seizure_start:min(i_seizure_stop, len(y))] = 1
            print(f"Seizure detected from {start_sec}s to {end_sec}s.")
        else:
            print("No seizure detected in this file.")

        assert X.shape[1] == len(y)
        return X, y, sfreq

    def epoch_and_fft(self, X, y, epoch_length=1, overlap=0.5):
        samples_per_epoch = int(epoch_length * self.sfreq)
        step_size = int(samples_per_epoch * (1 - overlap))  # Calculate step size based on overlap
        
        X_final = []
        y_final = []

        for start_idx in range(0, X.shape[1] - samples_per_epoch + 1, step_size):
            end_idx = start_idx + samples_per_epoch
            X_epoch = X[:, start_idx:end_idx]
            y_epoch = y[start_idx:end_idx]
            
            # Apply FFT to the epoch for each channel
            X_fft = np.abs(np.fft.fft(X_epoch, axis=1))
            
            # Take only the first half of the FFT result (positive frequencies)
            X_fft = X_fft[:, :samples_per_epoch // 2]
            
            # Append the FFT-transformed epoch and the label
            X_final.append(X_fft)
            y_final.append([1, 0] if np.any(y_epoch) else [0, 1])
        
        return np.array(X_final), np.array(y_final)    
    
    def create_tf_datasets(self, windows_array, label_array):
        print(f"Splitting data into train and test datasets...")
        X_train, X_test, y_train, y_test = train_test_split(
            windows_array, label_array, test_size=0.2, random_state=None)

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

        train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        test_dataset = test_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        print(f"Train dataset shape: {X_train.shape}, Test dataset shape: {X_test.shape}")
        return train_dataset, test_dataset

    def load_dataset(self):
        for subject_folder in sorted(glob(os.path.join(self.base_path, "chb*"))):
            subject_id = os.path.basename(subject_folder)
            subject_key = f"subject_{subject_id[-2:]}"

            edf_file_names = sorted(glob(os.path.join(subject_folder, "*.edf")))
            summary_file = os.path.join(subject_folder, f"{subject_id}-summary.txt")
            
            if not os.path.exists(summary_file):
                print(f"Warning: Summary file missing for {subject_id}. Skipping...")
                continue

            summary_content = open(summary_file, 'r').read()

            all_X = []
            all_y = []
            epoch_length = 2  # 2 second epochs

            for edf_file_name in edf_file_names:
                X, y, sfreq = self.extract_data_and_labels(edf_file_name, summary_content)
                if X is None or y is None or sfreq is None:
                    print(f"Skipping file {edf_file_name} due to read errors.")
                    continue

                self.sfreq = sfreq  # Update the class variable with the current sampling frequency
                X_epochs, y_epochs = self.epoch_and_fft(X, y, epoch_length, overlap=0.5)
                all_X.append(X_epochs)
                all_y.append(y_epochs)

            if len(all_X) == 0 or len(all_y) == 0:
                print(f"No valid data found for {subject_key}. Skipping subject.")
                continue

            all_X = np.vstack(all_X)
            all_y = np.vstack(all_y)

            print(f"Final X shape for {subject_key}: {all_X.shape}, Final y shape: {all_y.shape}")

            train_dataset, test_dataset = self.create_tf_datasets(all_X, all_y)

            self.eeg_data[subject_key] = {
                'train_ds': train_dataset,
                'test_ds': test_dataset
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
        eeg_channels = ['EEG Fp1', 'EEG F3', 'EEG C3', 'EEG P3', 'EEG O1', 'EEG F7', 'EEG T3', 'EEG T5', 'EEG Fc1', 'EEG Fc5', 'EEG Cp1', 'EEG Cp5', 'EEG F9', 'EEG Fz', 'EEG Cz', 'EEG Pz', 'EEG F4', 'EEG C4', 'EEG P4', 'EEG O2', 'EEG F8', 'EEG T4', 'EEG T6', 'EEG Fc2', 'EEG Fc6', 'EEG Cp2', 'EEG Cp6', 'EEG F10']
        edf.pick_channels(eeg_channels)

        # Get the data as a numpy array and convert units to microvolts
        X = edf.get_data().astype(np.float32) * 1e6  # to µV
        y = np.zeros(X.shape[1], dtype=np.int64)
        sfreq = edf.info['sfreq']  # Get the sampling frequency

        # Extract information for seizure events from summary content
        i_text_start = summary_content.index(basename)
        print(f"Processing file: {basename}, Sampling frequency: {sfreq} Hz")

        if 'File name' in summary_content[i_text_start:]:
            i_text_stop = summary_content.index('File name', i_text_start) + len('File name')
        else:
            i_text_stop = len(summary_content)

        assert i_text_stop > i_text_start

        file_text = summary_content[i_text_start:i_text_stop]

        # Extract registration start time
        registration_start_time_str = re.search(r"Registration start time: (\d{2}[:.]\d{2}[:.]\d{2})", summary_content).group(1)

        # Extract seizure start and stop
        if 'Seizure start time' in file_text:
            seizure_start_time_str = re.search(r"Seizure start time: (\d{2}[:.]\d{2}[:.]\d{2})", summary_content).group(1)
            seizure_end_time_str = re.search(r"Seizure end time: (\d{2}[:.]\d{2}[:.]\d{2})", summary_content).group(1)

            # Calculate start and end times in seconds relative to the registration start time
            start_sec = self.calculate_seconds(registration_start_time_str, seizure_start_time_str)
            end_sec = self.calculate_seconds(registration_start_time_str, seizure_end_time_str)
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
            if np.any(y_epoch):
                print(f"Epoch {start_idx // step_size + 1}: X_epoch shape: {X_epoch.shape}, y_epoch label: {y_final[-1]}")
        
        return np.array(X_final), np.array(y_final)


    def create_tf_datasets(self, windows_array, label_array):
        X_train, X_test, y_train, y_test = train_test_split(
            windows_array, label_array, test_size=0.2, random_state=42)

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

        train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
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

    def segment_data(self, data):
        """Segment the data into overlapping windows."""
        segment_length = int(self.segment_length_sec * self.target_fs)
        overlap = int(self.overlap_sec * self.target_fs)
        step = segment_length - overlap

        segments = []
        for start in range(0, data.shape[1] - segment_length + 1, step):
            segments.append(data[:, start:start + segment_length])
        return np.array(segments)

    def create_tf_datasets(self, trials, labels, test_size=0.2):
        """Split trials into training and testing datasets, and convert to TensorFlow Dataset format."""
        X_train, X_test, y_train, y_test = train_test_split(trials, labels, test_size=test_size, stratify=labels)

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

        # Batch and shuffle the dataset
        train_dataset = train_dataset.shuffle(buffer_size=1000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        test_dataset = test_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        return train_dataset, test_dataset

    def load_dataset(self):
        """Load EEG data for each subject, downsample, segment, and prepare train/test datasets."""
        # Identify all subjects based on the directory contents
        subject_files = sorted([f for f in os.listdir(self.base_path) if f.endswith('.edf')])
        subjects = set([f.split('_')[0] for f in subject_files])  # Extract unique subject IDs

        for subject in subjects:
            print(f"Processing Subject: {subject}")

            # Load EDF files for both resting and task sessions
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
            print(f"Downsampled data: Resting state: {downsampled_eeg_resting.shape}, Task state: {downsampled_eeg_task.shape}")

            # Trim to make data length consistent for segmentation
            resting_trimmed = downsampled_eeg_resting[:, :23040]
            task_trimmed = downsampled_eeg_task[:, :7680]
            print(f"Trimmed data: Resting state: {resting_trimmed.shape}, Task state: {task_trimmed.shape}")

            # Segment the data
            resting_segments = self.segment_data(resting_trimmed)
            task_segments = self.segment_data(task_trimmed)
            print(f"Segmented data: Resting state: {resting_segments.shape}, Task state: {task_segments.shape}")

            # Concatenate resting and task data and create labels
            X = np.concatenate((resting_segments, task_segments), axis=0)
            y = np.array([0] * resting_segments.shape[0] + [1] * task_segments.shape[0])

            # Convert labels to one-hot encoded format
            y = np_utils.to_categorical(y, num_classes=2)
            print(f"Final data: X: {X.shape}, y: {y.shape}")

            # Create train and test datasets
            train_dataset, test_dataset = self.create_tf_datasets(X, y)

            # Store the datasets in the dictionary
            self.eeg_data[subject] = {
                'train_ds': train_dataset,
                'test_ds': test_dataset
            }

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

                        # Label and save epochs
                        if stimulus_type_train[trial, sample] == 1:
                            positive_epochs += 1
                            labels_train.append(1)
                            epochs_P300.append(filtered_epoch)

                            # Replicate P300 epoch 4 times to maintain balance
                            for _ in range(4):
                                positive_epochs += 1
                                labels_train.append(1)
                                epochs_train.append(filtered_epoch)
                        else:
                            negative_epochs += 1
                            labels_train.append(0)
                            epochs_noP300.append(filtered_epoch)

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

        # Prepare Butterworth filter
        butter_order = 8
        f_cut_low = 0.1
        f_cut_high = 20
        sos = signal.butter(butter_order, [f_cut_low, f_cut_high], 'bandpass', fs=self.target_fs, output='sos')

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
                        epoch = signal.sosfiltfilt(sos, epoch, axis=0)  # Apply filter
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


    def _to_tf_dataset(self, epochs, labels):
        """Convert epochs and labels to TensorFlow dataset."""
        dataset = tf.data.Dataset.from_tensor_slices((epochs, labels))
        return dataset.shuffle(buffer_size=10000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

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

            # Convert to TensorFlow datasets
            self.train_dataset = self._to_tf_dataset(epochs_train, labels_train)
            self.test_dataset = self._to_tf_dataset(epochs_test, labels_test)

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
        self.max_abs_val = 800
        self.original_freq = None
        self.target_freq = 128  # Target frequency to resample the data
        self.sec_to_cut = 60  # Seconds to cut from start and end
        self.selected_ch_names = np.array([
            'A1', 'A2', 'C3', 'C4', 'CZ', 'F3', 'F4', 'F7', 'F8', 'FP1', 'FP2', 'FZ',
            'O1', 'O2', 'P3', 'P4', 'PZ', 'T3', 'T4', 'T5', 'T6'
        ])
        self.eeg_data = {}
        self.eeg_data['subject'] = {}

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
                # Ensure there are enough time points for 60 seconds segments
                if data.shape[1] > 2 * 60 * self.target_freq:
                    rnd_start_idxs = np.random.randint(
                        int(60 * self.target_freq),
                        int(data.shape[1] - (60 * self.target_freq)),
                        size=max(1, data.shape[1] // (60 * self.target_freq) // 5)
                    )
                    data = np.vstack([
                        data[np.newaxis, :, idx:int(idx + 60 * self.target_freq)] for idx in rnd_start_idxs
                    ])
                    print(f"Data shape after extracting random segments for training: {data.shape}")
                else:
                    # If not enough data, use the entire available segment (or skip this file)
                    print(f"Not enough data for 60-second segments, using entire available data for {filename}")
                    data = data[:, :60 * self.target_freq]  # Use the first available minute
                    data = data[np.newaxis, :, :]  # Add a new axis for the batch

            elif '/eval/' in filename:
                # Cut the first and final 60 seconds and then take the first minute for evaluation
                data = data[:, int(60 * self.target_freq):-int(60 * self.target_freq)]
                data = data[:, :60 * self.target_freq]  # Extract the first minute
                data = data[np.newaxis, :, :]  # Add a new axis for the batch
                print(f"Data shape for evaluation after cutting: {data.shape}")

            # Clip the data to maximum absolute value
            data = data.clip(-self.max_abs_val, self.max_abs_val)
            print(f"Data shape after clipping: {data.shape}")

            # Create labels: 1 for abnormal, 0 for normal (based on filename)
            label = np.repeat(int('abnormal' in filename), data.shape[0])
            label = np_utils.to_categorical(label, 2)
            print(f"Label shape: {label.shape}, Labels: {label}")

            return data, label

        except IndexError:
            print(f"Skipping file {filename} due to missing channels.")
            return None, None

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
            train_ds = train_ds.shuffle(buffer_size=1000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
            print("Train dataset created successfully.")
        else:
            train_ds = None
            print("No training data found.")

        if test_data.size > 0:
            test_ds = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
            test_ds = test_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
            print("Test dataset created successfully.")
        else:
            test_ds = None
            print("No evaluation data found.")

        # Store the datasets in the 'subject' key of the eeg_data dictionary
        self.eeg_data['subject']['train_ds'] = train_ds
        self.eeg_data['subject']['test_ds'] = test_ds

        return self.eeg_data

