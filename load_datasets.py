import os
import re
from glob import glob
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
from scipy.signal import resample, butter, filtfilt 

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
    def __init__(self, folder_path, window_size=512, overlap=128):
        self.folder_path = folder_path  # Path containing EEG files
        self.channels = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]  # List of channel names
        self.sfreq = 128  # Sampling frequency
        self.window_size = window_size  # Window size in samples
        self.overlap = overlap  # Overlap in samples
        self.batch_size = 16  # Batch size for the datasets
        self.eeg_data = {}  # Dictionary to store train and test datasets for each subject

        # Get all file paths
        self.all_files_path = glob(os.path.join(folder_path, '*.txt'))
        self.file_names = [x for x in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, x)) and x.endswith(".txt") and x.startswith("sub")]
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


# Example usage
folder_path = "../Dataset/STEW/"
loader = STEWLoader(folder_path)

# Load the dataset for all subjects
eeg_data = loader.load_dataset()

# Access train and test datasets for a specific subject (e.g., subject 01)
train_dataset = eeg_data['subject_01']['train_ds']
test_dataset = eeg_data['subject_01']['test_ds']

# Iterate through the dataset and print the shape of the first batch
for data, labels in train_dataset.take(1):
    print("Training batch shape:", data.shape, labels.shape)

for data, labels in test_dataset.take(1):
    print("Testing batch shape:", data.shape, labels.shape)


