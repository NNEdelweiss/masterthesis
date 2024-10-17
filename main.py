import argparse
import os
from datetime import datetime
import json  # Added import for JSON handling
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau # type: ignore
from tensorflow.keras import backend as K  # type: ignore
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, recall_score, precision_score
from load_datasets import *  # Import the classes for loading datasets
import EEG_Models as eeg_models
# from utils import *
import h5py # To save/load datasets
import tensorflow as tf

# Configure TensorFlow to allow memory growth for GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled for GPU")
    except RuntimeError as e:
        print(f"Error enabling memory growth: {e}") 
        
# Global cache file
cache_file = "training_cache.json"
metrics_dir = None
result_dir = None

# Load or initialize the cache
def load_cache():
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    return {}

# Save the cache
def save_cache(cache):
    with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=4)

# Check if a model has already been run for a subject
def is_model_completed(cache, dataset_name, model_name, subject):
    return cache.get(dataset_name, {}).get(model_name, {}).get(subject, False)

# Mark the model as completed for a subject
def mark_model_as_completed(cache, dataset_name, model_name, subject):
    if dataset_name not in cache:
        cache[dataset_name] = {}
    if model_name not in cache[dataset_name]:
        cache[dataset_name][model_name] = {}
    cache[dataset_name][model_name][subject] = True
    save_cache(cache)

# Setup logging
def get_logger(save_result, save_dir, save_file):
    logger = logging.getLogger(__name__)  # Use a specific logger name to avoid conflicts
    logger.setLevel(logging.INFO)

    # Check if handlers already exist to avoid adding them multiple times
    if not logger.hasHandlers():
        formatter = logging.Formatter(fmt="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

        # StreamHandler for console output
        str_handler = logging.StreamHandler()
        str_handler.setFormatter(formatter)
        logger.addHandler(str_handler)

        # If saving logs to file, add a FileHandler
        if save_result:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            file_handler = logging.FileHandler(os.path.join(save_dir, save_file), mode='w')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    # Disable propagation to avoid duplicate logging from root logger
    logger.propagate = False

    return logger

def evaluate_model(model, test_dataset):
    y_true = np.concatenate([label.numpy() for _, label in test_dataset], axis=0)
    y_pred = model.predict(test_dataset)
    y_pred = np.argmax(y_pred, axis=1)

    if len(y_true.shape) > 1:  # Only apply argmax if y_true is one-hot encoded
        y_true = np.argmax(y_true, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_true, y_pred)
    classification_rep = classification_report(y_true, y_pred, output_dict=True)

    return accuracy, f1, recall, precision, conf_matrix, classification_rep

def setup_callbacks(dataset_name, model_name, subject):
    global result_dir

    checkpoint_filepath = os.path.join(result_dir, f'{dataset_name}_{model_name}_{subject}_best_model.h5')
    model_checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    csv_logger = CSVLogger(os.path.join(result_dir, f'{dataset_name}_{model_name}_{subject}_training_log.txt'), append=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    return [model_checkpoint, csv_logger, reduce_lr]

def train_model(model_name, train_dataset, test_dataset, dataset_name, subject, label_names, nb_classes, nchan, trial_length, epochs=50):
    global metrics_dir
    global result_dir

    # Set up callbacks
    callbacks = setup_callbacks(dataset_name, model_name, subject)

    # Initialize the model based on the type
    if model_name == 'DeepSleepNet':
        # Train the pre-trained model first
        pretrained_model = eeg_models.DSN_preTrainingNet(nchan=nchan, trial_length=trial_length, n_classes=nb_classes)
        pretrain_history = pretrained_model.fit(train_dataset, epochs=int(epochs / 2), verbose=1, validation_data=test_dataset, callbacks=callbacks)

        # Plot training history for the pre-trained model
        plot_training_history(pretrain_history, dataset_name, model_name + "_pretrain", subject, epochs)

        # Create and train the fine-tuning model
        fine_tuned_model = eeg_models.DSN_fineTuningNet(nchan=nchan, trial_length=trial_length, n_classes=nb_classes, preTrainedNet=pretrained_model)
        finetune_history = fine_tuned_model.fit(train_dataset, epochs=int(epochs / 2), verbose=1, validation_data=test_dataset, callbacks=callbacks)

        # Plot training history for the fine-tuned model
        plot_training_history(finetune_history, dataset_name, model_name + "_finetune", subject, epochs)

        # Use fine-tuned model for evaluation
        final_model = fine_tuned_model
    else:
        # For all other models, load and train
        model = eeg_models.load_model(model_name, nb_classes=nb_classes, nchan=nchan, trial_length=trial_length)
        history = model.fit(train_dataset, epochs=epochs, verbose=1, validation_data=test_dataset, callbacks=callbacks)

        # Plot training history for the regular model
        plot_training_history(history, dataset_name, model_name, subject, epochs)

        final_model = model

    # Evaluate the model
    metrics = evaluate_model(final_model, test_dataset)
    accuracy = metrics[0]

    # Save metrics and confusion matrix
    save_metrics_and_plots(*metrics, dataset_name, model_name, subject, label_names)

    return accuracy

def save_metrics_and_plots(accuracy, f1, recall, precision, conf_matrix, classification_rep, dataset_name, model_name, subject, label_names):
    global metrics_dir

    # Construct file names with dataset and model names
    metrics_file = os.path.join(metrics_dir, f'{dataset_name}_{model_name}_{subject}_metrics.json')
    conf_matrix_file = os.path.join(metrics_dir, f'{dataset_name}_{model_name}_{subject}_confusion_matrix.csv')
    conf_matrix_plot = os.path.join(metrics_dir, f'{dataset_name}_{model_name}_{subject}_confusion_matrix.pdf')

    # Save metrics as JSON
    metrics = {
        'accuracy': round(accuracy, 3),
        'f1_score': round(f1, 3),
        'recall': round(recall, 3),
        'precision': round(precision, 3),
        'classification_report': classification_rep
    }
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)

    # Save confusion matrix as a CSV file
    np.savetxt(conf_matrix_file, conf_matrix, delimiter=",", fmt='%d')

    # Plot confusion matrix
    plt.figure(figsize=(10, 10))
    percentages = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis] * 100 # Normalize by row

    # Use sns.heatmap to plot the confusion matrix with percentages
    sns.heatmap(percentages, annot=np.vectorize(lambda x: f'{x:.2f}%')(percentages), fmt='', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names, annot_kws={"size": 12}, cbar=False)

    # Labeling and title improvements
    plt.xlabel('Predicted Label', fontsize=13, labelpad=11)
    plt.ylabel('Actual Label', fontsize=13, labelpad=11)
    plt.xticks(size=12)
    plt.yticks(size=12)

    plt.savefig(conf_matrix_plot, format='pdf', bbox_inches='tight')
    plt.tight_layout()
    plt.close()  # Close the plot when running in non-interactive environments

    print(f"Metrics and confusion matrix saved to {metrics_dir}")

def plot_training_history(history, dataset_name, model_name, subject, epochs):
    global metrics_dir

    """Plot and save the training history of accuracy and loss."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    # Plotting accuracy and loss
    epochs_range = range(1, len(acc) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs_range, acc, 'b', label='Training Accuracy')
    ax1.plot(epochs_range, val_acc, 'r', label='Validation Accuracy')
    ax1.set_title('Training and Validation Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    ax2.plot(epochs_range, loss, 'b', label='Training Loss')
    ax2.plot(epochs_range, val_loss, 'r', label='Validation Loss')
    ax2.set_title('Training and Validation Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()

    # Save the plot as a high-quality PDF
    pdf_plot_file = os.path.join(metrics_dir, f'{dataset_name}_{model_name}_{subject}_training_history.pdf')
    plt.savefig(pdf_plot_file, format='pdf', bbox_inches='tight')
    plt.close()  # Close the plot when running in non-interactive environments
    print(f"Training history saved as PDF to {pdf_plot_file}")

def save_dataset_h5(eeg_data, filename):
    with h5py.File(filename, 'w') as f:
        for subject, datasets in eeg_data.items():
            for dataset_type, dataset in datasets.items():
                x_data, y_data = [], []
                for x_batch, y_batch in dataset:
                    x_data.append(x_batch.numpy().astype(np.float32))  # Explicit data type
                    y_data.append(y_batch.numpy().astype(np.float32))  # Explicit data type
                if len(x_data) == 0 or len(y_data) == 0:
                    print(f"Skipping {subject}_{dataset_type} due to empty dataset")
                    continue
                # Saving with compression
                f.create_dataset(f'{subject}_{dataset_type}_x', data=np.concatenate(x_data, axis=0), compression="gzip")
                f.create_dataset(f'{subject}_{dataset_type}_y', data=np.concatenate(y_data, axis=0), compression="gzip")
                print(f"Saved {subject}_{dataset_type}_x and {subject}_{dataset_type}_y")

def load_dataset_h5(filename):
    with h5py.File(filename, 'r') as f:
        eeg_data = {}
        print(f"Keys in HDF5 file: {list(f.keys())}")
        
        subject_keys = set([key.split('_')[0] for key in f.keys()])
        for subject_key in subject_keys:
            eeg_data[subject_key] = {}
            for dataset_type in ['train_ds', 'test_ds']:
                x_key = f'{subject_key}_{dataset_type}_x'
                y_key = f'{subject_key}_{dataset_type}_y'

                if x_key not in f or y_key not in f:
                    print(f"Key {x_key} or {y_key} not found in HDF5 file.")
                    continue

                x_data = f[x_key][:]
                y_data = f[y_key][:]

                if x_data.shape[0] != y_data.shape[0]:
                    raise ValueError(f"Shape mismatch between {x_key} and {y_key}")

                x_tensor = tf.convert_to_tensor(x_data, dtype=tf.float32)
                y_tensor = tf.convert_to_tensor(y_data, dtype=tf.float32)
                dataset = tf.data.Dataset.from_tensor_slices((x_tensor, y_tensor)).batch(16)
                eeg_data[subject_key][dataset_type] = dataset
                print(f"Loaded {subject_key} {dataset_type} dataset")

    return eeg_data

def load_dataset(args, dataset_config):
    # Dataset storage path
    os.makedirs('loading_datasets', exist_ok=True)

    dataset_file = os.path.join('loading_datasets', f'{args.dataset}_data.h5')

    if args.dataset in dataset_config:
        config = dataset_config[args.dataset]
        nb_classes = config['nb_classes']
        nchan = config['nchan']
        trial_length = config['trial_length']
        label_names = config['label_names']
        data_loader_class = globals()[config['loader_class']]
        data_loader = data_loader_class(**config.get('loader_args', {}))
    else:
        raise ValueError(f"Dataset {args.dataset} is not supported.")

    # # Load dataset
    # eeg_data = data_loader.load_dataset()

    if os.path.exists(dataset_file):
        # Load the dataset if it already exists
        eeg_data = load_dataset_h5(dataset_file)
    else:
        # Load the dataset using the loader if the file doesn't exist
        eeg_data = data_loader.load_dataset()
        save_dataset_h5(eeg_data, dataset_file)
    
    return eeg_data, nb_classes, nchan, trial_length, label_names

def train_all_models(args, models, eeg_data, nb_classes, nchan, trial_length, label_names, cache):
    global metrics_dir
    global result_dir
    
    # Run through all models for the dataset
    for model_name in models:
        subfolder = f'{args.dataset}_{model_name}'
        metrics_dir = os.path.join(os.getcwd(), 'metrics', args.dataset, subfolder)
        os.makedirs(metrics_dir, exist_ok=True)

        result_dir = os.path.join(os.getcwd(), 'best_model', args.dataset, subfolder)
        os.makedirs(result_dir, exist_ok=True)

        current_time = datetime.now().strftime('%Y%m%d_%H%M')
        save_dir = f"{os.getcwd()}/log/"
        logger = get_logger(save_result=True, save_dir=save_dir, save_file=f"{current_time}_{args.dataset}.log")
        logger.info(f"Starting Experiment for {args.dataset}")
        logger.info(f"Running model: {model_name}")

        accuracy_file = os.path.join(result_dir, f'{args.dataset}_{model_name}_accuracy.txt')

        accuracies = []
        try:
            with open(accuracy_file, 'w') as f:
                for subject, datasets in eeg_data.items():
                    train_dataset = datasets.get('train_ds')
                    test_dataset = datasets.get('test_ds')

                    if train_dataset is None or test_dataset is None:
                        logger.warning(f"Missing datasets for subject {subject}. Skipping.")
                        continue

                    # Check cache to see if this model has already been run for this subject
                    if is_model_completed(cache, args.dataset, model_name, subject):
                        logger.info(f"Skipping {model_name} for subject {subject} (already completed)")
                        continue

                    # Train and evaluate model for each subject
                    try:
                        accuracy = train_model(model_name, train_dataset, test_dataset, args.dataset, subject, label_names, nb_classes, nchan, trial_length, epochs=args.epochs)
                        accuracies.append(accuracy)
                    except Exception as e:
                        logger.error(f"Error training {model_name} for subject {subject}: {e}")
                        continue

                    logger.info(f"Subject {subject}, Model {model_name}: Accuracy = {accuracy}")
                    f.write(f"Subject {subject}: Accuracy = {accuracy}\n")

                # Calculate average accuracy for the model across all subjects
                avg_accuracy = np.mean(accuracies) if accuracies else 0.0
                logger.info(f"Model {model_name}: Average Accuracy across subjects: {avg_accuracy}")
                f.write(f"\nAccuracies for all subjects: {accuracies}\n")
                f.write(f'Average Accuracy: {avg_accuracy}\n')
            
            logger.info(f"Accuracies and average accuracy saved to {accuracy_file}")

            # Mark the model as completed for all subjects after calculating average accuracy
            for subject in eeg_data.keys():
                mark_model_as_completed(cache, args.dataset, model_name, subject)

            logger.info(f"Model {model_name} marked as completed for all subjects.")

        except Exception as e:
            logger.error(f"Error processing model {model_name}: {e}")
        
        # Clear TensorFlow session to free up memory before training the next model
        K.clear_session()

def main():
    # Load cache at the start of the script
    cache = load_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bciciv2a', 
                        help='dataset used for the experiments')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')  # Added epochs as an argument
    args = parser.parse_args()

    # List of models to run
    models = ['EEGNet', 'DeepConvNet', 'ShallowConvNet', 'CNN_FC', 
              'CRNN', 'MMCNN', 'ChronoNet', 'ResNet', 'Attention_1DCNN',
              'EEGTCNet', 'BLSTM_LSTM','DeepSleepNet']    
    
    # Load dataset configuration from JSON file
    with open('dataset_config.json', 'r') as f:
        dataset_config = json.load(f)

    # Load dataset
    eeg_data, nb_classes, nchan, trial_length, label_names = load_dataset(args, dataset_config)

    # Train all models
    train_all_models(args, models, eeg_data, nb_classes, nchan, trial_length, label_names, cache)    
    

if __name__ == '__main__':
    main()

