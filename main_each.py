import argparse
import os
from datetime import datetime
import json  # Added import for JSON handling
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, LearningRateScheduler # type: ignore
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, recall_score, precision_score
from load_datasets import *  # Import the classes for loading datasets
import EEG_Models as eeg_models
import h5py # To save/load datasets
import tensorflow as tf
from tensorflow.keras import backend as K  # type: ignore

# Configure TensorFlow to allow memory growth for GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled for GPU")
    except RuntimeError as e:
        print(f"Error enabling memory growth: {e}") 
        
# Declare global variables
metrics_dir = None
accuracy_file = None

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

def train_model(model_name, train_dataset, test_dataset, dataset_name, subject, label_names, nb_classes, nchan, trial_length, epochs=20):
    global metrics_dir

    # Initialize the model based on the type
    if model_name == 'DeepSleepNet':
        # Train the pre-trained model first
        pretrained_model = eeg_models.DSN_preTrainingNet(nchan=nchan, trial_length=trial_length, n_classes=nb_classes)
        pretrain_history = pretrained_model.fit(train_dataset, epochs=int(epochs / 2), verbose=1)

        # Plot training history for the pre-trained model
        plot_training_history(pretrain_history, dataset_name, model_name + "_pretrain", subject, epochs)

        # Create and train the fine-tuning model
        fine_tuned_model = eeg_models.DSN_fineTuningNet(nchan=nchan, trial_length=trial_length, n_classes=nb_classes, preTrainedNet=pretrained_model)
        finetune_history = fine_tuned_model.fit(train_dataset, epochs=int(epochs / 2), verbose=1)

        # Plot training history for the fine-tuned model
        plot_training_history(finetune_history, dataset_name, model_name + "_finetune", subject, epochs)

        # Use fine-tuned model for evaluation
        final_model = fine_tuned_model
    else:
        # For all other models, load and train
        model = eeg_models.load_model(model_name, nb_classes=nb_classes, nchan=nchan, trial_length=trial_length)
        history = model.fit(train_dataset, epochs=epochs, verbose=1)

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
                xticklabels=label_names, yticklabels=label_names, annot_kws={"size": 12}, cbar=False,
                vmin=0, vmax=100)

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
    # val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    # val_loss = history.history['val_loss']
    
    # Plotting accuracy and loss
    epochs_range = range(1, len(acc) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs_range, acc, 'b', label='Training Accuracy')
    # ax1.plot(epochs_range, val_acc, 'r', label='Validation Accuracy')
    # ax1.set_title('Training and Validation Accuracy')
    ax1.set_title('Training Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim([0, 1]) # Limit y-axis to 0-1
    ax1.legend()

    ax2.plot(epochs_range, loss, 'b', label='Training Loss')
    # ax2.plot(epochs_range, val_loss, 'r', label='Validation Loss')
    # ax2.set_title('Training and Validation Loss')
    ax2.set_title('Training Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()

    # Save the plot as a high-quality PDF
    pdf_plot_file = os.path.join(metrics_dir, f'{dataset_name}_{model_name}_{subject}_training_history.pdf')
    plt.savefig(pdf_plot_file, format='pdf', bbox_inches='tight')
    plt.close()  # Close the plot when running in non-interactive environments
    print(f"Training history saved as PDF to {pdf_plot_file}")

def save_dataset_h5(eeg_data, filename, cross_validation=False):
    with h5py.File(filename, 'w') as f:
        if cross_validation:
            # Save the entire dataset in 'all'
            all_trials = eeg_data['all']['trials']
            all_labels = eeg_data['all']['labels']
            f.create_dataset('all_x', data=all_trials, compression="gzip")
            f.create_dataset('all_y', data=all_labels, compression="gzip")
            print("Saved the entire dataset for cross-validation as 'all_x' and 'all_y'.")

            # Save indices for each fold
            for fold, indices in eeg_data.items():
                if isinstance(fold, int):  # Only save fold-specific data (fold+1 keys)
                    train_indices = indices['train_indices']
                    test_indices = indices['test_indices']
                    f.create_dataset(f'fold_{fold}_train_indices', data=train_indices, compression="gzip")
                    f.create_dataset(f'fold_{fold}_test_indices', data=test_indices, compression="gzip")
                    print(f"Saved indices for fold {fold} (train and test).")
        else:
            # Original subject-specific saving (unchanged)
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

def load_dataset_h5(filename, cross_validation=False):
    with h5py.File(filename, 'r') as f:
        eeg_data = {}
        print(f"Keys in HDF5 file: {list(f.keys())}")

        if cross_validation:
            # Load the entire dataset into 'all'
            all_trials = f['all_x'][:]
            all_labels = f['all_y'][:]
            eeg_data['all'] = {'trials': all_trials, 'labels': all_labels}
            print("Loaded the entire dataset for cross-validation as 'all_x' and 'all_y'.")

            # Load fold indices only
            for key in f.keys():
                if key.startswith("fold_") and key.endswith("_train_indices"):
                    fold = int(key.split('_')[1])
                    train_indices = f[f'fold_{fold}_train_indices'][:]
                    test_indices = f[f'fold_{fold}_test_indices'][:]
                    
                    eeg_data[fold] = {'train_indices': train_indices, 'test_indices': test_indices}
                    print(f"Loaded indices for fold {fold} (train and test).")
        else:
            # Original subject-specific loading (unchanged)
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

    if os.path.exists(dataset_file):
        # Load the dataset if it already exists
        eeg_data = load_dataset_h5(dataset_file, cross_validation=args.cross_validation)
    else:
        # Load the dataset using the loader if the file doesn't exist
        eeg_data = data_loader.load_dataset()
        save_dataset_h5(eeg_data, dataset_file, cross_validation=args.cross_validation)
    
    return eeg_data, nb_classes, nchan, trial_length, label_names

# Helper function for setting up directories
def setup_dirs(args):
    subfolder = f'{args.dataset}_{args.model}'
    metrics_dir = os.path.join(os.getcwd(), 'metrics', args.dataset, subfolder)
    os.makedirs(metrics_dir, exist_ok=True)

    print(f"Starting Experiment for {args.dataset}")
    print(f"Running model: {args.model}")

    accuracy_file = os.path.join(metrics_dir, f'accuracy_{args.dataset}_{args.model}.txt')

    return metrics_dir, accuracy_file

# Function for cross-validation training on a model
def cross_validate_model(eeg_data, args, label_names, nb_classes, nchan, trial_length):
    global accuracy_file
    all_trials = eeg_data['all']['trials']
    all_labels = eeg_data['all']['labels']
    fold_accuracies = []

    # Dynamically determine the number of folds based on eeg_data keys
    fold_keys = [key for key in eeg_data.keys() if isinstance(key, int) and 'train_indices' in eeg_data[key] and 'test_indices' in eeg_data[key]]
    print(f"Folds: {fold_keys}")

    for fold in fold_keys:
        fold_name = f"Fold_{fold}"

        train_indices = eeg_data[fold]['train_indices']
        test_indices = eeg_data[fold]['test_indices']

        # Directly create train and test datasets within cross-validation
        X_train, y_train = all_trials[train_indices], all_labels[train_indices]
        X_test, y_test = all_trials[test_indices], all_labels[test_indices]

        #print shape
        print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

        train_dataset = train_dataset.shuffle(buffer_size=10000).batch(16).prefetch(tf.data.AUTOTUNE)
        test_dataset = test_dataset.batch(16).prefetch(tf.data.AUTOTUNE)
        
        print(f"Training {args.model} for {fold_name}...")  

        try:
            accuracy = train_model(
                args.model, train_dataset, test_dataset, args.dataset, f"fold_{fold}", 
                label_names, nb_classes, nchan, trial_length, epochs=args.epochs
            )

            with open(accuracy_file, 'a') as f:
                f.write(f"Fold {fold}: {accuracy:.2f}\n")

            fold_accuracies.append(accuracy)

            print(f"{fold_name}, Model {args.model}: Accuracy = {accuracy}")

        except Exception as e:
            print(f"Error in cross-validation for {args.model} on fold {fold}: {e}")
            continue

        K.clear_session()

    avg_fold_accuracy = np.mean(fold_accuracies) if fold_accuracies else 0.0
    # print(f"Model {args.model}, Cross-Validation Average Accuracy: {avg_fold_accuracy}")
    return fold_accuracies, avg_fold_accuracy

# def cross_validate_model(eeg_data, args, label_names, nb_classes, nchan, trial_length):
#     global accuracy_file
#     all_trials = eeg_data['all']['trials']
#     all_labels = eeg_data['all']['labels']
#     fold_accuracies = []

#     # Dynamically determine the number of folds based on eeg_data keys
#     # fold_keys = [key for key in eeg_data.keys() if isinstance(key, int) and 'train_indices' in eeg_data[key] and 'test_indices' in eeg_data[key]]
#     fold_keys = [2,3]
#     print(f"Folds: {fold_keys}")

#     for fold in fold_keys:
#         fold_name = f"Fold_{fold}"

#         train_indices = eeg_data[fold]['train_indices']
#         test_indices = eeg_data[fold]['test_indices']

#         # Directly create train and test datasets within cross-validation
#         X_train, y_train = all_trials[train_indices], all_labels[train_indices]
#         X_test, y_test = all_trials[test_indices], all_labels[test_indices]

#         #print shape
#         print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
#         print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

#         train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
#         test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

#         train_dataset = train_dataset.shuffle(buffer_size=10000).batch(16).prefetch(tf.data.AUTOTUNE)
#         test_dataset = test_dataset.batch(16).prefetch(tf.data.AUTOTUNE)

#         print(f"Training {args.model} for {fold_name}...")

#         try:
#             accuracy = train_model(
#                 args.model, train_dataset, test_dataset, args.dataset, f"fold_{fold}", 
#                 label_names, nb_classes, nchan, trial_length, epochs=args.epochs
#             )

#             with open(accuracy_file, 'a') as f:
#                 f.write(f"Fold {fold}: {accuracy:.2f}\n")

#             fold_accuracies.append(accuracy)

#             print(f"{fold_name}, Model {args.model}: Accuracy = {accuracy}")

#         except Exception as e:
#             print(f"Error in cross-validation for {args.model} on fold {fold}: {e}")
#             continue

#         K.clear_session()

#     avg_fold_accuracy = np.mean(fold_accuracies) if fold_accuracies else 0.0
#     # print(f"Model {args.model}, Cross-Validation Average Accuracy: {avg_fold_accuracy}")
#     return fold_accuracies, avg_fold_accuracy

# Function for direct training on each subject
def train_on_subjects(eeg_data, args, label_names, nb_classes, nchan, trial_length):
    global accuracy_file
    accuracies = []

    for idx, datasets in eeg_data.items():
        train_dataset = datasets.get('train_ds')
        test_dataset = datasets.get('test_ds')
        
        if train_dataset is None or test_dataset is None:
            print(f"Missing train/test datasets for subject {idx}. Skipping.")
            continue

        accuracy = None  # Initialize accuracy to avoid reference before assignment
        
        try:
            accuracy = train_model(
                args.model, train_dataset, test_dataset, args.dataset, idx, 
                label_names, nb_classes, nchan, trial_length, epochs=args.epochs
            )

            with open(accuracy_file, 'a') as f:
                f.write(f"subject {idx}: Accuracy = {accuracy:.2f}\n")

            accuracies.append(accuracy)

        except Exception as e:
            print(f"Error training {args.model} for subject {idx}: {e}")
            # continue

        if accuracy is not None:
            print(f"Subject {idx}, Model {args.model}: Accuracy = {accuracy}")
            
        K.clear_session()

    avg_accuracy = np.mean(accuracies) if accuracies else 0.0
    # print(f"Model {args.model}: Average Accuracy: {avg_accuracy}")
    return accuracies, avg_accuracy

# Main function to train all models
def train_all_models(args, eeg_data, nb_classes, nchan, trial_length, label_names, cross_validation=False):
    global metrics_dir
    global accuracy_file

    metrics_dir, accuracy_file = setup_dirs(args)
    
    try:
        with open(accuracy_file, 'a') as f:
            if cross_validation:
                accuracies, avg_accuracy = cross_validate_model(
                    eeg_data, args, label_names, nb_classes, nchan, trial_length
                )
            else:
                accuracies, avg_accuracy = train_on_subjects(
                    eeg_data, args, label_names, nb_classes, nchan, trial_length
                )

            print(f"Model {args.model}: Final Average Accuracy: {avg_accuracy}")
            f.write(f"\nAccuracies for all subjects/folds: {accuracies}\n")
            f.write(f"\nFinal Average Accuracy: {avg_accuracy:.2f}\n")

        print(f"Model {args.model} marked as completed for all subjects.")

    except Exception as e:
        print(f"Error processing model {args.model}: {e}")
    
    K.clear_session()
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bciciv2a', 
                        help='dataset used for the experiments')
    parser.add_argument('--model', type=str, default='EEGNet', 
                        help='model used for the experiments')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')  # Added epochs as an argument
    parser.add_argument('--cross_validation', action='store_true', 
                        help='Enable k-fold cross-validation')  # New argument for cross-validation
    
    args = parser.parse_args()

    # Load dataset configuration from JSON file
    with open('dataset_config.json', 'r') as f:
        dataset_config = json.load(f)

    # Load dataset
    eeg_data, nb_classes, nchan, trial_length, label_names = load_dataset(args, dataset_config)

    # Train all models
    train_all_models(args, eeg_data, nb_classes, nchan, trial_length, label_names, cross_validation=args.cross_validation)    

if __name__ == '__main__':
    main()
