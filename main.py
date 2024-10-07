import argparse
import os
import time
import json  # Added import for JSON handling
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau # type: ignore
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, recall_score, precision_score
from load_datasets import *  # Import the classes for loading datasets
import EEG_Models as eeg_models
from utils import get_logger
import h5py # To save/load datasets
import tensorflow as tf

# Declare metrics_dir as a global variable
metrics_dir = None

def train_model(model, train_dataset, test_dataset, dataset_name, model_name, subject, label_names, epochs=20):
    global metrics_dir
    
    # Ensure result directory exists
    subfolder = f'{dataset_name}_{model_name}'
    result_dir = os.path.join(os.getcwd(), 'result', subfolder)
    os.makedirs(result_dir, exist_ok=True)

    # Set up callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint_filepath = os.path.join(result_dir, f'{dataset_name}_{model_name}_{subject}_best_model.h5')
    model_checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    csv_logger = CSVLogger(os.path.join(result_dir, f'{dataset_name}_{model_name}_{subject}_training_log.txt'), append=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

    # Train the model
    history = model.fit(
        train_dataset, 
        epochs=epochs, 
        verbose=1, 
        validation_data=test_dataset, 
        callbacks=[reduce_lr, model_checkpoint, csv_logger]
        #callbacks=[early_stopping, model_checkpoint, csv_logger]
    )

    # Evaluate the model
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

    # Save metrics and confusion matrix
    save_metrics_and_plots(accuracy, f1, recall, precision, conf_matrix, classification_rep, dataset_name, model_name, subject, label_names)

    # Plot and save training history
    plot_training_history(history, dataset_name, model_name, subject, epochs)

    return accuracy

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

def save_metrics_and_plots(accuracy, f1, recall, precision, conf_matrix, classification_rep, dataset_name, model_name, subject, label_names):
    global metrics_dir

    # Construct file names with dataset and model names
    metrics_file = os.path.join(metrics_dir, f'{dataset_name}_{model_name}_{subject}_metrics.json')
    conf_matrix_file = os.path.join(metrics_dir, f'{dataset_name}_{model_name}_{subject}_confusion_matrix.csv')
    conf_matrix_plot = os.path.join(metrics_dir, f'{dataset_name}_{model_name}_{subject}_confusion_matrix.png')

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
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(conf_matrix_plot)
    plt.show()

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

    plt.suptitle(f'{dataset_name}-{model_name}-Subject {subject} Training History')
    plot_file = os.path.join(metrics_dir, f'{dataset_name}_{model_name}_{subject}_training_history.png')
    plt.savefig(plot_file)
    plt.show()

    print(f"Training history saved to {plot_file}")

def save_dataset_h5(eeg_data, filename):
    with h5py.File(filename, 'w') as f:
        for subject, datasets in eeg_data.items():
            for dataset_type, dataset in datasets.items():
                x_data, y_data = [], []
                for x_batch, y_batch in dataset:
                    x_data.append(x_batch.numpy())  # Convert Tensor to NumPy
                    y_data.append(y_batch.numpy())  # Convert Tensor to NumPy
                f.create_dataset(f'{subject}_{dataset_type}_x', data=np.concatenate(x_data, axis=0))
                f.create_dataset(f'{subject}_{dataset_type}_y', data=np.concatenate(y_data, axis=0))

def load_dataset_h5(filename):
    with h5py.File(filename, 'r') as f:
        eeg_data = {}
        for subject_key in set([key.split('_')[0] for key in f.keys()]):
            eeg_data[subject_key] = {}
            for dataset_type in ['train_ds', 'test_ds']:
                x_data = f[f'{subject_key}_{dataset_type}_x'][:]
                y_data = f[f'{subject_key}_{dataset_type}_y'][:]
                x_tensor = tf.convert_to_tensor(x_data, dtype=tf.float32)  # Convert NumPy back to Tensor
                y_tensor = tf.convert_to_tensor(y_data, dtype=tf.float32)  # Convert NumPy back to Tensor
                dataset = tf.data.Dataset.from_tensor_slices((x_tensor, y_tensor)).batch(16)
                eeg_data[subject_key][dataset_type] = dataset
    return eeg_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bciciv2a', choices=['bciciv2a', 'physionetIM', 'bciciv2b','dreamer_arousal', 'dreamer_valence', 
                                                                            'seed', 'deap_arousal', 'deap_valence', 'stew', 'chbmit', 'siena', 'eegmat', 
                                                                            'tuh_abnormal', 'bciciii2'], 
                        help='dataset used for the experiments')
    parser.add_argument('--model', type=str, default='EEGNet', choices=['EEGNet', 'DeepConvNet_origin', 'ATCNet', 'DeepConvNet', 'ShallowConvNet', 'CNN_FC', 
                                                                        'CRNN', 'MMCNN_model', 'ChronoNet', 'EEGTCNet', 'ResNet', 'CNN3D', 'Attention_1DCNN',
                                                                        'EEGTCNet'], 
                        help='model used for the experiments')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for training')  # Added epochs as an argument
    # parser.add_argument('--earlystopping', type=bool, default=False, help='Whether to use early stopping')  # Added early stopping as an argument
    args = parser.parse_args()

    # Define metrics_dir globally based on arguments
    subfolder = f'{args.dataset}_{args.model}'
    metrics_dir = os.path.join(os.getcwd(), 'metrics', subfolder)
    os.makedirs(metrics_dir, exist_ok=True)

    # Setup logging and result directory
    save_dir = f"{os.getcwd()}/save/{int(time.time())}_{args.dataset}_{args.model}/"
    logger = get_logger(save_result=True, save_dir=save_dir, save_file='result.log')
    logger.info("Starting experiment")

    # Dataset storage path
    os.makedirs('loading_datasets', exist_ok=True)

    dataset_file = os.path.join('loading_datasets', f'{args.dataset}_data.h5')

    # Initialize these variables regardless of loading or creating the dataset
    if args.dataset == 'bciciv2a':
        nb_classes, chans, samples = 4, 22, 256
        label_names = ['Left', 'Right', 'Foot', 'Tongue']
        data_loader = BCICIV2aLoader(filepath="../Dataset/BCICIV_2a_gdf/", stimcodes=['769', '770', '771', '772'])
    elif args.dataset == 'bciciv2b':
        nb_classes, chans, samples = 2, 3, 256
        label_names = ['Left', 'Right']
        data_loader = BCICIV2bLoader(filepath="../Dataset/BCICIV_2b_gdf/")
    elif args.dataset == 'physionetIM':
        nb_classes, chans, samples = 4, 64, 512
        label_names = ['Rest', 'Fist Left', 'Fist Right', 'Feet']
        data_loader = PhysionetMILoader(filepath="../Dataset/physionet_mi")
    elif args.dataset == 'dreamer_arousal':
        nb_classes, chans, samples = 2, 14, 128
        label_names = ['Low Arousal', 'High Arousal']
        data_loader = DREAMERLoader(filepath="../Dataset/DREAMER/DREAMER.mat", label_type='arousal')
    elif args.dataset == 'dreamer_valence':
        nb_classes, chans, samples = 2, 14, 128
        label_names = ['Low Valence', 'High Valence']
        data_loader = DREAMERLoader(filepath="../Dataset/DREAMER/DREAMER.mat", label_type='valence')
    elif args.dataset == 'seed':
        nb_classes, chans, samples = 3, 62, 1024
        label_names = ['Negative', 'Neural', 'Positive']
        data_loader = SEEDLoader(filepath="../Dataset/SEED", label_path = "../Dataset/SEED/label.mat")
    elif args.dataset == 'deap_arousal':
        nb_classes, chans, samples = 2, 32, 1024
        label_names = ['Low Arousal', 'High Arousal']
        data_loader = DEAPLoader(filepath="../Dataset/DEAP", label_type='arousal')
    elif args.dataset == 'deap_valence':
        nb_classes, chans, samples = 2, 32, 1024
        label_names = ['Low Valence', 'High Valence']
        data_loader = DEAPLoader(filepath="../Dataset/DEAP", label_type='valence')
    elif args.dataset == 'stew':
        nb_classes, chans, samples = 2, 14, 512
        label_names = ['Rest', 'Task']
        data_loader = STEWLoader(filepath="../Dataset/STEW")
    # elif args.dataset == 'chbmit':
    #     nb_classes, chans, samples = 2, 14, 512
    #     label_names = ['Seizure', 'Non-seizure']
    #     data_loader = CHBMITLoader(filepath="../Dataset/CHBMIT")
    elif args.dataset == 'siena':
        nb_classes, chans, samples = 2, 29, 128
        label_names = ['Seizure', 'Non-seizure']    
        data_loader = SienaLoader(filepath = "../Dataset/SienaScalp")
    elif args.dataset == 'eegmat':
        nb_classes, chans, samples = 2, 21, 128
        label_names = ['Resting', 'With Task']    
        data_loader = EEGMATLoader(filepath = "../Dataset/EEGMAT")
    elif args.dataset == 'tuh_abnormal':
        nb_classes, chans, samples = 2, 21, 7680
        label_names = ['Normal', 'Abnormal']    
        data_loader = TUHAbnormalLoader(filepath = "../Dataset/TUHAbnormal")
    elif args.dataset == 'bciciii2':
        nb_classes, chans, samples = 2, 64, 85
        label_names = ['Non-P300', 'P300']    
        data_loader = BCICIII2Loader(filepath = "../Dataset/BCICIII2a")

    if os.path.exists(dataset_file):
        # Load the dataset if it already exists
        eeg_data = load_dataset_h5(dataset_file)
    else:
        # Load the dataset using the loader if the file doesn't exist
        eeg_data = data_loader.load_dataset()
        save_dataset_h5(eeg_data, dataset_file)

    # Iterate over each subject
    accuracies = []
    for subject, datasets in eeg_data.items():
        train_dataset = datasets.get('train_ds')
        test_dataset = datasets.get('test_ds')

        for x_batch, y_batch in train_dataset.take(1):
            print(f"Shape of input batch: {x_batch.shape}")

        if train_dataset is None or test_dataset is None:
            logger.warning(f"Missing datasets for subject {subject}. Skipping.")
            continue

        # Load model 
        model = eeg_models.load_model(args.model, nb_classes=nb_classes, nchan=chans, trial_length=samples)

        # Train and evaluate model for each subject
        accuracy = train_model(model, train_dataset, test_dataset, args.dataset, args.model, subject, label_names, epochs=args.epochs)
        accuracies.append(accuracy)
        logger.info(f"Subject {subject}: Accuracy = {accuracy}")

    # Print overall results
    avg_accuracy = np.mean(accuracies)
    print("Accuracies for all subjects:", accuracies)
    print("Average Accuracy:", avg_accuracy)
    logger.info(f"Average Accuracy across subjects: {avg_accuracy}")

    # Save the average accuracy to a file
    avg_accuracy_file = os.path.join('result', f'{args.dataset}_{args.model}_average_accuracy.txt')
    with open(avg_accuracy_file, 'w') as f:
        f.write(f"Accuracies for all subjects: {accuracies}\n")
        f.write(f'Average Accuracy: {avg_accuracy}\n')
    print(f"Average accuracy saved to {avg_accuracy_file}")

