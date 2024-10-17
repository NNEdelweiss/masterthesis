import argparse
import os
from datetime import datetime
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

# Declare global variables
metrics_dir = None
result_dir = None

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test).argmax(axis=1)
    y_true = y_test.argmax(axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_true, y_pred)
    classification_rep = classification_report(y_true, y_pred, output_dict=True)

    return accuracy, f1, recall, precision, conf_matrix, classification_rep

def train_model(model_name, X_train, y_train, X_test, y_test, dataset_name, subject, label_names, nb_classes, nchan, trial_length, epochs=20):
    global metrics_dir
    global result_dir

    # Initialize the model based on the type
    if model_name == 'DeepSleepNet':
        # Train the pre-trained model first
        pretrained_model = eeg_models.DSN_preTrainingNet(nchan=nchan, trial_length=trial_length, n_classes=nb_classes)
        pretrain_history = pretrained_model.fit(X_train, y_train, epochs=int(epochs / 2), batch_size = 64, verbose=1)

        # Plot training history for the pre-trained model
        plot_training_history(pretrain_history, dataset_name, model_name + "_pretrain", subject, epochs)

        # Create and train the fine-tuning model
        fine_tuned_model = eeg_models.DSN_fineTuningNet(nchan=nchan, trial_length=trial_length, n_classes=nb_classes, preTrainedNet=pretrained_model)
        finetune_history = fine_tuned_model.fit(X_train, y_train, epochs=int(epochs / 2), batch_size = 64, verbose=1)

        # Plot training history for the fine-tuned model
        plot_training_history(finetune_history, dataset_name, model_name + "_finetune", subject, epochs)

        # Use fine-tuned model for evaluation
        final_model = fine_tuned_model
    else:
        # For all other models, load and train
        model = eeg_models.load_model(model_name, nb_classes=nb_classes, nchan=nchan, trial_length=trial_length)
        history = model.fit(X_train, y_train, epochs=epochs, batch_size = 64, verbose=1)

        # Plot training history for the regular model
        plot_training_history(history, dataset_name, model_name, subject, epochs)

        final_model = model

    # Evaluate the model
    metrics = evaluate_model(final_model, X_test, y_test)
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

def main():
    global metrics_dir
    global result_dir

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bciciv2a', 
                        help='dataset used for the experiments')
    parser.add_argument('--model', type=str, default='EEGNet', 
                        help='model used for the experiments')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for training')  # Added epochs as an argument
    # parser.add_argument('--earlystopping', type=bool, default=False, help='Whether to use early stopping')  # Added early stopping as an argument
    args = parser.parse_args()

    # Define metrics_dir globally based on arguments
    subfolder = f'{args.dataset}_{args.model}'
    metrics_dir = os.path.join(os.getcwd(), 'metrics', args.dataset, subfolder)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Define result_dir globally
    result_dir = os.path.join(os.getcwd(), 'best_model', args.dataset, subfolder)
    os.makedirs(result_dir, exist_ok=True)

    # Setup logging and result directory
    current_time = datetime.now().strftime('%Y%m%d_%H%M')
    save_dir = f"{os.getcwd()}/log/"
    logger = get_logger(save_result=True, save_dir=save_dir, save_file=f"{current_time}_{args.dataset}.log")
    logger.info("Starting experiment")

    # Dataset storage path
    os.makedirs('loading_datasets', exist_ok=True)
    dataset_file = os.path.join('loading_datasets', f'{args.dataset}_data.h5')

    # Load dataset configuration from JSON file
    with open('dataset_config.json', 'r') as f:
        dataset_config = json.load(f)

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


    eeg_data = data_loader.load_dataset()


    # Iterate over each subject
    accuracies = []
    avg_accuracy_file = os.path.join(result_dir, f'{args.dataset}_{args.model}_accuracy.txt')

    for subject, datasets in eeg_data.items():
        X_train = datasets.get('X_train')
        y_train = datasets.get('y_train')
        X_test = datasets.get('X_test')
        y_test = datasets.get('y_test')

        # Train and evaluate model for each subject
        accuracy = train_model(args.model, X_train, y_train, X_test, y_test, args.dataset, subject, label_names, nb_classes, nchan, trial_length, epochs=args.epochs)
        accuracies.append(accuracy)
        with open(avg_accuracy_file, 'a') as f:
            f.write(f"Subject {subject}: Accuracy = {accuracy}\n")
        logger.info(f"Subject {subject}: Accuracy = {accuracy}")

    # Print overall results
    avg_accuracy = np.mean(accuracies)
    print("Accuracies for all subjects:", accuracies)
    print("Average Accuracy:", avg_accuracy)
    logger.info(f"Average Accuracy across subjects: {avg_accuracy}")

    # Save the average accuracy to a file
    with open(avg_accuracy_file, 'a') as f:
        f.write(f"Accuracies for all subjects: {accuracies}\n")
        f.write(f'Average Accuracy: {avg_accuracy}\n')
    print(f"Average accuracy saved to {avg_accuracy_file}")

if __name__ == '__main__':
    main()
