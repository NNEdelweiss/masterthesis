# masterthesis

## Description
- The [EEG_Models.py](https://github.com/NNEdelweiss/masterthesis/blob/main/EEG_Models.py) script implements all selected models
- The [load_datasets.py](https://github.com/NNEdelweiss/masterthesis/blob/main/load_datasets.py) script loads and preprocesses all selected EEG-datasets
- The [main.py](https://github.com/NNEdelweiss/masterthesis/blob/main/main.py) script trains a list of deep learning models on a EEG dataset for classification tasks; allows specifying the dataset, the number of epochs, to train via command-line arguments
- The [dataset_config.json] file contains configuration of all dataset for training models
- The [main_each.py](https://github.com/NNEdelweiss/masterthesis/blob/main/main_each.py) script trains a deep learning model on a EEG dataset for classification tasks; allows specifying the dataset, the number of epochs, and the model to train via command-line arguments

### Usage
To run the script, use the following command:
```python main.py --dataset <dataset_name> --epochs <number_of_epochs>```
or 
```python main_each.py --dataset <dataset_name> --epochs <number_of_epochs> --models <model_name>```

For example:
```python main.py --dataset bciciv2a --models EEGNet --epochs 50```

## Selected Models
| No. | Model(with source)   | Used Datasets   | Notes   |  
|-----|------------|------------|------------|------------|
| 1   | [EEGNet]()     | P300 ERP, ERN, MRCP, SMR |
| 2   | [DeepConvNet]()| bciciv2a, highgamma |
| 3   | [ShallowConvNet]()| bciciv2a, highgamma |
| 4   | [ChronoNet]()  | TUH Abnormal |
| 5   | [CNN_FC]()     | Row 2 Col 3|
| 6   | [CRNN]()       | Row 3 Col 3|
| 7   | [DeepSleepNet]()| Sleep-EDF, Mass|
| 8   | [MMCNN_model]()| Row 2 Col 3|
| 9   | [EEGTCNet]()   | Row 3 Col 3|
| 10  | [ResNet]()     | Row 3 Col 3|
| 11  | [CNN_3D]()     | Row 3 Col 3| not use b/c not generalizable for all datasets

## Selected Datsets
| No. | Dataset   | Notes   | 
|------------|------------|------------|
| 1| BCICIV2a |        |
| 2| BCICIV2b |        |
| 3| DREAMER arousal |        |
| 4| DREAMER valence |        |
| 5| DEAP arousal |        |
| 6| DEAP valence |        |
| 7| SEED |        |
| 8| PhysioNetMI |        |
| 9| STEW |        |
| 10| CHBMIT |        |
| 11| Siena |        |
| 9| Sleep-EDF |        |
| 10| TUHAbnormal |        |
| 11| EEGMAT |        |
| 10| BCICIII2 |        |
| 11| SEED IV |        |
| 12| High Gamma |        |