import re 
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential # type: ignore
from tensorflow.keras.layers import (   # type: ignore
    Dense, Activation, Permute, Dropout, Conv2D, MaxPooling2D, Conv3D, MaxPooling3D,
    AveragePooling2D, SeparableConv2D, DepthwiseConv2D, BatchNormalization, SeparableConv1D,
    SpatialDropout2D, Input, Flatten, Reshape, LSTM, TimeDistributed, 
    Bidirectional, RepeatVector, Conv1D, MaxPool1D, GlobalAveragePooling1D, MaxPooling1D,
    Add, multiply, Concatenate, Reshape, concatenate, GRU, Lambda, LeakyReLU, Multiply
) 
from tensorflow.keras.regularizers import l1_l2, l2 # type: ignore
from tensorflow.keras.constraints import max_norm # type: ignore
from tensorflow.keras import backend as K, utils as np_utils, regularizers # type: ignore
from tensorflow.keras.optimizers import Adam, RMSprop # type: ignore 
from sklearn import svm, neighbors, discriminant_analysis
from sklearn.metrics import accuracy_score

def load_model(model_name, nb_classes, nchan, trial_length, **kwargs):
    if model_name == 'EEGNet':
        return EEGNet(nb_classes, nchan, trial_length, **kwargs)
    # elif model_name == 'DeepConvNet_origin':
    #     return DeepConvNet_origin(nb_classes, nchan, trial_length, **kwargs)
    elif model_name == 'DeepConvNet':
        return DeepConvNet(nb_classes, nchan, trial_length, **kwargs)
    elif model_name == 'ShallowConvNet':
        return ShallowConvNet(nb_classes, nchan, trial_length, **kwargs)
    elif model_name == 'CNN_FC':
        return CNN_FC(nb_classes, nchan, trial_length, **kwargs)
    elif model_name == 'CRNN':
        return CRNN(nb_classes, nchan, trial_length, **kwargs)
    # elif model_name == 'DeepSleepNet':
    #     return DeepSleepNet(nb_classes, **kwargs)
    elif model_name == 'MMCNN':
        return MMCNN(nb_classes, nchan, trial_length, **kwargs)
    elif model_name == 'ChronoNet':
        return ChronoNet(nb_classes, nchan, trial_length, **kwargs)
    elif model_name == 'EEGTCNet':
        return EEGTCNet(nb_classes, nchan, trial_length, layers=2, kernel_s=4,filt=12, dropout=0.3, activation='elu', F1=8, D=2, kernLength=32, dropout_eeg=0.2)
    elif model_name == 'ResNet':
        return ResNet(nb_classes, nchan, trial_length, **kwargs)
    # elif model_name == 'CNN3D': 
    #     return CNN_3D(nb_classes, nchan, trial_length, **kwargs)
    elif model_name == 'Attention_1DCNN':
        return Attention_1DCNN(nb_classes, nchan, trial_length, **kwargs)
    elif model_name == 'BLSTM_LSTM':
        return BLSTM_LSTM(nb_classes, nchan, trial_length, **kwargs)
    elif model_name == 'OneD_CNN_LSTM':
        return OneD_CNN_LSTM(nb_classes, nchan, trial_length, **kwargs) 
    else:
        raise ValueError(f"Model '{model_name}' is not recognized. Available models: 'EEGNet', 'DeepConvNet'.")


def SVM(train_features, train_labels, test_features, test_labels):
    # Train an SVM
    svm_model = svm.SVC(kernel='linear')
    svm_model.fit(train_features, train_labels)

    test_predictions = svm_model.predict(test_features)

    accuracy = accuracy_score(test_labels, test_predictions)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return svm_model, accuracy

def KNN(train_features, train_labels, test_features, test_labels):
    # Train a KNN classifier
    knn_model = neighbors.KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(train_features, train_labels)

    test_predictions = knn_model.predict(test_features)

    accuracy = accuracy_score(test_labels, test_predictions)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return knn_model, accuracy

def LDA(train_features, train_labels, test_features, test_labels):
    # Train an LDA classifier
    lda_model = discriminant_analysis.LinearDiscriminantAnalysis()
    lda_model.fit(train_features, train_labels)

    test_predictions = lda_model.predict(test_features)

    accuracy = accuracy_score(test_labels, test_predictions)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return lda_model, accuracy


def EEGNet(nb_classes, nchan = 22, trial_length = 128, 
             dropoutRate = 0.5, kernLength = 22, F1 = 8, 
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout'):
    
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input_main   = Input(shape = (nchan, trial_length))
    expand   = Reshape((nchan, trial_length, 1))(input_main)

    ##################################################################
    block1       = Conv2D(F1, (1, kernLength), padding = 'same',
                                   input_shape = (nchan, trial_length, 1),
                                   use_bias = False)(expand)
    block1       = BatchNormalization()(block1)
    block1       = DepthwiseConv2D((nchan, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1       = BatchNormalization()(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((1, 4))(block1)
    block1       = dropoutType(dropoutRate)(block1)
    
    block2       = SeparableConv2D(F2, (1, 16),
                                   use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization()(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D((1, 8))(block2)
    block2       = dropoutType(dropoutRate)(block2)
        
    flatten      = Flatten(name = 'flatten')(block2)
    
    dense        = Dense(nb_classes, name = 'dense', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    softmax      = Activation('softmax', name = 'softmax')(dense)

    model        = Model(inputs=input_main, outputs=softmax)

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', 
                    metrics = ['accuracy'])
    
    return model

def DeepConvNet(nb_classes, nchan = 64, trial_length = 256,
                dropoutRate = 0.5):
    """ Keras implementation of the Deep Convolutional Network as described in
    Schirrmeister et. al. (2017), Human Brain Mapping.
    
    This implementation assumes the input is a 2-second EEG signal sampled at 
    128Hz, as opposed to signals sampled at 250Hz as described in the original
    paper. We also perform temporal convolutions of length (1, 5) as opposed
    to (1, 10) due to this sampling rate difference. 
    
    Note that we use the max_norm constraint on all convolutional layers, as 
    well as the classification layer. We also change the defaults for the
    BatchNormalization layer. We used this based on a personal communication 
    with the original authors.
    
                      ours        original paper
    pool_size        1, 2        1, 3
    strides          1, 2        1, 3
    conv filters     1, 5        1, 10
    
    Note that this implementation has not been verified by the original 
    authors. 
    
    """

    # start the model
    input_main   = Input(shape = (nchan, trial_length))
    expand       = Reshape((nchan, trial_length, 1))(input_main)
    block1       = Conv2D(25, (1, 5), 
                                 input_shape=(nchan, trial_length, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(expand)
    block1       = Conv2D(25, (nchan, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block1       = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
    block1       = Activation('elu')(block1)
    block1       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block1)
    block1       = Dropout(dropoutRate)(block1)
  
    block2       = Conv2D(50, (1, 5),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block2       = BatchNormalization(epsilon=1e-05, momentum=0.9)(block2)
    block2       = Activation('elu')(block2)
    block2       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block2)
    block2       = Dropout(dropoutRate)(block2)
    
    block3       = Conv2D(100, (1, 5),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block2)
    block3       = BatchNormalization(epsilon=1e-05, momentum=0.9)(block3)
    block3       = Activation('elu')(block3)
    block3       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block3)
    block3       = Dropout(dropoutRate)(block3)
    
    block4       = Conv2D(200, (1, 5),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block3)
    block4       = BatchNormalization(epsilon=1e-05, momentum=0.9)(block4)
    block4       = Activation('elu')(block4)
    block4       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block4)
    block4       = Dropout(dropoutRate)(block4)
    
    flatten      = Flatten()(block4)
    
    dense        = Dense(nb_classes, kernel_constraint = max_norm(0.5))(flatten)
    softmax      = Activation('softmax')(dense)
    model        = Model(inputs=input_main, outputs=softmax)

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', 
                    metrics = ['accuracy'])
    
    return model


# need these for ShallowConvNet
def square(x):
    return K.square(x)

def log(x):
    return K.log(K.clip(x, min_value = 1e-7, max_value = 10000))   

def ShallowConvNet(nb_classes, nchan = 64, trial_length = 128, dropoutRate = 0.5):
    """ Keras implementation of the Shallow Convolutional Network as described
    in Schirrmeister et. al. (2017), Human Brain Mapping.
    
    Assumes the input is a 2-second EEG signal sampled at 128Hz. Note that in 
    the original paper, they do temporal convolutions of length 25 for EEG
    data sampled at 250Hz. We instead use length 13 since the sampling rate is 
    roughly half of the 250Hz which the paper used. The pool_size and stride
    in later layers is also approximately half of what is used in the paper.
    
    Note that we use the max_norm constraint on all convolutional layers, as 
    well as the classification layer. We also change the defaults for the
    BatchNormalization layer. We used this based on a personal communication 
    with the original authors.
    
                     ours        original paper
    pool_size        1, 35       1, 75
    strides          1, 7        1, 15
    conv filters     1, 13       1, 25    
    
    Note that this implementation has not been verified by the original 
    authors. We do note that this implementation reproduces the results in the
    original paper with minor deviations. 
    """

    # start the model
    input_main   = Input(shape = (nchan, trial_length))
    expand       = Reshape((nchan, trial_length, 1))(input_main)
    block1       = Conv2D(40, (1, 13), 
                                 input_shape=(nchan, trial_length, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(expand)
    block1       = Conv2D(40, (nchan, 1), use_bias=False, 
                          kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block1       = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
    block1       = Activation(square)(block1)
    block1       = AveragePooling2D(pool_size=(1, 35), strides=(1, 7))(block1)
    block1       = Activation(log)(block1)
    block1       = Dropout(dropoutRate)(block1)
    flatten      = Flatten()(block1)
    dense        = Dense(nb_classes, kernel_constraint = max_norm(0.5))(flatten)
    softmax      = Activation('softmax')(dense)
    model        = Model(inputs=input_main, outputs=softmax)

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', 
                    metrics = ['accuracy'])
    
    return model


def CNN_FC(nclasses, nchan, trial_length=960, l1=0):
    input_shape = (nchan, trial_length)
    model = Sequential()
    model.add(Permute((2, 1), input_shape=input_shape))
    model.add(Reshape((trial_length, nchan, 1)))
    model.add(Conv2D(40, (30, 1), activation="relu", kernel_regularizer=regularizers.l1(l1), padding="same", input_shape=input_shape))
    model.add(Conv2D(40, (1, nchan), activation="relu", kernel_regularizer=regularizers.l1(l1), padding="valid"))
    model.add(AveragePooling2D((30, 1), strides=(15, 1)))
    model.add(Flatten())
    model.add(Dense(80, activation="relu"))
    model.add(Dense(nclasses, activation="softmax"))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

def CRNN(nclasses, nchan, trial_length=128, l1=0, full_output=False):
    """
    CRNN model definition
    """
    input_shape = (nchan, trial_length)
    model = Sequential()
    model.add(Permute((2, 1), input_shape=input_shape))
    model.add(Reshape((trial_length, nchan, 1)))
    model.add(Conv2D(40, (30, 1), activation="relu", kernel_regularizer=regularizers.l1(l1), padding="same", input_shape=input_shape))
    model.add(Conv2D(40, (1, nchan), activation="relu", kernel_regularizer=regularizers.l1(l1), padding="valid"))
    model.add(AveragePooling2D((5, 1), strides=(5, 1)))
    model.add(TimeDistributed(Flatten()))
    # Use default activation and remove dropout to enable cuDNN kernels
    model.add(LSTM(40, return_sequences=full_output))  # Remove activation and dropout for cuDNN compatibility
    model.add(Dense(nclasses, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    return model

def MMCNN(n_classes, nchan, trial_length, activation='elu', learning_rate=0.0001, dropout=0.8, inception_filters=[16, 16, 16, 16],
                inception_kernel_length=[[5, 10, 15, 10], [40, 45, 50, 100], [60, 65, 70, 100], [80, 85, 90, 100], [160, 180, 200, 180]],
                inception_stride=[2, 4, 4, 4, 16], first_maxpooling_size=4, first_maxpooling_stride=4,
                res_block_filters=[16, 16, 16], res_block_kernel_stride=[8, 7, 7, 7, 6], se_block_kernel_stride=16,
                se_ratio=8, second_maxpooling_size=[4, 3, 3, 3, 2], second_maxpooling_stride=[4, 3, 3, 3, 2]):
    
    def squeeze_excitation_layer(x, out_dim, activation, ratio=8):
        squeeze = GlobalAveragePooling1D()(x)
        excitation = Dense(units=out_dim // ratio)(squeeze)
        excitation = Activation(activation)(excitation)
        excitation = Dense(units=out_dim, activation='sigmoid')(excitation)
        excitation = Reshape((1, out_dim))(excitation)
        scale = multiply([x, excitation])
        return scale

    def conv_block(x, nb_filter, length, activation):
        k1, k2, k3 = nb_filter
        out = Conv1D(k1, length, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.002))(x)
        out = BatchNormalization()(out)
        out = Activation(activation)(out)
        out = Conv1D(k2, length, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.002))(out)
        out = BatchNormalization()(out)
        out = Activation(activation)(out)
        out = Conv1D(k3, length, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.002))(out)
        out = BatchNormalization()(out)
        x = Conv1D(k3, 1, strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        out = Add()([out, x])
        out = Activation(activation)(out)
        out = Dropout(dropout)(out)
        return out

    def inception_block(x, ince_filter, ince_length, stride, activation):
        k1, k2, k3, k4 = ince_filter
        l1, l2, l3, l4 = ince_length
        inception = []
        x1 = Conv1D(k1, l1, strides=stride, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
        x1 = BatchNormalization()(x1)
        x1 = Activation(activation)(x1)
        inception.append(x1)
        x2 = Conv1D(k2, l2, strides=stride, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
        x2 = BatchNormalization()(x2)
        x2 = Activation(activation)(x2)
        inception.append(x2)
        x3 = Conv1D(k3, l3, strides=stride, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
        x3 = BatchNormalization()(x3)
        x3 = Activation(activation)(x3)
        inception.append(x3)
        x4 = MaxPooling1D(pool_size=l4, strides=stride, padding='same')(x)
        x4 = Conv1D(k4, 1, strides=1, padding='same')(x4)
        x4 = BatchNormalization()(x4)
        x4 = Activation(activation)(x4)
        inception.append(x4)
        v1 = Concatenate(axis=-1)(inception)
        return v1

    output_conns = []
    origin_input = Input(shape =(nchan, trial_length))
    input_tensor = Permute((2, 1))(origin_input)

    # EIN-a
    x = inception_block(input_tensor, inception_filters, inception_kernel_length[0], inception_stride[0], activation)
    x = MaxPooling1D(pool_size=first_maxpooling_size, strides=first_maxpooling_stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    x = conv_block(x, res_block_filters, res_block_kernel_stride[0], activation)
    x = squeeze_excitation_layer(x, se_block_kernel_stride, activation, ratio=se_ratio)
    x = MaxPooling1D(pool_size=second_maxpooling_size[0], strides=second_maxpooling_stride[0], padding='same')(x)
    x = Flatten()(x)
    output_conns.append(x)

    # EIN-b
    y1 = inception_block(input_tensor, inception_filters, inception_kernel_length[1], inception_stride[1], activation)
    y1 = MaxPooling1D(pool_size=first_maxpooling_size, strides=first_maxpooling_stride, padding='same')(y1)
    y1 = BatchNormalization()(y1)
    y1 = Dropout(dropout)(y1)
    y1 = conv_block(y1, res_block_filters, res_block_kernel_stride[1], activation)
    y1 = squeeze_excitation_layer(y1, se_block_kernel_stride, activation, ratio=se_ratio)
    y1 = MaxPooling1D(pool_size=second_maxpooling_size[1], strides=second_maxpooling_stride[1], padding='same')(y1)
    y1 = Flatten()(y1)
    output_conns.append(y1)

    # EIN-c
    y2 = inception_block(input_tensor, inception_filters, inception_kernel_length[2], inception_stride[2], activation)
    y2 = MaxPooling1D(pool_size=first_maxpooling_size, strides=first_maxpooling_stride, padding='same')(y2)
    y2 = BatchNormalization()(y2)
    y2 = Dropout(dropout)(y2)
    y2 = conv_block(y2, res_block_filters, res_block_kernel_stride[2], activation)
    y2 = squeeze_excitation_layer(y2, se_block_kernel_stride, activation, ratio=se_ratio)
    y2 = MaxPooling1D(pool_size=second_maxpooling_size[2], strides=second_maxpooling_stride[2], padding='same')(y2)
    y2 = Flatten()(y2)
    output_conns.append(y2)

    # EIN-d
    y3 = inception_block(input_tensor, inception_filters, inception_kernel_length[3], inception_stride[3], activation)
    y3 = MaxPooling1D(pool_size=first_maxpooling_size, strides=first_maxpooling_stride, padding='same')(y3)
    y3 = BatchNormalization()(y3)
    y3 = Dropout(dropout)(y3)
    y3 = conv_block(y3, res_block_filters, res_block_kernel_stride[3], activation)
    y3 = squeeze_excitation_layer(y3, se_block_kernel_stride, activation, ratio=se_ratio)
    y3 = MaxPooling1D(pool_size=second_maxpooling_size[3], strides=second_maxpooling_stride[3], padding='same')(y3)
    y3 = Flatten()(y3)
    output_conns.append(y3)

    # EIN-e
    z = inception_block(input_tensor, inception_filters, inception_kernel_length[4], inception_stride[4], activation)
    z = MaxPooling1D(pool_size=first_maxpooling_size, strides=first_maxpooling_stride, padding='same')(z)
    z = BatchNormalization()(z)
    z = Dropout(dropout)(z)
    z = conv_block(z, res_block_filters, res_block_kernel_stride[4], activation)
    z = squeeze_excitation_layer(z, se_block_kernel_stride, activation, ratio=se_ratio)
    z = MaxPooling1D(pool_size=second_maxpooling_size[4], strides=second_maxpooling_stride[4], padding='same')(z)
    z = Flatten()(z)
    output_conns.append(z)

    output_conns = Concatenate(axis=-1)(output_conns)
    output_conns = Dropout(dropout)(output_conns)
    output_tensor = Dense(n_classes, activation='sigmoid')(output_conns)
    model = Model(origin_input, output_tensor)
    
    adam = Adam(learning_rate=learning_rate)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def ChronoNet(n_classes, nchan=22, trial_length=256):
    inputsin= Input(shape=(nchan,trial_length))
    # First Inception
    tower1 = Conv1D(32, 2, strides=2,activation='relu',padding="causal")(inputsin)
    tower1 = BatchNormalization()(tower1)
    tower2 = Conv1D(32, 4, strides=2,activation='relu',padding="causal")(inputsin)
    tower2 = BatchNormalization()(tower2)
    tower3 = Conv1D(32, 8, strides=2,activation='relu',padding="causal")(inputsin)
    tower3 = BatchNormalization()(tower3)
    x = concatenate([tower1,tower2,tower3],axis=2)
    x = Dropout(0.45)(x)

    # Second Inception
    tower1 = Conv1D(32, 2, strides=2,activation='relu',padding="causal")(x)
    tower1 = BatchNormalization()(tower1)
    tower2 = Conv1D(32, 4, strides=2,activation='relu',padding="causal")(x)
    tower2 = BatchNormalization()(tower2)
    tower3 = Conv1D(32, 8, strides=2,activation='relu',padding="causal")(x)
    tower3 = BatchNormalization()(tower3)
    x = concatenate([tower1,tower2,tower3],axis=2)
    x = Dropout(0.45)(x)

    # Third Inception
    tower1 = Conv1D(32, 2, strides=2,activation='relu',padding="causal")(x)
    tower1 = BatchNormalization()(tower1)
    tower2 = Conv1D(32, 4, strides=2,activation='relu',padding="causal")(x)
    tower2 = BatchNormalization()(tower2)
    tower3 = Conv1D(32, 8, strides=2,activation='relu',padding="causal")(x)
    tower3 = BatchNormalization()(tower3)
    x = concatenate([tower1,tower2,tower3],axis=2)
    x = Dropout(0.55)(x)

    # GRU layers
    res1 = GRU(32,activation='tanh',return_sequences=True)(x)
    res2 = GRU(32,activation='tanh',return_sequences=True)(res1)
    res1_2 = concatenate([res1,res2],axis=2)
    res3 = GRU(32,activation='tanh',return_sequences=True)(res1_2)
    x = concatenate([res1,res2,res3])
    x = GRU(32,activation='tanh')(x)

    # Output layer
    predictions = Dense(n_classes,activation='softmax')(x)
    model = Model(inputs=inputsin, outputs=predictions)

    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

    return model

def EEGTCNet(nb_classes, nchan=64, trial_length=128, layers=3, kernel_s=10,filt=10, dropout=0, activation='relu', F1=4, D=2, kernLength=64, dropout_eeg=0.1):
    input_main   = Input(shape = (nchan, trial_length))
    input1 = Reshape((1, nchan, trial_length))(input_main)
    input2 = Permute((3,2,1))(input1)
    regRate=.25
    numFilters = F1
    F2= numFilters*D

    EEGNet_sep = EEGNet_sub(input_layer=input2,F1=F1,kernLength=kernLength,D=D,Chans=nchan,dropout=dropout_eeg)
    block2 = Lambda(lambda x: x[:,:,-1,:])(EEGNet_sep)
    outs = TCN_block(input_layer=block2,input_dimension=F2,depth=layers,kernel_size=kernel_s,filters=filt,dropout=dropout,activation=activation)
    out = Lambda(lambda x: x[:,-1,:])(outs)
    dense        = Dense(nb_classes, name = 'dense',kernel_constraint = max_norm(regRate))(out)
    softmax      = Activation('softmax', name = 'softmax')(dense)

    model = Model(inputs=input_main, outputs=softmax)
    opt = Adam(learning_rate=0.001)
    model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics=['accuracy'])

    return model

def EEGNet_sub(input_layer,F1=4,kernLength=64,D=2,Chans=22,dropout=0.1):
    F2= F1*D
    block1 = Conv2D(F1, (kernLength, 1), padding = 'same',data_format='channels_last',use_bias = False)(input_layer)
    block1 = BatchNormalization(axis = -1)(block1)
    block2 = DepthwiseConv2D((1, Chans), use_bias = False, 
                                    depth_multiplier = D,
                                    data_format='channels_last',
                                    depthwise_constraint = max_norm(1.))(block1)
    block2 = BatchNormalization(axis = -1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((8,1),data_format='channels_last')(block2)
    block2 = Dropout(dropout)(block2)
    block3 = SeparableConv2D(F2, (16, 1),
                            data_format='channels_last',
                            use_bias = False, padding = 'same')(block2)
    block3 = BatchNormalization(axis = -1)(block3)
    block3 = Activation('elu')(block3)
    block3 = AveragePooling2D((8,1),data_format='channels_last')(block3)
    block3 = Dropout(dropout)(block3)
    return block3

def TCN_block(input_layer,input_dimension,depth,kernel_size,filters,dropout,activation='relu'):
    """ TCN_block from Bai et al 2018
        Temporal Convolutional Network (TCN)
        
        Notes
        -----
        THe original code available at https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
        This implementation has a slight modification from the original code
        and it is taken from the code by Ingolfsson et al at https://github.com/iis-eth-zurich/eeg-tcnet
        See details at https://arxiv.org/abs/2006.00622

        References
        ----------
        .. Bai, S., Kolter, J. Z., & Koltun, V. (2018).
           An empirical evaluation of generic convolutional and recurrent networks
           for sequence modeling.
           arXiv preprint arXiv:1803.01271.
    """    
    block = Conv1D(filters,kernel_size=kernel_size,dilation_rate=1,activation='linear',
                   padding = 'causal',kernel_initializer='he_uniform')(input_layer)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    block = Conv1D(filters,kernel_size=kernel_size,dilation_rate=1,activation='linear',
                   padding = 'causal',kernel_initializer='he_uniform')(block)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    if(input_dimension != filters):
        conv = Conv1D(filters,kernel_size=1,padding='same')(input_layer)
        added = Add()([block,conv])
    else:
        added = Add()([block,input_layer])
    out = Activation(activation)(added)
    
    for i in range(depth-1):
        block = Conv1D(filters,kernel_size=kernel_size,dilation_rate=2**(i+1),activation='linear',
                   padding = 'causal',kernel_initializer='he_uniform')(out)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        block = Conv1D(filters,kernel_size=kernel_size,dilation_rate=2**(i+1),activation='linear',
                   padding = 'causal',kernel_initializer='he_uniform')(block)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        added = Add()([block, out])
        out = Activation(activation)(added)
        
    return out

def ResNet(nclasses, nchan, trial_length, keep_prob=0.5):
    original_input   = Input(shape=(nchan, trial_length))
    input_permuted   = Permute((2, 1))(original_input)
    inputs           = Reshape((nchan, trial_length, 1))(input_permuted)

    # First Residual Block
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(keep_prob)(x)

    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.add([x, inputs])
    x = LeakyReLU()(x)
    x = Dropout(keep_prob)(x)

    x = MaxPooling2D((2, 2), padding='same')(x)

    # Second Residual Block
    shortcut = Conv2D(64, (1, 1), padding='same')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(keep_prob)(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.add([x, shortcut])
    x = LeakyReLU()(x)
    x = Dropout(keep_prob)(x)

    x = MaxPooling2D((2, 2), padding='same')(x)

    # Third Residual Block
    shortcut = Conv2D(128, (1, 1), padding='same')(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(keep_prob)(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.add([x, shortcut])
    x = LeakyReLU()(x)
    x = Dropout(keep_prob)(x)

    x = MaxPooling2D((2, 2), padding='same')(x)

    # Fully Connected Layers
    x = Flatten()(x)
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(keep_prob)(x)

    outputs = Dense(nclasses, activation='softmax')(x)

    model = Model(original_input, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def CNN_3D(nchan, nclasses, trial_length, w=32, m=8):
    # Ch: Number of channels
    # D: Total number of samples in a trial
    # w: Window size
    # m: Chunk size (number of consecutive frames)
    
    # Calculate the number of frames per channel
    num_frames = trial_length // w
    
    # Adjusted input shape to include chunk size
    input_shape = (nchan, trial_length)  # Considering 'm' frames, 'Ch' channels, 'w' window size, and 1 for single color channel
    
    model = Sequential()
    
    # Input layer with the adjusted shape
    model.add(Input(shape=input_shape))
    model.add(Reshape(target_shape=(m, nchan, w, 1)))
    
    # 3D Convolutional layers
    model.add(Conv3D(filters=8, kernel_size=(3, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    
    model.add(Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu'))
    # Adjust the pool size or strides if necessary to match the input dimensions
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))
    
    # Flatten and Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(600, activation='relu'))
    model.add(Dense(nclasses, activation='softmax'))
    
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    
    return model

# Attention Block Function
def attention_block(inputs, num_filters):
    # Parallel Branch with Average Pooling
    avg_pool = GlobalAveragePooling1D()(inputs)
    
    # Dense layers for computing attention weights
    dense_1 = Dense(units=100, activation='sigmoid')(avg_pool)
    dense_2 = Dense(units=num_filters, activation='sigmoid')(dense_1)
    
    # Repeat and Reshape to match the input dimensions
    repeat = RepeatVector(inputs.shape[1])(dense_2)
    reshape = Reshape((inputs.shape[1], num_filters))(repeat)
    
    # Multiply attention weights with the input
    attention_output = Multiply()([inputs, reshape])
    
    return attention_output

# Create Attention-based 1D-CNN Model with Ensemble Majority Voting
def Attention_1DCNN(nclasses, nchan, trial_length):
    original_input   = Input(shape=(nchan, trial_length))
    inputs   = Permute((2, 1))(original_input)
    
    # Four 1D-CNN Blocks with Alternating Time and Channel Convolution
    for i in range(4):
        # 1D-CNN Layer (alternating on time and channel axis)
        if i % 2 == 0:
            x = Conv1D(filters=64, kernel_size=4, padding='same')(inputs)
        else:
            x = Conv1D(filters=50, kernel_size=4, padding='same')(x)
        
        # LeakyReLU Activation
        x = LeakyReLU(alpha=0.001)(x)
        
        # Attention Block
        x = attention_block(x, num_filters=x.shape[-1])
    
    # Fully Connected Layers
    x = Flatten()(x)
    
    x = Dense(2048, kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-07, l2=1e-05))(x)
    x = LeakyReLU(alpha=0.001)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Dense(1024, kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-07, l2=1e-05))(x)
    x = LeakyReLU(alpha=0.001)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Dense(512, kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-07, l2=1e-05))(x)
    x = LeakyReLU(alpha=0.001)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(128, kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-07, l2=1e-05))(x)
    x = LeakyReLU(alpha=0.001)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(32, kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-07, l2=1e-05))(x)
    x = LeakyReLU(alpha=0.001)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Output Softmax Layer
    outputs = Dense(nclasses, activation='softmax')(x)  # Assuming binary classification

    # Create Model
    model = Model(inputs=original_input, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.09, beta_2=0.0999, epsilon=1e-07)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# # Create Attention-based 1D-CNN Model with Ensemble Majority Voting
# def Attention_1DCNN(nclasses, nchan, trial_length):
#     original_input   = Input(shape=(nchan, trial_length))
#     inputs   = Permute((2, 1))(original_input)
    
#     # Three 1D-CNN Blocks with Reduced Filters
#     for i in range(3):  # Reduced number of blocks
#         if i % 2 == 0:
#             x = SeparableConv1D(filters=32, kernel_size=4, padding='same')(inputs)
#         else:
#             x = SeparableConv1D(filters=25, kernel_size=4, padding='same')(x)
#         x = LeakyReLU(alpha=0.001)(x)
#         x = attention_block(x, num_filters=x.shape[-1])
    
#     # Global Average Pooling
#     x = GlobalAveragePooling1D()(x)  # Replace Flatten with GAP
    
#     x = Dense(2048, kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-07, l2=1e-05))(x)
#     x = LeakyReLU(alpha=0.001)(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.4)(x)

#     x = Dense(1024, kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-07, l2=1e-05))(x)
#     x = LeakyReLU(alpha=0.001)(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.4)(x)

#     x = Dense(512, kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-07, l2=1e-05))(x)
#     x = LeakyReLU(alpha=0.001)(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.3)(x)

#     x = Dense(128, kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-07, l2=1e-05))(x)
#     x = LeakyReLU(alpha=0.001)(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.3)(x)

#     x = Dense(32, kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-07, l2=1e-05))(x)
#     x = LeakyReLU(alpha=0.001)(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.3)(x)
    
#     # Output Softmax Layer
#     outputs = Dense(nclasses, activation='softmax')(x)  # Assuming binary classification

#     # Create Model
#     model = Model(inputs=original_input, outputs=outputs)

#     optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.09, beta_2=0.0999, epsilon=1e-07)
#     model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
#     return model


# Ensemble Majority Voting
def majority_voting_ensemble(models, X):
    predictions = [model.predict(X) for model in models]
    avg_prediction = tf.reduce_mean(predictions, axis=0)
    final_prediction = tf.argmax(avg_prediction, axis=-1)
    return final_prediction

def DSN_ConvLayers(inputLayer, sfreq = 128):
    # two conv-nets in parallel for feature learning, 
    # one with fine resolution another with coarse resolution    
    # network to learn fine features
    convFine = Conv1D(filters=64, kernel_size=int(sfreq/2), strides=int(sfreq/16), padding='same', activation='relu', name='fConv1')(inputLayer)
    convFine = MaxPool1D(pool_size=8, strides=8, name='fMaxP1')(convFine)
    convFine = Dropout(rate=0.5, name='fDrop1')(convFine)
    convFine = Conv1D(filters=128, kernel_size=8, padding='same', activation='relu', name='fConv2')(convFine)
    convFine = Conv1D(filters=128, kernel_size=8, padding='same', activation='relu', name='fConv3')(convFine)
    convFine = Conv1D(filters=128, kernel_size=8, padding='same', activation='relu', name='fConv4')(convFine)
    convFine = MaxPool1D(pool_size=4, strides=4, name='fMaxP2')(convFine)
    fineShape = convFine.get_shape()
    convFine = Flatten(name='fFlat1')(convFine)
    
    # network to learn coarse features
    convCoarse = Conv1D(filters=32, kernel_size=sfreq*4, strides=int(sfreq/2), padding='same', activation='relu', name='cConv1')(inputLayer)
    convCoarse = MaxPool1D(pool_size=4, strides=4, name='cMaxP1')(convCoarse)
    convCoarse = Dropout(rate=0.5, name='cDrop1')(convCoarse)
    convCoarse = Conv1D(filters=128, kernel_size=6, padding='same', activation='relu', name='cConv2')(convCoarse)
    convCoarse = Conv1D(filters=128, kernel_size=6, padding='same', activation='relu', name='cConv3')(convCoarse)
    convCoarse = Conv1D(filters=128, kernel_size=6, padding='same', activation='relu', name='cConv4')(convCoarse)
    convCoarse = MaxPool1D(pool_size=2, strides=2, name='cMaxP2')(convCoarse)
    coarseShape = convCoarse.get_shape()
    convCoarse = Flatten(name='cFlat1')(convCoarse)
    
    # concatenate coarse and fine cnns
    mergeLayer = concatenate([convFine, convCoarse], name='merge')
    
    return mergeLayer, (coarseShape, fineShape)

def DSN_preTrainingNet(nchan, trial_length, n_classes, sfreq = 128):
    input = Input(shape=(nchan, trial_length), name='inputLayer')
    input_reshape = Reshape((-1, 1))(input)

    mLayer, (_, _) = DSN_ConvLayers(input_reshape) 
    outLayer = Dense(n_classes, activation='softmax', name='outLayer')(mLayer)
    
    network = Model(inputs=input, outputs=outLayer)
    network.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return network 

def DSN_fineTuningNet(nchan, trial_length, n_classes, preTrainedNet, sfreq = 128):
    input_layer = Input(shape=(nchan, trial_length), name='inputLayer')
    input_reshape = Reshape((nchan * trial_length, 1))(input_layer)  # Reshape to (nchan * trial_length, 1)

    mLayer, (cShape, fShape) = DSN_ConvLayers(input_reshape)
    outLayer = Dropout(rate=0.5, name='mDrop1')(mLayer)

    # LSTM layers
    outLayer = Reshape((1, int(fShape[1] * fShape[2] + cShape[1] * cShape[2])))(outLayer)
    outLayer = Bidirectional(LSTM(256, activation='tanh', recurrent_activation='sigmoid', dropout=0.5, name='bLstm1', return_sequences=True))(outLayer)
    
    # Adjust reshape based on actual shape
    shape_after_lstm = K.int_shape(outLayer)
    if len(shape_after_lstm) == 3:
        outLayer = Reshape((1, int(shape_after_lstm[2])))(outLayer)
    else:
        raise ValueError("Expected output from LSTM to have 3 dimensions.")

    outLayer = Bidirectional(LSTM(256, activation='tanh', recurrent_activation='sigmoid', dropout=0.5, name='bLstm2'))(outLayer)
    outLayer = Dense(n_classes, activation='softmax', name='outLayer')(outLayer)

    network = Model(inputs=input_layer, outputs=outLayer)

    # Setting weights from pre-trained model
    allPreTrainLayers = dict([(layer.name, layer) for layer in preTrainedNet.layers])
    allFineTuneLayers = dict([(layer.name, layer) for layer in network.layers])

    allPreTrainLayerNames = [layer.name for layer in preTrainedNet.layers]
    # Define your unwanted layer names and patterns
    exclude_patterns = ['inputLayer', 'outLayer', r'reshape/s+']

    # Filter allPreTrainLayerNames based on whether each layer name matches any exclude pattern
    allPreTrainLayerNames = [
        l for l in allPreTrainLayerNames 
        if not any(re.search(pattern, l) for pattern in exclude_patterns)
    ]

    print("Pre-trained layer names:", allPreTrainLayerNames)
    print("Fine-tune layer names:", allFineTuneLayers.keys())

    # Now set weights of fine-tune network based on pre-trained network
    for l in allPreTrainLayerNames:
        if l in allFineTuneLayers:
            allFineTuneLayers[l].set_weights(allPreTrainLayers[l].get_weights())
        else:
            print(f"Warning: Layer {l} not found in fine-tune model.")

    network.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return network

def BLSTM_LSTM(nclasses, nchan, trial_length, keep_prob=0.5, reg=0.01):
    """
    Creates the BLSTM-LSTM guided classifier model with L2 regularization and dropout.

    Parameters:
    nclasses (int): Number of output classes.
    nchan (int): Number of channels (features).
    trial_length (int): Length of each trial (time steps).
    keep_prob (float): Dropout rate.
    reg (float): L2 regularization factor.

    Returns:
    model (tf.keras.Model): Compiled TensorFlow model.
    """
    inputs = Input(shape=(nchan, trial_length))
    
    x = Bidirectional(LSTM(256, return_sequences=True, kernel_regularizer=l2(reg)))(inputs)
    x = Dropout(keep_prob)(x)
    x = BatchNormalization()(x)
    
    x = LSTM(128, return_sequences=True, kernel_regularizer=l2(reg))(x)
    x = Dropout(keep_prob)(x)
    x = BatchNormalization()(x)
    
    x = LSTM(64, return_sequences=False, kernel_regularizer=l2(reg))(x)
    x = Dropout(keep_prob)(x)
    x = BatchNormalization()(x)
    
    x = Dense(32, activation='relu', kernel_regularizer=l2(reg))(x)
    x = Dropout(keep_prob)(x)
    outputs = Dense(nclasses, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    # Compile the model with learning rate scheduling
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=10000,
        decay_rate=0.9)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def OneD_CNN_LSTM(nb_classes, nchan = 22, trial_length = 128):
    """
    1D CNN-LSTM model for epileptic seizure recognition with input shape (nchan, trial_length).
    :param nchan: Number of EEG channels
    :param trial_length: Length of each trial
    :param nb_classes: Number of output classes
    :return: Compiled Keras model
    """
    input_shape = (nchan, trial_length)
    inputs = Input(shape=input_shape)

    # Permute to match the expected input for 1D CNN (time_steps, features)
    x = Permute((2, 1))(inputs)

    # First Convolutional Block
    x = Conv1D(filters=64, kernel_size=3, strides=1, activation='relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)

    # Second Convolutional Block
    x = Conv1D(filters=128, kernel_size=3, strides=1, activation='relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)

    # Third Convolutional Block
    x = Conv1D(filters=512, kernel_size=3, strides=1, activation='relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)

    # Fourth Convolutional Block
    x = Conv1D(filters=1024, kernel_size=3, strides=1, activation='relu')(x)

    # Flatten and Fully Connected Layer before LSTM
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)

    # Reshape the data back to 3D for LSTM
    time_steps = 16  # Adjust time_steps based on your data or trial/error
    features_per_time_step = x.shape[-1] // time_steps  # Divide flattened features into time steps
    x = Reshape((time_steps, features_per_time_step))(x)

    # LSTM Layers
    x = LSTM(64, return_sequences=True)(x)
    x = LSTM(64, return_sequences=False)(x)

    # Fully Connected Layers after LSTM
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    # Output Layer
    outputs = Dense(nb_classes, activation='softmax')(x)

    # Build Model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the Model
    model.compile(
        optimizer=Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

