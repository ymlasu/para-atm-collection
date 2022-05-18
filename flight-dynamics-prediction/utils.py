import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import load_model

def scaling_Stand(x):
    meanX = np.mean(x)
    stdX = np.std(x)
    return (x-meanX)/stdX

def scalingbk_Stand(scaled_x, org_x):
    meanX = np.mean(org_x)
    stdX = np.std(org_x)
    return (scaled_x*stdX)+meanX

def toNumpy(dataFrame):
    numpyList = np.zeros(dataFrame.shape)
    for i in range(numpyList.shape[1]):
        numpyList[:,i] = dataFrame.iloc[:,i]
    return numpyList
    
def read_input(data_dir):
    gtm_trj = pd.read_csv(data_dir,sep=',')
    gtm_trj_orignal = toNumpy(gtm_trj)
    #print(gtm_trj_orignal)
    scale_cols = ['u', 'v', 'w', 'p', 'q', 'r', 'p_dot', 'q_dot', 'r_dot','lat','lon','alt']
    control_cols = ['aileron','elevator','rudder']
    gtm_trj_ToBe_scaled = gtm_trj[scale_cols]

    gtm_trj_scaled_list = np.zeros(gtm_trj_ToBe_scaled.shape)
    for i in range(gtm_trj_scaled_list.shape[1]):
        gtm_trj_scaled_list[:,i] = scaling_Stand(gtm_trj_ToBe_scaled.iloc[:,i])

    gtm_control = gtm_trj[control_cols]
    gtm_control_list = toNumpy(gtm_control)

    gtm_data_list = np.concatenate([gtm_trj_scaled_list,gtm_control_list],axis=1)
    print("data size",gtm_data_list.shape)

    return gtm_data_list,gtm_trj_ToBe_scaled,gtm_trj_orignal

def getTime(TotalSteps,stepSize):
    return (TotalSteps*stepSize) - stepSize

def train_split(gtm_data_list,split_ratio=0.8):
    split_time = round(split_ratio*gtm_data_list.shape[0])
    x_train = gtm_data_list[:split_time]
    x_valid = gtm_data_list[split_time:]
    print(x_train.shape)
    print(x_valid.shape)
    return x_train, x_valid


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size+1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size+1))
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.shuffle(buffer_size=shuffle_buffer) # data shuffling
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

def windowed_dataset_predict(series, window_size, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size+1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size+1))
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


def dataSet_numpy(btch_sz,window_size,parmCtrl,dataset):
    dataSetX = np.zeros((btch_sz,window_size,parmCtrl))
    dataSetY = np.zeros((btch_sz,12))

    counter = 0
    for x, y in dataset:
        if counter ==0:
            dataSetX[0:btch_sz,:,:] = x.numpy()
            dataSetY[0:btch_sz,:] = y.numpy()[:,0:12]
        else:
            dataSetX = np.append(dataSetX,x.numpy(),axis=0)
            dataSetY = np.append(dataSetY,y.numpy()[:,0:12],axis=0)
        counter +=1

    print(dataSetX.shape)
    print(dataSetY.shape)

    return dataSetX, dataSetY

def myLSTM(sysFlag):
    if sysFlag ==0:
        import tensorflow as tf
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.layers import Flatten
        from tensorflow.keras.layers import Reshape
        from tensorflow.keras.layers import Lambda
        from tensorflow.keras.layers import Bidirectional
        from tensorflow.keras.layers import LSTM
        from tensorflow.keras.layers import BatchNormalization
        from tensorflow.keras.layers import Conv1D
    elif sysFlag==1:
        import tensorflow as tf
        from keras.layers import Dense
        from keras.layers import Flatten
        from keras.layers import Reshape
        from keras.layers import Lambda
        from keras.layers import Bidirectional
        from keras.layers import LSTM
        from keras.layers import BatchNormalization
        from keras.layers import Conv1D

    model = tf.keras.models.Sequential()
    #model.add(Lambda(lambda x: tf.expand_dims(x, axis=-1),input_shape=[None]))
    model.add(Bidirectional(LSTM(32, return_sequences=True)))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dense(12))
    model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(learning_rate=1e-6, momentum=0.9))
    return model

def myLSTM_DNN(sysFlag):
    if sysFlag ==0:
        import tensorflow as tf
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.layers import Flatten
        from tensorflow.keras.layers import Reshape
        from tensorflow.keras.layers import Lambda
        from tensorflow.keras.layers import Bidirectional
        from tensorflow.keras.layers import LSTM
        from tensorflow.keras.layers import BatchNormalization
        from tensorflow.keras.layers import Conv1D
    elif sysFlag==1:
        import tensorflow as tf
        from keras.layers import Dense
        from keras.layers import Flatten
        from keras.layers import Reshape
        from keras.layers import Lambda
        from keras.layers import Bidirectional
        from keras.layers import LSTM
        from keras.layers import BatchNormalization
        from keras.layers import Conv1D

    model = tf.keras.models.Sequential()
    #model.add(Lambda(lambda x: tf.expand_dims(x, axis=-1),input_shape=[None]))
    model.add(Bidirectional(LSTM(32, return_sequences=True)))
    model.add(Bidirectional(LSTM(32,return_sequences=True)))
    model.add(Dense(70, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(70, activation="relu"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(12))
    model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(learning_rate=1e-6, momentum=0.9))
    return model

def myLSTM_DNN_Conv(window_size,paramCrl,sysFlag):
    if sysFlag ==0:
        import tensorflow as tf
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.layers import Flatten
        from tensorflow.keras.layers import Reshape
        from tensorflow.keras.layers import Lambda
        from tensorflow.keras.layers import Bidirectional
        from tensorflow.keras.layers import LSTM
        from tensorflow.keras.layers import BatchNormalization
        from tensorflow.keras.layers import Conv1D
    elif sysFlag==1:
        import tensorflow as tf
        from keras.layers import Dense
        from keras.layers import Flatten
        from keras.layers import Reshape
        from keras.layers import Lambda
        from keras.layers import Bidirectional
        from keras.layers import LSTM
        from keras.layers import BatchNormalization
        from keras.layers import Conv1D
        
    model = tf.keras.models.Sequential()
    #model.add(Lambda(lambda x: tf.expand_dims(x, axis=-1),input_shape=[None]))
    model.add(Conv1D(filters=32, kernel_size=5,
                      strides=1, padding="same",
                      activation="relu",
                      input_shape=(window_size,paramCrl)))
    model.add(Bidirectional(LSTM(32, return_sequences=True)))
    model.add(Bidirectional(LSTM(32,return_sequences=True)))
    model.add(Dense(70, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(70, activation="relu"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(12))
    model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(learning_rate=1e-6, momentum=0.9))
    return model


def plotLoss(epochs,loss,outDir):
    if os.path.isdir(outDir):
        pass
    else:
        os.mkdir(outDir)
    fig = plt.figure(figsize=[5.8,5.8],dpi=520)
    ax = fig.add_subplot()
    ax.set_yscale("log")
    plt.xlabel('epochs')
    plt.ylabel('Training Loss')
    fig.tight_layout()
    plt.semilogy(epochs, loss, 'k',antialiased=True,linewidth=2.0)
    FileOut = "loss_trainingV0_Clstm.png"
    plt.savefig(os.path.join(outDir,FileOut),bbox_inches='tight')