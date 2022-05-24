import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib
import os
from tensorflow.python.client import device_lib


def scaling_Stand(x):
    meanX = np.mean(x)
    stdX = np.std(x)
    return (x-meanX)/stdX

def scalingbk_Stand(scaled_x, org_x):
    meanX = np.mean(org_x)
    stdX = np.std(org_x)
    return (scaled_x*stdX)+meanX

def scaling_Stand_Testing(x,org_x):
    meanX = np.mean(org_x)
    stdX = np.std(org_x)
    return (x-meanX)/stdX


def toNumpy(dataFrame):
    numpyList = np.zeros(dataFrame.shape)
    for i in range(numpyList.shape[1]):
        numpyList[:,i] = dataFrame.iloc[:,i]
    return numpyList
    
def createFileList(myDir, format='.csv'):
   fileList = []
   for root, dirs, files in os.walk(myDir):
      files.sort()
      for name in files:
         if name.endswith(format):
               fullName = os.path.join(root, name)
               fileList.append(fullName)
   return fileList

def predict(model,gtm_trj_ToBe_scaled,dataSetX_val,dataSetY_val,btchSize=32):
    forecast = []
    forecast = np.zeros((1,dataSetY_val.shape[1]))
    forecast =np.append(forecast,model.predict(dataSetX_val,batch_size=btchSize),axis=0)
    forecast = np.delete(forecast,(0),axis=0)
    for i in range(dataSetY_val.shape[1]):
        forecast[:,i] = scalingbk_Stand(forecast[:,i],gtm_trj_ToBe_scaled[:,i])
    
    return forecast

def savePred(predictedData,Fname = ""):
    if Fname =="":
        Fname = "CLSTM_Predicted.csv"
    else:
        Fname = Fname+ ".csv"
    np.savetxt(Fname, predictedData, delimiter=",")

def read_input(data_dir):
    fileList = createFileList(data_dir)
    minStep = getMinSteps(fileList)
    scale_cols = ['u', 'v', 'w', 'p', 'q', 'r', 'p_dot', 'q_dot', 'r_dot','lat','lon','alt','alpha']
    control_cols = ['aileron','elevator','rudder']
    gtm_data_list_M_trajec = np.zeros((1,1,1));
    for it in range(len(fileList)):
        gtm_trj = pd.read_csv(fileList[it],sep=',')
        gtm_trj_orignal = toNumpy(gtm_trj)
        gtm_trj_ToBe_scaled = gtm_trj[scale_cols]
        #gtm_trj_scaled_list = np.zeros(gtm_trj_ToBe_scaled.shape)

        #for i in range(gtm_trj_scaled_list.shape[1]):
        #gtm_trj_scaled_list[:,i] = scalingHelper.scaling_Stand(gtm_trj_ToBe_scaled.iloc[:,i])

        gtm_control = gtm_trj[control_cols]
        gtm_control_list = toNumpy(gtm_control)

        gtm_data_list = np.concatenate([gtm_trj_ToBe_scaled,gtm_control_list],axis=1)
        if it ==0:
            gtm_data_list_M_trajec = gtm_data_list[0:minStep,:,np.newaxis] #np.expand_dims(gtm_data_list,axis = 3)
            #gtm_data_list_M_trajec[:,:,0] = gtm_data_list
        else:
            gtm_data_list_M_trajec = np.append(gtm_data_list_M_trajec, np.atleast_3d(gtm_data_list[0:minStep,:]),axis=2)

    print('Done reading Data with size\t',gtm_data_list_M_trajec.shape)
    return gtm_data_list_M_trajec
    #return gtm_data_list,gtm_trj_ToBe_scaled,gtm_trj_orignal

def getMinSteps(filesList):
    TotalSteps = np.zeros(1)
    for it in range(len(filesList)):
        fileLen = len(pd.read_csv(filesList[it],sep=','))
        if it ==0:
            TotalSteps[it] = fileLen
        else:
            TotalSteps = np.append(TotalSteps,fileLen)
    minIndex = int(min(TotalSteps))
    return minIndex

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
    dataSetY = np.zeros((btch_sz,39))

    counter = 0
    for x, y in dataset:
        if counter ==0:
            dataSetX[0:btch_sz,:,:] = x.numpy()
            dataSetY[0:btch_sz,:] = y.numpy()[:,0:39]
        else:
            dataSetX = np.append(dataSetX,x.numpy(),axis=0)
            dataSetY = np.append(dataSetY,y.numpy()[:,0:39],axis=0)
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
        #import os
        #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        #configuration = tf.compat.v1.ConfigProto(device_count={"GPU": 0})
        #session = tf.compat.v1.Session(config=configuration)

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
    model.add(Dense(39))
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
    model.add(Dense(39))
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
    model.add(Dense(39))
    model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(learning_rate=1e-6, momentum=0.9))
    return model


def plotLoss(epochs,loss,outDir,notebook=False):
    if os.path.isdir(outDir):
        pass
    else:
        os.mkdir(outDir)
    
    if notebook==False:
        figsize=[5.8,5.8]
        dpi = 520
    
    if notebook==True:
        figsize = [3.5,3.5]
        dpi = 150
     
    fig = plt.figure(figsize=figsize,dpi=dpi)
    ax = fig.add_subplot()
    ax.set_yscale("log")
    plt.title('Loss Function')
    plt.xlabel('epochs')
    plt.ylabel('Training Loss')
    fig.tight_layout()
    plt.semilogy(epochs, loss, 'k',antialiased=True,linewidth=2.0)
    FileOut = "loss_trainingV0_Clstm.png"
    plt.savefig(os.path.join(outDir,FileOut),bbox_inches='tight')

def plotResponse(predictedData,orignalData,Steps,InitialTime,TotalTime,windowSize,deltaT,outDir,notebook=False):
    if os.path.isdir(outDir):
        pass
    else:
        os.mkdir(outDir)
        
    if notebook==False:
        figsize=[5.8,5.8]
        dpi = 520
        loc = 'upper right'
    
    if notebook==True:
        figsize = [3.5,3.5]
        dpi = 150
        loc = 'best'
        
    fig = plt.figure(figsize=figsize,dpi=dpi)
    LegendFont = matplotlib.font_manager.FontProperties(family="DejaVu Sans",
                                    weight="bold",
                                    style="normal", size=10)
    time_array = np.linspace(InitialTime+(windowSize*deltaT),TotalTime,num=Steps-windowSize)
    plt.plot(time_array,predictedData[:,1],linestyle='-',color="k",antialiased=True,linewidth=2.0, label ='ML')
    #plt.fill_between(time_array,predictedData[:,0],predictedData[:,2],color='k', alpha=.5)
    plt.plot(time_array,orignalData[windowSize:,1],linestyle='--',color="r",antialiased=True,linewidth=2.0, label ='TCM')
    #plt.fill_between(time_array,orignalData[windowSize:,0],orignalData[windowSize:,2],color='r', alpha=.5)

    plt.xlabel('Time (s)')
    plt.ylabel('u (ft/s)')
    fig.tight_layout()
    plt.legend(loc=loc,frameon=False,prop=LegendFont,markerscale=1,handlelength=3)

    FileOut = "u_ML.png"
    plt.savefig(os.path.join(outDir,FileOut))

    fig = plt.figure(figsize=figsize,dpi=dpi)
    plt.plot(time_array,predictedData[:,4],linestyle='-',color="k",antialiased=True,linewidth=2.0, label ='ML')
    plt.plot(time_array,orignalData[windowSize:,4],linestyle='--',color="r",antialiased=True,linewidth=2.0, label ='TCM')

    plt.xlabel('Time (s)')
    plt.ylabel('v (ft/s)')
    fig.tight_layout()
    plt.legend(loc=loc,frameon=False,prop=LegendFont,markerscale=1,handlelength=3)

    FileOut = "v_ML.png"
    plt.savefig(os.path.join(outDir,FileOut))

    fig = plt.figure(figsize=figsize,dpi=dpi)
    plt.plot(time_array,predictedData[:,7],linestyle='-',color="k",antialiased=True,linewidth=2.0, label ='ML')
    plt.plot(time_array,orignalData[windowSize:,7],linestyle='--',color="r",antialiased=True,linewidth=2.0, label ='TCM')
    plt.xlabel('Time (s)')
    plt.ylabel('w (ft/s)')
    fig.tight_layout()
    plt.legend(loc=loc,frameon=False,prop=LegendFont,markerscale=1,handlelength=3)

    FileOut = "w_ML.png"
    plt.savefig(os.path.join(outDir,FileOut))

    fig = plt.figure(figsize=figsize,dpi=dpi)
    plt.plot(time_array,predictedData[:,10],linestyle='-',color="k",antialiased=True,linewidth=2.0, label ='ML')
    plt.plot(time_array,orignalData[windowSize:,10],linestyle='--',color="r",antialiased=True,linewidth=2.0, label ='TCM')

    plt.xlabel('Time (s)')
    plt.ylabel('p (rad/s)')
    fig.tight_layout()
    plt.legend(loc=loc,frameon=False,prop=LegendFont,markerscale=1,handlelength=3)

    FileOut = "p_ML.png"
    plt.savefig(os.path.join(outDir,FileOut))


    fig = plt.figure(figsize=figsize,dpi=dpi)
    plt.plot(time_array,predictedData[:,13],linestyle='-',color="k",antialiased=True,linewidth=2.0, label ='ML')
    plt.plot(time_array,orignalData[windowSize:,13],linestyle='--',color="r",antialiased=True,linewidth=2.0, label ='TCM')
    plt.xlabel('Time (s)')
    plt.ylabel('q (rad/s)')
    fig.tight_layout()
    plt.legend(loc=loc,frameon=False,prop=LegendFont,markerscale=1,handlelength=3)

    FileOut = "q_ML.png"
    plt.savefig(os.path.join(outDir,FileOut))

    fig = plt.figure(figsize=figsize,dpi=dpi)
    plt.plot(time_array,predictedData[:,16],linestyle='-',color="k",antialiased=True,linewidth=2.0, label ='ML')
    plt.plot(time_array,orignalData[windowSize:,16],linestyle='--',color="r",antialiased=True,linewidth=2.0, label ='TCM')
    plt.xlabel('Time (s)')
    plt.ylabel('r (rad/s)')
    fig.tight_layout()
    plt.legend(loc=loc,frameon=False,prop=LegendFont,markerscale=1,handlelength=3)

    FileOut = "r_ML.png"
    plt.savefig(os.path.join(outDir,FileOut))


    fig = plt.figure(figsize=figsize,dpi=dpi)
    plt.plot(time_array,predictedData[:,19],linestyle='-',color="k",antialiased=True,linewidth=2.0, label ='ML')
    plt.plot(time_array,orignalData[windowSize:,19],linestyle='--',color="r",antialiased=True,linewidth=2.0, label ='TCM')
    plt.xlabel('Time (s)')
    plt.ylabel('pdot (rad/s^2)')
    fig.tight_layout()
    plt.legend(loc=loc,frameon=False,prop=LegendFont,markerscale=1,handlelength=3)

    FileOut = "pdot_ML.png"
    plt.savefig(os.path.join(outDir,FileOut))


    fig = plt.figure(figsize=figsize,dpi=dpi)
    plt.plot(time_array,predictedData[:,22],linestyle='-',color="k",antialiased=True,linewidth=2.0, label ='ML')
    plt.plot(time_array,orignalData[windowSize:,22],linestyle='--',color="r",antialiased=True,linewidth=2.0, label ='TCM')
    plt.xlabel('Time (s)')
    plt.ylabel('qdot (rad/s^2)')
    fig.tight_layout()
    plt.legend(loc=loc,frameon=False,prop=LegendFont,markerscale=1,handlelength=3)

    FileOut = "qdot_ML.png"
    plt.savefig(os.path.join(outDir,FileOut))


    fig = plt.figure(figsize=figsize,dpi=dpi)
    plt.plot(time_array,predictedData[:,25],linestyle='-',color="k",antialiased=True,linewidth=2.0, label ='ML')
    plt.plot(time_array,orignalData[windowSize:,25],linestyle='--',color="r",antialiased=True,linewidth=2.0, label ='TCM')

    plt.xlabel('Time (s)')
    plt.ylabel('rdot (rad/s^2)')
    fig.tight_layout()
    plt.legend(loc=loc,frameon=False,prop=LegendFont,markerscale=1,handlelength=3)

    FileOut = "rdot_ML.png"
    plt.savefig(os.path.join(outDir,FileOut))

    fig = plt.figure(figsize=figsize,dpi=dpi)
    plt.plot(time_array,predictedData[:,28],linestyle='-',color="k",antialiased=True,linewidth=2.0, label ='ML')
    plt.plot(time_array,orignalData[windowSize:,28],linestyle='--',color="r",antialiased=True,linewidth=2.0, label ='TCM')

    plt.xlabel('Time (s)')
    plt.ylabel('Latitude (ft)')
    fig.tight_layout()
    plt.legend(loc=loc,frameon=False,prop=LegendFont,markerscale=1,handlelength=3)

    FileOut = "Latitude_ML.png"
    plt.savefig(os.path.join(outDir,FileOut))

    fig = plt.figure(figsize=figsize,dpi=dpi)
    plt.plot(time_array,predictedData[:,31],linestyle='-',color="k",antialiased=True,linewidth=2.0, label ='ML')
    plt.plot(time_array,orignalData[windowSize:,31],linestyle='--',color="r",antialiased=True,linewidth=2.0, label ='TCM')
    plt.xlabel('Time (s)')
    plt.ylabel('Longitude (ft)')
    fig.tight_layout()
    plt.legend(loc=loc,frameon=False,prop=LegendFont,markerscale=1,handlelength=3)

    FileOut = "Longitude_ML.png"
    plt.savefig(os.path.join(outDir,FileOut))


    fig = plt.figure(figsize=figsize,dpi=dpi)
    plt.plot(time_array,predictedData[:,34],linestyle='-',color="k",antialiased=True,linewidth=2.0, label ='ML')
    plt.plot(time_array,orignalData[windowSize:,34],linestyle='--',color="r",antialiased=True,linewidth=2.0, label ='TCM')

    plt.xlabel('Time (s)')
    plt.ylabel('Altitude (ft)')
    fig.tight_layout()
    plt.legend(loc=loc,frameon=False,prop=LegendFont,markerscale=1,handlelength=3)

    FileOut = "Altitude_ML.png"
    plt.savefig(os.path.join(outDir,FileOut))

    fig = plt.figure(figsize=figsize,dpi=dpi)
    plt.plot(time_array,predictedData[:,37],linestyle='-',color="k",antialiased=True,linewidth=2.0, label ='ML')
    plt.plot(time_array,orignalData[windowSize:,37],linestyle='--',color="r",antialiased=True,linewidth=2.0, label ='TCM')

    plt.xlabel('Time (s)')
    plt.ylabel('Alpha (degree)')
    fig.tight_layout()
    plt.legend(loc=loc,frameon=False,prop=LegendFont,markerscale=1,handlelength=3)

    FileOut = "AngleofAttack_ML.png"
    plt.savefig(os.path.join(outDir,FileOut))


    '''ax = fig.add_subplot()
    x_axisfont = {'fontname':"Calibri"} # see note below for family font list
    y_axisfont = {'fontname':"Calibri"}
    number_font = {'fontname':'cmb10'}'''


def getCI(allTraj,trainOrTest,TrainingO=[]):
    z95 = 1.96 # Assuming a normal distribution, Need to be modified for each prob dist.
    size_allTraj = allTraj.shape
    gtm_data_list = np.zeros((size_allTraj[0],size_allTraj[1]*3)) # all scaled list w/ control
    gtm_trj_orignal = np.zeros((size_allTraj[0],size_allTraj[1]*3)) # original data
    gtm_trj_ToBe_scaled = np.zeros((size_allTraj[0],(size_allTraj[1]-3)*3)) # unscaled state parameters
    gtm_trj_scaled_list = np.zeros(gtm_trj_ToBe_scaled.shape)
    temp_mean = np.mean(allTraj,axis=2)
    temp_std = np.std(allTraj,axis=2)
    temp_LB = temp_mean - ((z95*temp_std)/np.sqrt(size_allTraj[2]))
    temp_UB = temp_mean + ((z95*temp_std)/np.sqrt(size_allTraj[2]))
    counter = 0
    for it in range(0,(size_allTraj[1]*3),3):
        gtm_trj_orignal[:,it] = temp_LB[:,counter]
        gtm_trj_orignal[:,it+1] = temp_mean[:,counter]
        gtm_trj_orignal[:,it+2] = temp_UB[:,counter]
        counter = counter+1
        it = it+2
    for i in range(gtm_trj_scaled_list.shape[1]):
        if trainOrTest ==0:
            #gtm_trj_scaled_list[:,i] = scalingHelper.scaling(gtm_trj_orignal[:,i]) #gtm_trj_orignal[:,i] #scalingHelper.scaling_Stand(gtm_trj_orignal[:,i])
            gtm_trj_scaled_list[:,i] = scaling_Stand(gtm_trj_orignal[:,i])
        elif trainOrTest ==1:
            #gtm_trj_scaled_list[:,i] = scalingHelper.scaling(gtm_trj_orignal[:,i]) #gtm_trj_orignal[:,i]  #scalingHelper.scaling_Stand_Testing(gtm_trj_orignal[:,i],TrainingO[:,i])
            gtm_trj_scaled_list[:,i] = scaling_Stand_Testing(gtm_trj_orignal[:,i],TrainingO[:,i])
        gtm_trj_ToBe_scaled[:,i] = gtm_trj_orignal[:,i]
    
    gtm_control_list = gtm_trj_orignal[:,((size_allTraj[1]-3)*3):(size_allTraj[1])*3]

    gtm_data_list = np.concatenate([gtm_trj_scaled_list,gtm_control_list],axis=1)
    #np.savetxt("gtm_data_listNew.csv", gtm_data_list, delimiter=",")
    #np.savetxt("gtm_trj_ToBe_scaledNew.csv", gtm_trj_ToBe_scaled, delimiter=",")
    #np.savetxt("gtm_trj_orignal_New.csv", gtm_trj_orignal, delimiter=",")

    #gtm_trj_orignal[:,0:size_allTraj[1]] = temp_LB 
    #gtm_trj_orignal[:,size_allTraj[1]:size_allTraj[1]*2] = temp_mean
    #gtm_trj_orignal[:,size_allTraj[1]*2:size_allTraj[1]*3] = temp_UB
    print(gtm_trj_orignal.shape)
    print(gtm_data_list.shape)
    print(gtm_trj_ToBe_scaled.shape)
    print(gtm_control_list.shape)
    return gtm_data_list,gtm_trj_ToBe_scaled,gtm_trj_orignal



    