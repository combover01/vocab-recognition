import tensorflow as tf
import soundfile as sf
import numpy as np
import os
from scipy.signal import ShortTimeFFT
from scipy.signal import decimate
from scipy.linalg import solve_toeplitz
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Precision

def autocorr(x):
    L = len(x)
    xzp = np.concatenate((x,np.zeros((L-1,))))
    r = np.zeros_like(x)
    for k in range(L):
        r[k] = np.sum(xzp[k:k+L]*x)
    return r

def show_performance_curve(training_result, metric, metric_label):

     train_perf = training_result.history[str(metric)]
     validation_perf = training_result.history['val_'+str(metric)]
     intersection_idx = np.argwhere(np.isclose(train_perf,
                                             validation_perf, atol=1e-2)).flatten()[0]
     intersection_value = train_perf[intersection_idx]

     plt.plot(train_perf, label=metric_label)
     plt.plot(validation_perf, label = 'val_'+str(metric))
     plt.axvline(x=intersection_idx, color='r', linestyle='--', label='Intersection')

     plt.annotate(f'Optimal Value: {intersection_value:.4f}',
          xy=(intersection_idx, intersection_value),
          xycoords='data',
          fontsize=10,
          color='green')
                    
     plt.xlabel('Epoch')
     plt.ylabel(metric_label)
     plt.legend(loc='lower right')
     plt.show()

def readData(baseFolder):
     # Read in the data
     words = [name for name in os.listdir(baseFolder)
               if os.path.isdir(os.path.join(baseFolder, name)) and name != '.git']

     maxlength = 0
     for word in words:
          files = os.listdir(baseFolder+'\\'+word+'\\shortened')
          for file in files:
               data, fs = sf.read(baseFolder+'\\'+word+'\\shortened\\'+file)
               if len(data) > maxlength:
                    maxlength = len(data)

     numWords = len(words)
     numFilesperWord = len(files)

     # Downsampling parameters
     q = 6
     maxlength = maxlength//q + 1

     rawData = np.zeros((numWords,numFilesperWord,maxlength))
     for n,word in enumerate(words):
          files = os.listdir(baseFolder+'\\'+word+'\\shortened')
          for m,file in enumerate(files):
               data, fs = sf.read(baseFolder+'\\'+word+'\\shortened\\'+file)
               data = np.array(decimate(data,q))
               rawData[n,m,:] = np.concatenate((data,np.zeros(maxlength-len(data))))
     '''
     # Decimate to arrive at 8kHz sampling rate
     dsData = np.zeros((numWords,numFilesperWord,maxlength))
     for n in range(numWords):
          for m in range(numFilesperWord):
               dsData[n,m,:] = decimate(rawData[n,m,:],q)
     '''
     fs = fs/q
     return rawData, numWords, numFilesperWord, maxlength, fs

def trainModelSpectrogram(baseFolder):
     dsData, numWords, numFilesperWord, maxlength, fs = readData(baseFolder)
     
     # STFT
     nFFT = 256 # Must be even
     lenWindow = 256
     w = np.hanning(lenWindow) # Wideband = 40, Narrowband = 240
     hop = 64
     SFT = ShortTimeFFT(w,hop=hop,fs=fs,mfft=nFFT,scale_to='magnitude')
     SFTData = np.zeros((numWords,numFilesperWord,nFFT//2+1,int(np.ceil(maxlength/hop))))

     for n in range(numWords):
          for m in range(numFilesperWord):
               data = dsData[n,m,:]
               data = SFT.stft(data)
               data = data/np.min(np.abs(data[data!=0]))
               SFTData[n,m,:,:] = 20*np.log10(np.abs(data[:,:int(np.ceil(maxlength/hop))]))
               if False: # Make true to display spectrograms of each word recording
                    fig1, ax1 = plt.subplots(figsize=(6., 4.))  # enlarge plot a bit
                    t_lo, t_hi = SFT.extent(len(dsData[n,m,:]))[:2]  # time range of plot
                    #ax1.set_title(rf"STFT ({SFT.m_num*SFT.T:g}$\,s$ Hanning window), {words[n]}_{m}")
                    ax1.set(xlabel=f"Time $t$ in seconds ({SFT.p_num(len(dsData[n,m,:]))} slices, " +
                         rf"$\Delta t = {SFT.delta_t:g}\,$s)",
                         ylabel=f"Freq. $f$ in Hz ({SFT.f_pts} bins, " +
                         rf"$\Delta f = {SFT.delta_f:g}\,$Hz)",
                         xlim=(t_lo, t_hi))

                    im1 = ax1.imshow(SFTData[n,m,:,:], origin='lower', aspect='auto',
                              extent=SFT.extent(len(dsData[n,m,:])), cmap='viridis')
                    fig1.colorbar(im1, label="Log Magnitude $20*log(|S_x(t, f)|)$")
                    plt.show()

     SFTData[np.isinf(SFTData)] = 0

     # Set up matrices for training the CNN
     Y = np.zeros((numFilesperWord*numWords))
     for i in range(numWords):
          for index in range(numFilesperWord):
               Y[i*numFilesperWord + index] = i

     X = np.zeros((SFTData.shape[2]-1, SFTData.shape[3],numWords*numFilesperWord))
     for i in range(numFilesperWord):
          X[:,:,i] = SFTData[0,i,:-1,:] # Drop the last sample of the STFT for a nice number
          X[:,:,i+numFilesperWord] = SFTData[1,i,:-1,:]
          X[:,:,i+2*numFilesperWord] = SFTData[2,i,:-1,:]

     Xtrain = np.zeros((X.shape[0],X.shape[1],numWords*(numFilesperWord-1)))
     Xval = np.zeros((X.shape[0],X.shape[1],numWords))
     Xtest = np.zeros((X.shape[0],X.shape[1],numWords))

     Ytrain = np.zeros((numWords*(numFilesperWord-1)))
     Yval = np.zeros((numWords))
     Ytest = np.zeros((numWords))

     for i in range(numWords):
          Xtrain[:,:,i*(numFilesperWord-1):(i+1)*(numFilesperWord-1)] = X[:,:,i*numFilesperWord:(i+1)*numFilesperWord-1]
          Xval[:,:,i] = X[:,:,(i+1)*numFilesperWord-1]

          Ytrain[i*(numFilesperWord-1):(i+1)*(numFilesperWord-1)] = Y[i*numFilesperWord:(i+1)*numFilesperWord-1]
          Yval[i] = Y[(i+1)*numFilesperWord-1]

     Ytrain = to_categorical(Ytrain,numWords)
     Yval = to_categorical(Yval,numWords)

     # Format the data the way that tensorflow wants it
     Xtrain = Xtrain.reshape((Xtrain.shape[2],Xtrain.shape[0],Xtrain.shape[1]))
     Xval = Xtest.reshape((Xval.shape[2],Xval.shape[0],Xval.shape[1]))

     model = models.Sequential()
     model.add(layers.Input((X.shape[0], X.shape[1], 1)))
     model.add(layers.BatchNormalization())
     model.add(layers.GaussianNoise(0.1))
     model.add(layers.Conv2D(16, (7,7),activation='relu'))
     model.add(layers.MaxPooling2D((2,2)))
     model.add(layers.Conv2D(32,(5,5), activation='relu'))
     model.add(layers.MaxPooling2D((2,2)))
     model.add(layers.Conv2D(64,(5,5), activation='relu'))
     model.add(layers.MaxPooling2D((2,2)))
     model.add(layers.Flatten())
     model.add(layers.Dense(nFFT//2, activation='relu'))
     model.add(layers.Dense(numWords, activation='softmax'))
     model.summary()

     METRICS = metrics=['accuracy',
               	Precision(name='precision')]

     model.compile(optimizer='adam',
               loss='categorical_crossentropy',
               metrics=METRICS)


     trainingHistory = model.fit(Xtrain,Ytrain,epochs=5,validation_data=(Xval,Yval))
     # testAccuracySpec = model.evaluate(Xtest, Ytest)[1]
     #show_performance_curve(trainingHistory,'accuracy','accuracy')
     return model, maxlength

def trainModelLPC(baseFolder):
     dsData, numWords, numFilesperWord, maxlength, fs = readData(baseFolder)
     # Linear Prediction
     p = 12 # order, must be even
     N = 100 # length of window
     offset = 50 # how much to offset the next window by

     w = np.hanning(N) # window values
     maxm = int(np.floor((maxlength-N)/offset))
     Xlpc = np.zeros((numWords*numFilesperWord,p,maxm))

     for n in range(numWords):
          for m in range(numFilesperWord):
               data = dsData[n,m,:]
               data = data[abs(data)>0] # Fixes the fact that the data was already zero padded
               numm = int(np.floor((len(data)-N)/offset)) # how many windowed time series
               A = np.concatenate((np.ones((1,numm)), np.zeros((p,numm))))

               for i in range(numm):
                    ym = data[offset*i:offset*i+N]*w
                    rm = autocorr(ym)
                    am = solve_toeplitz(rm[0:p],-rm[1:p+1])
                    A[1:,i] = am

               aroots = np.zeros((p,numm),dtype='complex')

               for i in range(numm):
                    aroots[:,i] = np.roots(A[:,i])

               thresh = 0.95
               aroots[np.abs(aroots) < thresh] = 0
               aroots[np.angle(aroots) < 0] = 0 # Only keep poles in top half of z-plane
               order = np.argsort(np.abs(aroots),axis=0)
               sortedroots = np.flipud(np.take_along_axis(aroots,order,axis=0))
               sortedroots = sortedroots[:p//2]

               Xlpc[n*numFilesperWord+m,:p//2,:] = np.concatenate((np.abs(sortedroots),np.zeros((p//2,maxm-numm))),axis=1)
               Xlpc[n*numFilesperWord+m,p//2:,:] = np.concatenate((np.angle(sortedroots),np.zeros((p//2,maxm-numm))),axis=1)

     Xtrainlpc = np.zeros((numWords*(numFilesperWord-1),Xlpc.shape[1],Xlpc.shape[2]))
     Xvallpc = np.zeros((numWords,Xlpc.shape[1],Xlpc.shape[2]))
     # Xtestlpc = np.zeros((numWords,Xlpc.shape[1],Xlpc.shape[2]))

     Ytrainlpc = np.zeros((numWords*(numFilesperWord-1)))
     Yvallpc = np.zeros((numWords))
     # Ytestlpc = np.zeros((numWords,1))

     Y = np.zeros((numFilesperWord*numWords))
     for i in range(numWords):
          for index in range(numFilesperWord):
               Y[i*numFilesperWord + index] = i

     for i in range(numWords):
          Xtrainlpc[i*(numFilesperWord-1):(i+1)*(numFilesperWord-1),:,:] = Xlpc[i*numFilesperWord:(i+1)*numFilesperWord-1,:,:]
          Xvallpc[i,:,:] = Xlpc[(i+1)*numFilesperWord-1,:,:]

          Ytrainlpc[i*(numFilesperWord-1):(i+1)*(numFilesperWord-1)] = Y[i*numFilesperWord:(i+1)*numFilesperWord-1]
          Yvallpc[i] = Y[(i+1)*numFilesperWord-1]

     Ytrainlpc = to_categorical(Ytrainlpc,numWords)
     Yvallpc = to_categorical(Yvallpc,numWords)

     modelLPC = models.Sequential()
     modelLPC.add(layers.Input((Xlpc.shape[1], Xlpc.shape[2], 1)))
     modelLPC.add(layers.GaussianNoise(0.05))
     modelLPC.add(layers.Conv2D(2, (3,3),  activation='relu'))
     modelLPC.add(layers.Dropout(0.25))
     modelLPC.add(layers.Conv2D(4, (3,3), activation='relu'))
     modelLPC.add(layers.Dropout(0.25))
     modelLPC.add(layers.Conv2D(8, (3,3), activation='relu'))
     modelLPC.add(layers.Dropout(0.25))
     modelLPC.add(layers.Conv2D(16, (3,3), activation='relu'))
     modelLPC.add(layers.Flatten())
     modelLPC.add(layers.Dense(32, activation='relu'))
     modelLPC.add(layers.Dense(numWords, activation='softmax'))
     modelLPC.summary()

     METRICSLPC = metrics=['accuracy',
               	Precision(name='precision')]

     modelLPC.compile(optimizer='adam',
               loss='categorical_crossentropy',
               metrics=METRICSLPC)

     modelLPC.fit(Xtrainlpc,Ytrainlpc,epochs=30,validation_data=(Xvallpc,Yvallpc))
     # testAccuracyLPC = modelLPC.evaluate(Xtestlpc, Ytestlpc)[1]
     return modelLPC, maxlength

def predictWithSpecModel(model,maxlength,filepath):
     data, fs = sf.read(filepath)
     rawData = np.array(data)
     
     q = 6
     fs = fs/q
     dsData = decimate(rawData,q)

     if len(dsData) < maxlength:
          dsData = np.concatenate((dsData,np.zeros(maxlength-len(dsData))))
     elif len(dsData) > maxlength:
          dsData = dsData[:maxlength]

     # STFT
     nFFT = 256 # Must be even
     lenWindow = 256
     w = np.hanning(lenWindow) # Wideband = 40, Narrowband = 240
     hop = 64
     SFT = ShortTimeFFT(w,hop=hop,fs=fs,mfft=nFFT,scale_to='magnitude')
     SFTdata = SFT.stft(dsData)
     SFTdata = SFTdata/np.min(np.abs(dsData[dsData!=0]))
     SFTdata = 20*np.log10(np.abs(SFTdata[:,:int(np.ceil(maxlength/hop))]))

     prediction = model.predict_on_batch(SFTdata.reshape(1, SFTdata.shape[0], SFTdata.shape[1]))
     labelPrediction = np.argmax(prediction, axis=1)[0]
     certaintyVal = np.max(prediction, axis=1)[0]
     return labelPrediction, certaintyVal

def predictWithLPCModel(model,maxlength,filepath):
     data, fs = sf.read(filepath)
     rawData = np.array(data)

     q = 6
     fs = fs/q
     dsData = decimate(rawData,q)

     # Linear Prediction
     p = 12 # order, must be even
     N = 100 # length of window
     offset = 50 # how much to offset the next window by

     w = np.hanning(N) # window values
     maxm = int(np.floor((maxlength-N)/offset))
     Xlpc = np.zeros((p,maxm))

     numm = int(np.floor((len(dsData)-N)/offset)) # how many windowed time series

     if numm > maxm:
          numm = maxm

     A = np.concatenate((np.ones((1,numm)), np.zeros((p,numm))))

     for i in range(numm):
          ym = dsData[offset*i:offset*i+N]*w
          rm = autocorr(ym)
          am = solve_toeplitz(rm[0:p],-rm[1:p+1])
          A[1:,i] = am

     aroots = np.zeros((p,numm),dtype='complex')

     for i in range(numm):
          aroots[:,i] = np.roots(A[:,i])

     thresh = 0.95
     aroots[np.abs(aroots) < thresh] = 0
     aroots[np.angle(aroots) < 0] = 0 # Only keep poles in top half of z-plane
     order = np.argsort(np.abs(aroots),axis=0)
     sortedroots = np.flipud(np.take_along_axis(aroots,order,axis=0))
     sortedroots = sortedroots[:p//2]

     Xlpc[:p//2,:] = np.concatenate((np.abs(sortedroots),np.zeros((p//2,maxm-numm))),axis=1)
     Xlpc[p//2:,:] = np.concatenate((np.angle(sortedroots),np.zeros((p//2,maxm-numm))),axis=1)

     prediction = model.predict_on_batch(Xlpc.reshape(1, Xlpc.shape[0], Xlpc.shape[1], 1))
     labelPrediction = np.argmax(prediction, axis=1)[0]
     certaintyVal = np.max(prediction, axis=1)[0]
     return labelPrediction, certaintyVal
'''
baseFolder = 'C:\\Users\\jayde\\School Stuff\\DSP of Speech Signals\\Project\\vocab-recognition\\MyTraining'

#modelS, maxlengthS = trainModelSpectrogram(baseFolder)
modelL, maxlengthL = trainModelLPC(baseFolder)

testFolder = 'C:\\Users\\jayde\\School Stuff\\DSP of Speech Signals\\Project\\vocab-recognition\\MyTest'
testExtensions = ['\\Ephemeral_9.wav', '\\Exquisite_9.wav', '\\publications_9.wav', '\\Thermodynamic_9.wav']

sum = 0
for ind, test in enumerate(testExtensions):
     labelL, valL = predictWithLPCModel(modelL,maxlengthL,testFolder + test)
     print(f'LPC {ind}: Label={labelL}, Val={valL}')
     #labelS, valS = predictWithSpecModel(modelS,maxlengthS,testFolder + test)
     #print(f'Spec {ind}: Label={labelS}, Val={valS}')
     if labelL == ind:
          sum = sum+1
     #if labelS == ind:
     #     sum=sum+1

print(f'Final Result: {sum}/4')
'''