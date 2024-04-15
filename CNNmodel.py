import tensorflow as tf
import soundfile as sf
import numpy as np
import os
from scipy.signal import ShortTimeFFT
from scipy.signal import decimate
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

baseFolder = 'C:\\Users\\jayde\\School Stuff\\DSP of Speech Signals\\Project\\vocab-recognition\\DATASET_48k'

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

rawData = np.zeros((numWords,numFilesperWord,maxlength))
for n,word in enumerate(words):
      files = os.listdir(baseFolder+'\\'+word+'\\shortened')
      for m,file in enumerate(files):
            data, fs = sf.read(baseFolder+'\\'+word+'\\shortened\\'+file)
            rawData[n,m,:] = np.concatenate((np.array(data),np.zeros(maxlength-len(data))))

# Decimate to arrive at 8kHz sampling rate
q = 6
fs = fs/q
dsData = np.zeros((numWords,numFilesperWord,maxlength//q + 1))
for n in range(numWords):
     for m in range(numFilesperWord):
          dsData[n,m,:] = decimate(rawData[n,m,:],q)

# STFT
nFFT = 64 # Must be even
lenWindow = 40
w = np.hanning(lenWindow) # Wideband = 40, Narrowband = 240
hop = lenWindow
SFT = ShortTimeFFT(w,hop=hop,fs=fs,mfft=nFFT,scale_to='magnitude')
SFTData = np.zeros((numWords,numFilesperWord,nFFT//2+1,int(np.ceil(maxlength/lenWindow))))

for n in range(numWords):
      for m in range(numFilesperWord):
        data = SFT.stft(rawData[n,m,:])
        data = data/np.min(np.abs(data[data!=0]))
        SFTData[n,m,:,:] = 20*np.log10(np.abs(data))
        if False: # Make true to display spectrograms of each word recording
            fig1, ax1 = plt.subplots(figsize=(6., 4.))  # enlarge plot a bit
            t_lo, t_hi = SFT.extent(len(rawData[n,m,:]))[:2]  # time range of plot
            ax1.set_title(rf"STFT ({SFT.m_num*SFT.T:g}$\,s$ Hanning window), {words[n]}_{m}")
            ax1.set(xlabel=f"Time $t$ in seconds ({SFT.p_num(len(rawData[n,m,:]))} slices, " +
                        rf"$\Delta t = {SFT.delta_t:g}\,$s)",
                    ylabel=f"Freq. $f$ in Hz ({SFT.f_pts} bins, " +
                        rf"$\Delta f = {SFT.delta_f:g}\,$Hz)",
                    xlim=(t_lo, t_hi))

            im1 = ax1.imshow(SFTData[n,m,:,:], origin='lower', aspect='auto',
                            extent=SFT.extent(len(rawData[n,m,:])), cmap='viridis')
            fig1.colorbar(im1, label="Log Magnitude $20*log(|S_x(t, f)|)$")
            plt.show()

SFTData[np.isinf(SFTData)] = 0

# Set up matrices for training the CNN
Y = np.zeros((numFilesperWord*numWords,1))
for i in range(numFilesperWord):
     Y[i] = 0
     Y[numFilesperWord+i] = 1
     Y[numFilesperWord*2+i] = 2

X = np.zeros((SFTData.shape[2]-1, SFTData.shape[3],numWords*numFilesperWord))
for i in range(numFilesperWord):
     X[:,:,i] = SFTData[0,i,:-1,:] # Drop the last sample of the STFT for a nice number
     X[:,:,i+numFilesperWord] = SFTData[1,i,:-1,:]
     X[:,:,i+2*numFilesperWord] = SFTData[2,i,:-1,:]

Xtrain = np.concatenate((X[:,:,:numFilesperWord-1],
                        X[:,:,numFilesperWord:2*numFilesperWord-1],
                        X[:,:,2*numFilesperWord:3*numFilesperWord-1]),axis=2)
Xtest = np.dstack([X[:,:,numFilesperWord-1],
                       X[:,:,2*numFilesperWord-1],
                       X[:,:,3*numFilesperWord-1]])

Ytrain = np.concatenate((Y[:numFilesperWord-1],
                        Y[numFilesperWord:2*numFilesperWord-1],
                        Y[2*numFilesperWord:3*numFilesperWord-1]))
Ytest = np.array((Y[numFilesperWord-1],
                       Y[2*numFilesperWord-1],
                       Y[3*numFilesperWord-1]))
Xtrain = Xtrain.reshape((Xtrain.shape[2],Xtrain.shape[0],Xtrain.shape[1]))
Xtest = Xtest.reshape((Xtest.shape[2],Xtest.shape[0],Xtest.shape[1]))

model = models.Sequential()
model.add(layers.Conv2D(X.shape[0], (3,3),  activation='relu', input_shape=(X.shape[0], X.shape[1], 1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(X.shape[0]*2, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(X.shape[0]*2, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(X.shape[0]*2, activation='relu'))
model.add(layers.Dense(X.shape[0]//2, activation='relu'))
model.add(layers.Dense(Y.shape[0]))
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(Xtrain,Ytrain,epochs=10,validation_data=(Xtest,Ytest))