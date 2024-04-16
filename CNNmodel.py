import tensorflow as tf
import soundfile as sf
import numpy as np
import os
from scipy.signal import ShortTimeFFT
from scipy.signal import decimate
from scipy.linalg import solve_toeplitz
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

def autocorr(x):
    L = len(x)
    xzp = np.concatenate((x,np.zeros((L-1,))))
    r = np.zeros_like(x)
    for k in range(L):
        r[k] = np.sum(xzp[k:k+L]*x)
    return r

baseFolder = 'C:\\Users\\jayde\\School Stuff\\DSP of Speech Signals\\Project\\vocab-recognition\\DATASET17_6'

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
maxlength = maxlength//q + 1
dsData = np.zeros((numWords,numFilesperWord,maxlength))
for n in range(numWords):
     for m in range(numFilesperWord):
          dsData[n,m,:] = decimate(rawData[n,m,:],q)

# STFT
nFFT = 64 # Must be even
lenWindow = 50
w = np.hanning(lenWindow) # Wideband = 40, Narrowband = 240
hop = 50
SFT = ShortTimeFFT(w,hop=hop,fs=fs,mfft=nFFT,scale_to='magnitude')
SFTData = np.zeros((numWords,numFilesperWord,nFFT//2+1,int(np.ceil(maxlength/hop))))

for n in range(numWords):
      for m in range(numFilesperWord):
        data = SFT.stft(dsData[n,m,:])
        data = data/np.min(np.abs(data[data!=0]))
        SFTData[n,m,:,:] = 20*np.log10(np.abs(data[:,:int(np.ceil(maxlength/hop))]))
        if False: # Make true to display spectrograms of each word recording
            fig1, ax1 = plt.subplots(figsize=(6., 4.))  # enlarge plot a bit
            t_lo, t_hi = SFT.extent(len(dsData[n,m,:]))[:2]  # time range of plot
            ax1.set_title(rf"STFT ({SFT.m_num*SFT.T:g}$\,s$ Hanning window), {words[n]}_{m}")
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

Xtrain = np.zeros((X.shape[0],X.shape[1],numWords*(numFilesperWord-2)))
Xval = np.zeros((X.shape[0],X.shape[1],numWords))
Xtest = np.zeros((X.shape[0],X.shape[1],numWords))

Ytrain = np.zeros((numWords*(numFilesperWord-2),1))
Yval = np.zeros((numWords,1))
Ytest = np.zeros((numWords,1))

for i in range(numWords):
     Xtrain[:,:,i*(numFilesperWord-2):(i+1)*(numFilesperWord-2)] = X[:,:,i*numFilesperWord+1:(i+1)*numFilesperWord-1]
     Xval[:,:,i] = X[:,:,(i)*numFilesperWord]
     Xtest[:,:,i] = X[:,:,(i+1)*numFilesperWord-1]

     Ytrain[i*(numFilesperWord-2):(i+1)*(numFilesperWord-2)] = Y[i*numFilesperWord:(i+1)*numFilesperWord-2]
     Yval[i] = Y[(i+1)*numFilesperWord-2]
     Ytest[i] = Y[(i+1)*numFilesperWord-1]

# Format the data the way that tensorflow wants it
Xtrain = Xtrain.reshape((Xtrain.shape[2],Xtrain.shape[0],Xtrain.shape[1]))
Xval = Xtest.reshape((Xval.shape[2],Xval.shape[0],Xval.shape[1]))
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

model.fit(Xtrain,Ytrain,epochs=40,validation_data=(Xval,Yval))
testAccuracySpec = model.evaluate(Xtest, Ytest)[1]
print(f'Test Accuracy for Spectrogram Model: {testAccuracySpec}')

'''
# Linear Prediction
p = 12 # order, must be even
N = 100 # length of window
offset = 100 # how much to offset the next window by

w = np.hanning(N) # window values
maxm = int(np.floor((maxlength-N)/offset))
Xlpc = np.zeros((numWords*numFilesperWord,p,maxm))
# fourth dimension: 0 is magnitude, 1 is phase

for n in range(numWords):
     for m in range(numFilesperWord):
          data = dsData[n,m,:]
          data = data[abs(data)>1e-9] # Fixes the fact that the data was already zero padded
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

Xtrainlpc = np.zeros((numWords*(numFilesperWord-2),Xlpc.shape[1],Xlpc.shape[2]))
Xvallpc = np.zeros((numWords,Xlpc.shape[1],Xlpc.shape[2]))
Xtestlpc = np.zeros((numWords,Xlpc.shape[1],Xlpc.shape[2]))

Ytrainlpc = np.zeros((numWords*(numFilesperWord-2),1))
Yvallpc = np.zeros((numWords,1))
Ytestlpc = np.zeros((numWords,1))

for i in range(numWords):
     Xtrainlpc[i*(numFilesperWord-2):(i+1)*(numFilesperWord-2),:,:] = Xlpc[i*numFilesperWord:(i+1)*numFilesperWord-2,:,:]
     Xvallpc[i,:,:] = Xlpc[(i+1)*numFilesperWord-2,:,:]
     Xtestlpc[i,:,:] = Xlpc[(i+1)*numFilesperWord-1,:,:]

     Ytrainlpc[i*(numFilesperWord-2):(i+1)*(numFilesperWord-2)] = Y[i*numFilesperWord:(i+1)*numFilesperWord-2]
     Yvallpc[i] = Y[(i+1)*numFilesperWord-2]
     Ytestlpc[i] = Y[(i+1)*numFilesperWord-1]

modelLPC = models.Sequential()
modelLPC.add(layers.Conv2D(Xlpc.shape[1], (3,3),  activation='relu', input_shape=(Xlpc.shape[1], Xlpc.shape[2], 1)))
modelLPC.add(layers.Conv2D(Xlpc.shape[1]*2, (3,3), activation='relu'))
modelLPC.add(layers.Conv2D(Xlpc.shape[1]*2, (3,3), activation='relu'))
modelLPC.add(layers.Conv2D(Xlpc.shape[1]*2, (3,3), activation='relu'))
modelLPC.add(layers.Flatten())
modelLPC.add(layers.Dense(Xlpc.shape[1]*2, activation='relu'))
modelLPC.add(layers.Dense(Y.shape[0]))
modelLPC.summary()

modelLPC.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

modelLPC.fit(Xtrainlpc,Ytrainlpc,epochs=100,validation_data=(Xvallpc,Yvallpc))
testAccuracyLPC = modelLPC.evaluate(Xtestlpc, Ytestlpc)[1]
print(f'Test Accuracy for LPC Model: {testAccuracyLPC}')
'''