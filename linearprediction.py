import scipy.io.wavfile
import scipy.linalg
import matplotlib.pyplot as plt
import numpy as np

def autocorr(x):
    L = len(x)
    r = np.zeros_like(x)
    for k in range(L):
        sum = 0
        for n in range(L):
            if (n-k >= 0) and (n-k < L):
                sum = sum + x[n]*x[n-k]
        r[k] = sum
    return r

fs, data = scipy.io.wavfile.read('apostrophe.wav')
data = data/(2**16)

# Linear Prediction
p = 10 # order, must be even
N = 100 # length of window
offset = 50 # how much to offset the next window by

w = np.hanning(N) # window values
numm = int(np.floor((len(data)-N)/offset)) # how many windowed time series
A = np.concatenate((np.ones((1,numm)), np.zeros((p,numm))))

for i in range(numm):
    ym = data[offset*i:offset*i+N]*w
    rm = autocorr(ym)
    am = scipy.linalg.solve_toeplitz(rm[0:p],-rm[1:p+1])
    A[1:,i] = am

aroots = np.zeros((p,numm),dtype='complex')

for i in range(numm):
    aroots[:,i] = np.roots(A[:,i])

aroots[np.abs(aroots) < 0.95] = 0
aroots[np.angle(aroots) < 0] = 0 # Only keep poles in top half of z-plane
order = np.argsort(np.abs(aroots),axis=0)
sortedroots = np.flipud(np.take_along_axis(aroots,order,axis=0))
sortedroots = sortedroots[:p//2]
print(np.sum(sortedroots > 0))

plt.plot(np.arange(numm)*offset/fs,np.angle(sortedroots[0])/np.pi,'bo')
plt.plot(np.arange(numm)*offset/fs,np.angle(sortedroots[1])/np.pi,'bo')
plt.plot(np.arange(numm)*offset/fs,np.angle(sortedroots[2])/np.pi,'bo')
plt.plot(np.arange(numm)*offset/fs,np.angle(sortedroots[3])/np.pi,'bo')
plt.plot(np.arange(numm)*offset/fs,np.angle(sortedroots[4])/np.pi,'bo')
plt.plot(np.arange(len(data))/fs,data)
plt.show()
print('yay')