import scipy.io.wavfile
import scipy.linalg
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import ShortTimeFFT

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

fs, data = scipy.io.wavfile.read('universal.wav')
data = data/(2**16)
t = np.arange(len(data))/fs

# Linear Prediction
p = 12 # order, must be even
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

thresh = 0.9
aroots[np.abs(aroots) < thresh] = 0
aroots[np.angle(aroots) < 0] = 0 # Only keep poles in top half of z-plane
order = np.argsort(np.abs(aroots),axis=0)
sortedroots = np.flipud(np.take_along_axis(aroots,order,axis=0))
sortedroots = sortedroots[:p//2]
print(np.sum(sortedroots > 0))

# STFT
nFFT = 128
w = np.hanning(40) # Wideband = 40, Narrowband = 240
hop = 5
SFT = ShortTimeFFT(w,hop=hop,fs=fs,mfft=512,scale_to='magnitude')

Sdata = SFT.stft(data)

fig1, ax1 = plt.subplots(figsize=(6., 4.))  # enlarge plot a bit
t_lo, t_hi = SFT.extent(len(data))[:2]  # time range of plot
ax1.set_title(rf"STFT ({SFT.m_num*SFT.T:g}$\,s$ Hanning window, Threshold={thresh})")
ax1.set(xlabel=f"Time $t$ in seconds ({SFT.p_num(len(data))} slices, " +
               rf"$\Delta t = {SFT.delta_t:g}\,$s)",
        ylabel=f"Freq. $f$ in Hz ({SFT.f_pts} bins, " +
               rf"$\Delta f = {SFT.delta_f:g}\,$Hz)",
        xlim=(t_lo, t_hi))

im1 = ax1.imshow(20*np.log(abs(Sdata)), origin='lower', aspect='auto',
                 extent=SFT.extent(len(data)), cmap='viridis')
fig1.colorbar(im1, label="Log Magnitude $20*log(|S_x(t, f)|)$")

for ind in range(p//2):
    ax1.plot(np.arange(numm)*offset/fs,np.angle(sortedroots[ind])/np.pi/2*fs,'kx')


fig1.tight_layout()
plt.show()
print('yay')