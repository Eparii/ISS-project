import numpy as np
import matplotlib.pyplot as plt
import scipy
import soundfile as sf
import IPython
from playsound import playsound
from scipy.signal import spectrogram, lfilter, freqz, tf2zpk, find_peaks, buttord, butter

# ukol 1
data, fs = sf.read('./audio/xtetau00.wav')
original_data = data
data = data[:250000]
t = np.arange(data.size) / fs
duration = data.size/fs

print ("\nPŮVODNÍ SIGNÁL")
print ("minimum  je:", data.min(),"\nmaximum je:", data.max())
print ("delka je", duration, "sekund nebo", data.size, "vzorku")

plt.figure(figsize=(6,3))
plt.plot(t, data)
plt.gca().set_xlabel('$t[s]$')
plt.gca().set_title('Původní signál')
plt.tight_layout()
plt.show()

# ukol 2
data = data - np.mean(data)
data = data / max(abs(data))
arr = np.empty( [int(data.size/512) + 1, 1024] )
i = 0

for x in range(int(data.size/512) + 1):
    if i != 0: i-=512
    for y in range(1024):
        arr[x,y] = data[i]
        if i < data.size-1: i+=1

t = np.arange(arr[60].size) / fs
plt.figure(figsize=(6,3))
plt.plot(t, arr[60])
plt.gca().set_xlabel('$t[s]$')
plt.gca().set_title('Znělý rámec')
plt.tight_layout()
plt.show()

#ukol 3
def DFT_mine(x):
    N = x.size
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

# implementace pomoci me vlastni DFT
x = np.arange(0,512)
y = np.abs(DFT_mine(arr[60]))
index = np.arange(512,1024)
y = np.delete(y, index)
plt.figure(figsize=(10,5))
plt.gca().set_title('DFT moje')
plt.gca().set_xlabel('$Koeficient$')
plt.plot(x,y)
plt.show()

# implementace pomoci knihovni funkce
y = np.abs(np.fft.fft(arr[60]))
y = np.delete(y, index)
plt.figure(figsize=(10,5))
plt.gca().set_title('DFT knihovna')
plt.gca().set_xlabel('$Koeficient$')
plt.plot(x,y)
plt.show()

#ukol 4
f, t, sgr = spectrogram(data, fs, nperseg=1024, noverlap=512)
sgr_log = 10 * np.log10(sgr+1e-20) 
plt.figure(figsize=(9,3))
plt.pcolormesh(t,f,sgr_log, shading="gouraud", cmap="jet")
plt.gca().set_xlabel('Čas [s]')
plt.gca().set_ylabel('Frekvence [Hz]')
plt.gca().set_title('Spektogram původního signálu')
cbar = plt.colorbar()
cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)
plt.tight_layout()
plt.show()

#ukol 5
#   vycteno z grafu
#   f1 = 812 Hz
#   f2 = 1624 Hz
#   f3 = 2436 Hz
#   f4 = 3248 Hz
#   Cosinusovky jsou harmonicky vztazene


#ukol 6
freq = [812, 1624, 2436, 3248]

samples = []

for i in range(data.size):
  samples.append(i/fs)

cos1 = np.cos(2 * np.pi * freq[0] * np.array(samples))
cos2 = np.cos(2 * np.pi * freq[1] * np.array(samples))
cos3 = np.cos(2 * np.pi * freq[2] * np.array(samples))
cos4 = np.cos(2 * np.pi * freq[3] * np.array(samples))

out_cos = cos1 + cos2 + cos3 + cos4


sf.write('./audio/4cos.wav', out_cos, fs, subtype='PCM_16')
# playsound('./audio/4cos.wav')

f, t, sgr = spectrogram(out_cos, fs, nperseg=1024, noverlap=512)
sgr_log = 10 * np.log10(sgr+1e-20) 
plt.figure(figsize=(9,3))
plt.pcolormesh(t,f,sgr_log, shading="gouraud", cmap="jet")
plt.gca().set_xlabel('Čas [s]')
plt.gca().set_ylabel('Frekvence [Hz]')
plt.gca().set_title('Spektogram cosinusovek')
cbar = plt.colorbar()
cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)
plt.tight_layout()
plt.show()

#ukol 7

wp = [[freq[0] - 80, freq[0] + 80], [freq[1] - 80, freq[1] + 80],\
      [freq[2] - 80, freq[2] + 80], [freq[3] - 80, freq[3] + 80]]
ws = [[freq[0] - 30, freq[0] + 30], [freq[1] - 30, freq[1] + 30],\
      [freq[2] - 30, freq[2] + 30], [freq[3] - 30, freq[3] + 30]]
nArr = []
wnArr = []
bArr = []
aArr = []
zArr = []
pArr = []
kArr = []

for x in range(4):
  n, wn = buttord(wp[x], ws[x], 3, 40, fs=fs)
  b, a = butter(n, wn, btype='bandstop', output='ba', fs=fs)
  z, p, k = butter(n, wn, btype='bandstop', output='zpk', fs=fs)
  nArr.append(n)
  wnArr.append(wn)
  bArr.append(b)
  aArr.append(a)
  zArr.append(z)
  pArr.append(p)
  kArr.append(k)

N_imp = 64
imp = [1, *np.zeros(N_imp-1)] # jednotkovy impuls
fig, axs = plt.subplots(4, figsize=(12,8))

for x in range(4):
  h = lfilter(bArr[x], aArr[x], imp)
  axs[x].set_title('Filtr {} s frekvenci {} Hz'.format(x+1, freq[x]))
  axs[x].stem(np.arange(N_imp), h, basefmt=' ')
  axs[x].grid(alpha=0.5, linestyle='--')
  
plt.tight_layout()
plt.show()

for x in range(len(bArr)):
  print ("\nFILTR", x+1)
  print ('{:<20}{:10}'.format("koeficienty a", "koeficienty b"))
  for y in range(len(bArr[x])):
    print('{:<20}{:10}'.format(aArr[x][y], bArr[x][y]))
  print()

#ukol 8
ang = np.linspace(0, 2*np.pi,100)
fig, axs = plt.subplots(2,2,figsize=(10,10))
filter_num = 0
for x in range (2):
  for y in range (2):
    axs[x][y].plot(np.cos(ang), np.sin(ang))
    axs[x][y].scatter(np.real(zArr[filter_num]), np.imag(zArr[filter_num]), marker='o', facecolors='none', edgecolors='r', label='nuly')
    axs[x][y].scatter(np.real(pArr[filter_num]), np.imag(pArr[filter_num]), marker='x', color='g', label='póly')
    axs[x][y].set_xlabel('Realná složka $\mathbb{R}\{$z$\}$')
    axs[x][y].set_ylabel('Imaginarní složka $\mathbb{I}\{$z$\}$')
    axs[x][y].grid(alpha=0.5, linestyle='--')
    if x == 0 and y == 0:
      axs[x][y].legend(loc='upper left')
    axs[x][y].set_title('Filtr {} s frekvenci {}Hz'.format(filter_num+1, freq[filter_num]))
    is_stable = (p[x+y].size == 0) or np.all(np.abs(p[x+y]) < 1) 
    print('filtr {} {}'.format(filter_num+1, 'je stabilni' if is_stable else 'neni stabilni'))
    filter_num += 1

plt.tight_layout()
plt.show()

#ukol 9
fig, axs = plt.subplots(4,2, figsize=(10,15))
axs[0][0].set_title('Modul frekvenční charakteristiky $|H(e^{j\omega})|$')
axs[0][1].set_title('Argument frekvenční charakteristiky $\mathrm{arg}\ H(e^{j\omega})$')

for x in range(4):
  w, H = freqz(bArr[x], aArr[x])
  axs[x][0].plot(w / 2 / np.pi * fs, np.abs(H))
  axs[x][0].set_xlabel('Frekvence [Hz]')
  axs[x][1].plot(w / 2 / np.pi * fs, np.angle(H))
  axs[x][1].set_xlabel('Frekvence [Hz]')
  axs[x][0].grid(alpha=0.5, linestyle='--')
  axs[x][1].grid(alpha=0.5, linestyle='--')
  axs[x][0].set_ylabel('Filtr {} s frekvenci {}Hz'.format(x+1, freq[x]))

plt.tight_layout()
plt.show()

#ukol 10
filtered = original_data
for x in range(4):
  filtered = lfilter(bArr[x],aArr[x],filtered)

filtered = filtered / max(abs(filtered))
beeping = 0.05*fs
filtered[:int(beeping)] = 0
print("\nVYFILTROVANÝ SIGNÁL S UPRAVENÝM ROZSAHEM")
print("minimum je:", filtered.min(),"\nmaximum je:", filtered.max())

sf.write('./audio/clean_bandstop.wav', filtered, fs, subtype='PCM_16')
playsound('./audio/clean_bandstop.wav')

f, t, sgr = spectrogram(filtered, fs, nperseg=1024, noverlap=512)
sgr_log = 10 * np.log10(sgr+1e-20) 
plt.figure(figsize=(9,3))
plt.pcolormesh(t,f,sgr_log, shading="gouraud", cmap="jet")
plt.gca().set_xlabel('Čas [s]')
plt.gca().set_ylabel('Frekvence [Hz]')
plt.gca().set_title('Spektogram vyfiltrovaneho signálu')
cbar = plt.colorbar()
cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)
plt.tight_layout()
plt.show()