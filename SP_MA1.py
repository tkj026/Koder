"""
Importing packages
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal.windows as sp
plt.rcParams["figure.figsize"] = [16,12]
plt.rcParams.update({"font.size":18})

"""
Reading data
"""
H1_PATH = 'C:\\Users\\tormh\\OneDrive\\Skrivebord\\Jobb\\Tor\\Codes\\Codes\\KODER_SIGNAL_PROCESSING\\H-H1_LOSC_4_V2-1126259446-32.hdf5'
L1_PATH = 'C:\\Users\\tormh\\OneDrive\\Skrivebord\\Jobb\\Tor\\Codes\\Codes\\KODER_SIGNAL_PROCESSING\\L-L1_LOSC_4_V2-1126259446-32.hdf5'
h=h5py.File(H1_PATH,"r")        # Reading file from H1
l = h5py.File(L1_PATH, "r")     # Reading file from L1

x_H = h["strain/Strain"]                                    # Hanford signal          
x_L = l["strain/Strain"]                                    # Livingston signal
x_H = x_H[()]
x_L = x_L[()]
"""
Data
"""
f_0 = 4096                                                  # Sample rate (Hz) for both signals
#1
"""
The amount of samples is the length of the list
"""
samples_H = len(x_H)                                        # The amount of samples from Hanford
samples_L = len(x_L)                                        # The amount of samples from Livingston

#2
"""
The amount of seconds is the amount of samples divided by the sampling rate
"""
seconds_H = len(x_H)/f_0                                    # Seconds of signal Hanford
seconds_L = len(x_L)/f_0                                    # Seconds of signal Livingston

#3
"""
The time signal have to go from the time at the first signal to the time at the last signal
"""
t_n = np.arange(0, seconds_L, 1/f_0)                        # Time signal from 0 - 32 seconds                        

#4, 5
"""
Nyquist frequency of the given samplerate
"""
Nyq_freq = f_0/2                                            # Nyquist frequency is 2048 Hz


"""
The sample rate is sufficiently high and the highest frequency that can be measured has to be below 2048 Hz
"""

# 6
# 7

"""
Plotting the data
"""
#1,2


plt.plot(t_n, x_H, label = "$x_H[n]$")                      # Plotting the time signal with the Hanford signal
plt.title("Hanford signal")                                 # Making title
plt.xlabel("Time (s)")                                      # Labeling x axis
plt.ylabel("Strain")                                        # Labeling y axis
plt.legend()                                                # Placing legend on the axis
plt.tight_layout()                                          # Changing the layout
plt.show()                                                  # Printing graph


plt.plot(t_n, x_L, label = "$x_L[n]$")                      # Plotting the time signal with the Livingston signal
plt.title("Livingston signal")                              # Making title
plt.xlabel("Time (s)")                                      # Labeling x axis
plt.ylabel("Strain")                                        # Labeling y axis
plt.legend()                                                # Placing legend on the axis
plt.tight_layout()                                          # Changing the layout
plt.show()                                                  # Printing graph



#3
max_H = np.max(x_H)                                         # Max value Hanford signal                     
min_H = np.min(x_H)                                         # Min value Hanford signal
#print(max_H, min_H)
max_L = np.max(x_L)                                         # Max value Livingston signal
min_L = np.min(x_L)                                         # Min Value Livingston signal
#print(max_L, min_L)

"""
Selecting window function
"""

N = np.arange(0, 1, 1/4096)                                 # Samples

def X1(n):                                                   
    return np.cos((2*np.pi* 31.5*n))                        # x[n]

def hamming(x,n):
    return sp.hamming(len(n))*x                             # Hamming window function

def hann(x,n):
    return sp.hann(len(n))*x                                # Hann window function

hamming = hamming(X1(N),N)                                  # Applying the Hamming window to the sinusoid  
hann = hann(X1(N),N)                                        # Applying the Hann window to th sinusoid 
"""
plotting the cos signal and the windowed cos signal
"""


plt.plot(N, X1(N), label = "$x[n] = \cos(2\pi f n/f_s)$")   # Plotting the sinusoid 
plt.xlabel("Time (s)")                                      # Defining the x axis
plt.ylabel("Amplitude")                                     # Defining the y axis
plt.legend()                                                # Placing legend on the axis
plt.tight_layout()                                          # Changing the layout
plt.show()                                                  # Printing the plot

plt.plot(N, hann, label = "$x[w]x_h[n]$")                   # Plotting the windowed sinusoid
plt.xlabel("Time (s)")                                      # Defining the x axis
plt.ylabel("Amplitude")                                     # Defining the y axis
plt.legend()                                                # Placing legend on the axis
plt.tight_layout()                                          # Changing the layout
plt.show()                                                  # Printing the plot

x_h_f = np.fft.rfft(X1(N))                                  # Computing rfft for the sinusoid
x_hamming_f = np.fft.rfft(hamming)                          # Computing rfft for the Hamming window
x_hann_f = np.fft.rfft(hann)                                # Computing rfft for the Hann window
fvec = np.arange(len(x_h_f))                                # Creating a frequency vector

"""
fft.fft plots
"""


plt.semilogy(fvec, np.abs(x_h_f), label = "$\hat{X}[k]$")   # The sinusoids spectral magnitude in base 10 log scale
plt.xlabel("Frequency (Hz)")                                # Defining the x axis
plt.ylabel("Magnitude")                                     # Defining the y axis
plt.legend()                                                # Placing legend on the axis
plt.tight_layout()                                          # Changing the layout
plt.show()                                                  # Printing the plot

plt.semilogy(fvec, np.abs(x_h_f), label = "$\hat{X}[k]$")   # Sinusoids spectral magnitude(SM) in 10 log scale
plt.semilogy(fvec, np.abs(x_hamming_f), label = "$\hat{X}_w[k] - Hamming $") # The hamming windows SM in 10 log scale
plt.semilogy(fvec, np.abs(x_hann_f), label = "$\hat{X}_w[k]$ - Hann") # The Hann window SM in base 10 log scale
plt.xlabel("Frequency (Hz)")                                # Defining the x axis
plt.ylabel("Magnitude")                                     # Defining y axis
plt.legend()                                                # Placing legends on axis
plt.tight_layout()                                          # Changing layout
plt.show()                                                  # Printing the plot


"""
spectral response at w = pi
"""



pivec = np.linspace(0, np.pi, len(x_h_f))                   # Vector from 0 to pi
plt.semilogy(pivec, np.abs(x_h_f), label = "$\hat{X}[k]$")   # Sinusoids spectral magnitude(SM) in 10 log scale
plt.semilogy(pivec, np.abs(x_hann_f), label = "$\hat{X}_w[k]$ - Hann") # The Hann window SM in base 10 log scale
plt.xlabel("Frequency (Hz)")                                # Defining the x axis
plt.ylabel("Magnitude")                                     # Defining y axis
plt.legend()                                                # Placing legends on axis
plt.tight_layout()                                          # Changing layout
plt.show()                                                  # Printing the plot


N = len(x_H)                                                # Defining N
w =  sp.hann(N)                                             # Hann window function
Livingston = np.fft.rfft(x_L)                               # rfft of the Livingstone signal
Livingston_window = np.fft.rfft(x_L*w)                      # rfft of the windowed livingstone signal
Hanford = np.fft.rfft(x_H)                                  # rfft of the Hanford signal
Hanford_window = np.fft.rfft(w*x_H)                         # rfft of the windowed Hanford signal


fvec = 1/(1/f_0*N) * np.arange(len(Hanford))                # Defining a frequency vector

plt.semilogy(fvec, np.abs(Hanford), label = "Hanford signal")   # SM of the Hanford signal log 10
plt.semilogy(fvec, np.abs(Hanford_window), label = "Hanford signal with Hann window")   # SM windowed Hanford signal 
plt.xlabel("Frequency (Hz)")                                    # Defining x axis
plt.ylabel("Magnitude")                                         # Defining y axis        
plt.legend()                                                    # Placing legend on axis
plt.tight_layout()                                              # Changing layout
plt.show()                                                      # Printing the plot

plt.semilogy(fvec, np.abs(Livingston), label = "Livingston signal") # SM  Livingston signal log 10
plt.semilogy(fvec, np.abs(Livingston_window), label = "Livingston signal with Hann window") # SM windowed Livingston
plt.xlabel("Frequency (Hz)")                                    # Defining x axis
plt.ylabel("Magnitude")                                         # Defining y axis
plt.legend()                                                    # Placing legend on the axis
plt.tight_layout()                                              # Changing layout
plt.show()                                                      # Printing the plot


"""
Power spectrum
"""
sample_rate = f_0                                               # Sampling rate

FT_Hamford = np.fft.rfft(x_H)                                   # rfft Hamford signal
W_FT_Hamford = np.fft.rfft(w*x_H)                               # rfft Hann windowed Hamford signal

FT_Livingston = np.fft.rfft(x_H)                                # rfft Livingstone signal
W_FT_Livingston = np.fft.rfft(w*x_L)                            # rfft Hann windowed Livingston signal

Abs_FT_Hamford = np.abs(FT_Hamford)                             # Absloute value rfft hamford
Abs_W_FT_Hamford = np.abs(W_FT_Hamford)                         # Absolute value rfft windowed hamford

Abs_FT_Livingston = np.abs(FT_Livingston)                       # Absolute value rfft Livingston
Abs_W_FT_Livingston = np.abs(W_FT_Livingston)                   # Absolute value rfft windowed Livingston

freq_vec = np.linspace(0, sample_rate/2, len(Abs_W_FT_Hamford)) # Frequency vector

plt.semilogy(freq_vec, (Abs_FT_Hamford)**2, label = "Hamford signal") # PS Hamford signal
plt.semilogy(freq_vec, (Abs_W_FT_Hamford)**2,label = "Hamford signal with Hann window" ) # PS Hamford signal w/ Hann 
plt.xlabel("Frequency (Hz)")                                    # Defining x axis                                                       
plt.ylabel("Power")                                             # Defining y axis
plt.legend()                                                    # Placing legend on the axis
plt.tight_layout()                                              # Changing layout
plt.show()                                                      # Printing the plot

plt.semilogy(freq_vec, (Abs_FT_Livingston), label = "Livingston signal") # PS Livingston signal
plt.semilogy(freq_vec, (Abs_W_FT_Livingston),label = "Livingsston signal with Hann window" )    # PS Livingston signal w/ Hann 
plt.xlabel("Frequency (Hz)")                                    # Defining x axis                                                       
plt.ylabel("Power")                                             # Defining y axis
plt.legend()                                                    # Placing legend on the axis
plt.tight_layout()                                              # Changing layout
plt.show()                                                      # Printing the plot

plt.semilogy(freq_vec, (Abs_W_FT_Hamford),label = "Hamford signal with Hann window" ) # PS Hamford signal w/ Hann 
plt.xlabel("Frequency (Hz)")                                    # Defining x axis                                                       
plt.ylabel("Power")                                             # Defining y axis
plt.vlines(x = 30, ymin=10**(-23), ymax=10**(-12), label = "Vertical threshold - low frequency noise") # Vertical threshold
plt.hlines(y = 10**(-18), xmin = 30, xmax = 2000, label = "Horizontal threshold - narrow band interference") # Horizontal threshold
plt.legend()                                                    # Placing legend on the axis
plt.tight_layout()                                              # Changing layout
plt.show()                                                      # Printing the plot


plt.semilogy(freq_vec, (Abs_W_FT_Livingston),label = "Livingsston signal with Hann window" )    # PS Livingston signal w/ Hann 
plt.xlabel("Frequency (Hz)")                                    # Defining x axis                                                       
plt.ylabel("Power")                                             # Defining y axis
plt.vlines(x = 30, ymin=10**(-23), ymax=10**(-12), label = "Vertical threshold - low frequency noise") #Vertical threshold
plt.hlines(y = 10**(-18), xmin = 30, xmax = 2000, label = "Horizontal threshold - narrow band interference") #Horizontal threshold
plt.legend()                                                    # Placing legend on the axis
plt.tight_layout()                                              # Changing layout
plt.show()                                                      # Printing the plot
