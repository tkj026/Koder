"""
Importing packages
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal.windows as sp
import scipy.signal as sc
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

"""
Windowing function
"""
def hann(x):
    return sp.hann(len(x))*x                                # Hann window function

Hann_Hanford = hann(x_H)                                    # Windowed Hanford signal
Hann_Livingston = hann(x_L)                                 # Windowed Livingston signal

Hanford = np.fft.rfft(Hann_Hanford)                         # DFT windowed Hanford signal
Livingston = np.fft.rfft(Hann_Livingston)                   # DFT windowed Livingston signal

H_han = 1/np.abs(Hanford)                                   # H[k] windowed Hanford signal
H_liv = 1/np.abs(Livingston)                                # H[k] windowed Livingston signal
 
Y_han = H_han*Hanford                                       # Y[k] windowed Hanford signal
Y_liv = H_liv*Livingston                                    # Y[k] windowed Livingston signal

"""
Assignment 8.1
Checking if all elements in |Y[k]| = 1 
"""

for i in np.round(np.abs(Y_han),1):                         # Itterating through |Y[k]|
    if i != 1.0:                                            # Checking if any datapoint is not equal to 1 
        #print("False")                                      # Print false if there are any datapoints not equal to 1                
        break                                               # Stop itteration if any points not equal to 1
    if np.round(np.abs(Y_han), 1)[-1] == 1:                 # Checking if the last element is equal to 1
        #print("All elements in Y[Hanford] are equal to 1")  # Print if the last element is equal to 1
        break                                               # Stop itteration

for i in np.round(np.abs(Y_liv),1):                         # Itterating through |Y[k]|
    if i != 1.0:                                            # Checking if any datapoint is not equal to 1 
        #print("False")                                      # Print false if there are any datapoints not equal to 1                
        break                                               # Stop itteration if any points not equal to 1
    if np.round(np.abs(Y_liv), 1)[-1] == 1:                 # Checking if the last element is equal to 1
        #print("All elements in Y[Livingston] are equal to 1") # Print if the last element is equal to 1
        break                                               # Stop itteration


"""
Assignment 8.4
"""
y_H = np.fft.irfft(Y_han)                                   # Inverse DFT for Hanford signal                                   
y_L = np.fft.irfft(Y_liv)                                   # Inverse DFT for Livingston signal

"""
Assignment 8.5
"""

seconds = len(x_H)/f_0                                      # Calculating t_max for signals
time = np.arange(0, seconds, 1/f_0)                         # Creating time array


plt.plot(time, y_H, label= "Inverse DFT Hanford signal")    # Plotting the Inverse DFT Hanford signal with time
plt.xlabel("Time(s)")                                       # Defining x axis
plt.ylabel("Amplitude")                                     # Defining y axis
plt.title("Hanford signal")                                 # Title
plt.legend()                                                # Placing legend on axis
plt.tight_layout()                                          # Changing Layout
plt.savefig("Whitened_Hanford_signal.png")                  # Saving figure    
plt.show()                                                  # Showing plot

plt.plot(time, y_L, label = "Inverse DFT Livingston signal")# Plotting the Inverse DFT Livingston signal with time
plt.xlabel("Time(s)")                                       # Defining x axis
plt.ylabel("Amplitude")                                     # Defining y axis
plt.title("Livingston signal")                              # Title
plt.legend()                                                # Placing legend on axis
plt.tight_layout()                                          # Changing Layout
plt.savefig("Whitened_Livingston_signal.png")               # Saving figure    
plt.show()                                                  # Showing plot


"""
Assignment 9.1
"""

"""
Making a function that returns magnitude response
"""
L = np.arange(1,40)                                         # Test array for value of  L
def H(L):                                                   # Creating a function for H
    w = 2*np.pi * 300 /4096                                 # Omega har
    up = 1 - np.exp(-1j*w*L)                                # Numerator
    down = 1- np.exp(-1j*w)                                 # Denominator
    x = 1/L * up/down                                       # Given equation for H(omega)
    x_abs = np.abs(x)                                       # Absolute value of H
    x_sq = np.square(x_abs)                                 # Squared absolute value
    H = 10*np.log10(x_sq)                                   # 10log10 value of sq abs
    return H                                                # Return 10log10 value of sq abs of H
    

for i in L:                                                 # Every test L
    if np.isclose(np.real(H(i)), -6, rtol=1e-01, atol=1e-08): # Checking what value of L when H(i) is close to -6dB
        L = i                                               # Integer value for L


checkint = isinstance(L/2, int)

if checkint == False:
    L = L+1



"""
Assignment 9.2
"""
N = 5000                                                    # A big number
h = np.repeat(1/L, L)                                       # Repeating 1/L

H_ei = np.fft.fftshift(np.fft.fft(h, N))                    # Shifting the dft og h
x = np.abs(H_ei)                                            # absolute
x_sq = np.square(x)                                         # squared
x_log = 10*np.log10(x_sq)                                   # log 
freqvec = np.fft.fftshift(np.fft.fftfreq(N, 1/(f_0)))       # Creating frequency array


plt.plot(freqvec, x_log, label = "$10log_{10}$""$|H(e^{i\omega})^2|$") # Plotting H(e^iw)      
plt.xlabel("Frequency (Hz)")                                # Labeling x axis
plt.ylabel("Power (dB)")                                    # Labeling y axis
plt.legend()                                                # Placing legend on axis
plt.tight_layout()                                          # Changing layout
plt.savefig("Power_spectral_response.png")                  # Saving figure
plt.show()                                                  # Show plot


"""
Assignment 9.3
Time delay tau = L/2 * 1/f_S
"""

"""
Assignment 9.4
"""
def run_avg(x, L):                                          # Creating a running avarage function
    h = np.repeat(1/L, L)                                   # Creating h 
    y = np.fft.ifft(np.fft.fft(x)*np.fft.fft(h,len(x)))     # Convolving the input signal with h
    return y                                                # Return new signal

y_Hanford = run_avg(y_H, L=9)                               # Applying running average to whitened Hanford signal 
y_Livingston = run_avg(y_L, L=9)                            # Applying running average to whitened Livingston signal

"""
Assignment 9.5
"""
timeshift = (L-1)/(2*f_0)                                   # Creating time shift array

"""
Assignment 9.6
"""


plt.plot(time - timeshift, y_Hanford, label = "Hanford signal")   # plot
plt.xlabel("Time (s)")                                      # Labeling x axis
plt.ylabel("Strain")                                        # Labeling y axis
plt.title("Low-pass filtered whitened signal")              # Title
plt.legend()                                                # Placing legend
plt.tight_layout()                                          # Changing layout
plt.savefig("Hanford_signal_lowpass.png")                   # Saving figure
plt.show()                                                  # Print plot

plt.plot(time - timeshift, y_Livingston, label = "Livingston signal")   # plot
plt.xlabel("Time (s)")                                      # Labeling x axis
plt.ylabel("Strain")                                        # Labeling y axis
plt.title("Low-pass filtered whitened signal")              # Title
plt.legend()                                                # Placing legend
plt.tight_layout()                                          # Changing layout
plt.savefig("Livingston_signal_lowpass.png")                # Saving figure
plt.show()                                                  # Print plot


"""
Assingment 9.7
The gravitational wave is much clearer now because....
"""

"""
Assingment 10.1
"""

"""
Finding parameters
"""
xmin = np.where(np.isclose(time, 16.1))
xmax = np.where(np.isclose(time, 16.6))
print(xmin)
print(xmax)

"""
Plotting
"""
fig, axis = plt.subplots(2,1)
plt.sca(axis[0])
plt.plot(time[65954:67994], y_H[65954:67994], label = "Whitened Hanford signal")
plt.plot(time[65954:67994] - timeshift, y_Hanford[65954:67994], label = "Filtered Hanford signal, shifted")
plt.ylabel("Strain")
plt.xlabel("Time (s)")
plt.title("Hanford signal")
plt.tight_layout()
plt.legend()

plt.sca(axis[1])
plt.plot(time[65954:67994], y_L[65954:67994], label = "Whitened Livingston signal")
plt.plot(time[65954:67994]-timeshift, y_Livingston[65954:67994], label = "Filtered Livingston signal, shifted")
plt.ylabel("Strain")
plt.xlabel("Time (s)")
plt.title("Livingston signal")
plt.tight_layout()
plt.legend()

plt.savefig("Whitened_and_lowpass.png")                  # Saving figure
plt.show()


fig, axis = plt.subplots(2,1)
plt.sca(axis[0])
plt.plot(time[65954:67994] - timeshift, y_Hanford[65954:67994], label = "Filtered Hanford signal, shifted")
plt.ylabel("Strain")
plt.xlabel("Time (s)")
plt.title("Hanford signal")
plt.tight_layout()
plt.legend()

plt.sca(axis[1])
plt.plot(time[65954:67994]-timeshift, y_Livingston[65954:67994], label = "Filtered Livingston signal, shifted")
plt.ylabel("Strain")
plt.xlabel("Time (s)")
plt.title("Livingston signal")
plt.tight_layout()
plt.legend()

plt.savefig("lowpass_plots.png")                  # Saving figure
plt.show()



"""
Assignment 10.3
"""


alpha = 10**(-19)                           # Constant 

"""
Plotting
"""
fig, axis = plt.subplots(2,1)
plt.sca(axis[0])
plt.plot(time[65954:67994] - timeshift, alpha*y_Hanford[65954:67994], label = "Filtered Hanford signal, shifted")
plt.ylabel("Strain")
plt.xlabel("Time (s)")
plt.title("Hanford signal")
plt.tight_layout()
plt.legend()

plt.sca(axis[1])
plt.plot(time[65954:67994]-timeshift, alpha*y_Livingston[65954:67994], label = "Filtered Livingston signal, shifted")
plt.ylabel("Strain")
plt.xlabel("Time (s)")
plt.title("Livingston signal")
plt.tight_layout()
plt.legend()

plt.savefig("lowpass_plots_scaled_by_alpha.png")                  # Saving figure
plt.show()

"""
Assignment 11
"""

"""
Finding minimum hanford value and max livingston value
"""
min_Hanford = np.min(y_Hanford)                                 # Min value Hanford signal
max_Livingston = np.max(y_Livingston)                           # Max value Livingston signal

index_min_Hanford = np.where(y_Hanford == min_Hanford)          # Finding where min hanford is
index_max_Livingston = np.where(y_Livingston == max_Livingston) # Finfing wher max Livingston is

index_Hanford = index_min_Hanford[0]                            # Index min Hanford
index_Livingston = index_max_Livingston[0]                      # Index max Livingston

n_0 = index_Hanford-index_Livingston                            # n_0
time_delay = n_0 * 1/f_0                                        # tau

print(n_0, time_delay)

"""
Plotting
"""

plt.plot(time[65954:67994]- timeshift - time_delay, np.abs(y_Hanford[65954:67994]), label = "$|y_H[n+n_0]|$")
plt.plot(time[65954:67994] - timeshift, np.abs(y_Livingston[65954:67994]), label = "$|y_L[n]|$")
plt.xlabel("Time (s)")
plt.ylabel("Magnitude")
plt.legend()
plt.tight_layout()
plt.show()

"""
Assingment 12
"""
"""
Parameters for the spectrogram plots
"""
NFFT = f_0//8                   # NFFT
NOVERLAP = NFFT*15//16          # No overlap
window = sp.hann(NFFT)          # Window function
alpha = 10**(-19)               # Alpha
VMIN = -460                     # Vmin Hanford
VMAX = -450                     # Vmax Hanford 

VMIN_L = -458                   # Vmin Livingston
VMAX_L = -448                   # Vmax Livingston

"""
Plot of the spectrograms
"""

fig, axis = plt.subplots(2,1)
plt.sca(axis[0])
plt.specgram(alpha*y_Hanford, NFFT = NFFT, Fs = f_0, noverlap = NOVERLAP, window = window, vmin = VMIN, vmax = VMAX, mode = "magnitude")
plt.ylim(0, 350)
plt.xlim(15.5, 17)
plt.ylabel("Frequency (Hz)")
plt.title("Hanford signal")
plt.colorbar(label = "dB")
plt.tight_layout()
plt.legend()

plt.sca(axis[1])
plt.specgram(alpha*y_Livingston, NFFT = NFFT, Fs = f_0, noverlap = NOVERLAP, window = window, vmax = VMAX_L, vmin = VMIN_L, mode="magnitude")
plt.ylim(0, 350)
plt.xlim(15.5, 17)
plt.ylabel("Frequency (Hz)")
plt.xlabel("Time (s)")
plt.title("Livingston signal")
plt.colorbar(label = "dB")
plt.tight_layout()
plt.legend()

plt.show()


"""
Extracting data from the dynamic spectrums to calculate resolutions
"""
mspec_H,mfreq_H,mt_H,mim_H = plt.specgram(alpha*y_Hanford, NFFT = NFFT, Fs = f_0, noverlap = NOVERLAP, window = window, vmin = VMIN, vmax = VMAX, mode = "magnitude")
mspec_L,mfreq_L,mt_L,mim_L = plt.specgram(alpha*y_Livingston, NFFT = NFFT, Fs = f_0, noverlap = NOVERLAP, window = window, vmax = VMAX_L, vmin = VMIN_L, mode="magnitude")



freq_resolution = f_0/len(mfreq_H)                  # Frequency resolution
time_resolution = freq_resolution**(-1)             # Time resolution

print(freq_resolution, time_resolution)    