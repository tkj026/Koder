"""
Importing packages
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

"""
Exercise 1a
"""

"""
Importing images
"""
head_img = Image.open(r'C:\\Users\\tormh\\OneDrive\\Skrivebord\\Jobb\\Tor\\Codes\\Codes\\KODER_IMAGE_PROCESSING\\Oblig1\\head.jpg')
leg_img = Image.open(r'C:\\Users\\tormh\\OneDrive\\Skrivebord\\Jobb\\Tor\\Codes\\Codes\\KODER_IMAGE_PROCESSING\\Oblig1\\leg.jpg')

"""
Creating median filter function
"""

def median_filter(Image, pads):                                    
    Image_arr = np.array(Image)                              # Turning image into array
    l,r = Image_arr.shape                                    # Saving shape of original array
    padded_image_arr = np.pad(Image_arr, pads, mode = 'edge')# Neighbor padding image array
    New_array = np.zeros([l,r])                              # Zero matrix with same shape as orig image

    """
    Creating two for loops to itterate through every pixel in the padded image array
    """

    for i in range(l):                                              
        for j in range(r):
            N = 2*pads+1                          # The mask size is NxN                                                                
            mask = padded_image_arr[i:i+N,j:j+N]  # The mask that moves through the image array
            median = np.median(mask)              # Calculating the median
            New_array[i,j] = median               # Creating the median filtered image array
    """
    The function returns the median filtered image,  this is done by replacing every pixel in
    the zero matrix with the median value of the NxN mask that itterated through the padded image array 
    """
    return New_array 
             

"""
Appplying the median filters to the images, first with N = 1 and then with N = 7 for 
the head image and N = 15 for the leg image.
"""


filtered_head_pad1 = median_filter(head_img, 1).astype(np.uint8)
filtered_head_final = median_filter(head_img, 3).astype(np.uint8)

filtered_leg_pad1 = median_filter(leg_img, 1).astype(np.uint8)
filtered_leg_final = median_filter(leg_img, 7).astype(np.uint8)                          

"""
Plotting the images together with the filtered images
"""

"""
Head image
"""
plt.subplot(131)
plt.imshow(head_img, cmap = 'gray', vmin = 0, vmax = 255)
plt.axis("off")
plt.title("Original image")
plt.subplot(132)
plt.imshow(filtered_head_pad1, cmap = 'gray', vmin = 0, vmax = 255)
plt.axis("off")
plt.title("3x3 Median filtered image")
plt.subplot(133)
plt.imshow(filtered_head_final, cmap = 'gray', vmin = 0, vmax = 255)
plt.axis("off")
plt.title("7x7 Median filtered image")
plt.show()

"""
Leg image
"""
plt.subplot(131)
plt.imshow(leg_img, cmap = 'gray', vmin = 0, vmax = 255)
plt.axis("off")
plt.title("Original image")
plt.subplot(132)
plt.imshow(filtered_leg_pad1, cmap = 'gray', vmin = 0, vmax = 255)
plt.axis("off")
plt.title("3x3 Median filtered image")
plt.subplot(133)
plt.imshow(filtered_leg_final, cmap = 'gray', vmin = 0, vmax = 255)
plt.axis("off")
plt.title("15x15 Median filtered image")
plt.show()


"""
Exercise 1b
"""

"""
Saving the different filtered images
"""
Head1 = np.array(head_img)
Head2 = np.array(filtered_head_pad1)
Head3 = np.array(filtered_head_final)

Leg1 = np.array(leg_img)
Leg2 = np.array(filtered_leg_pad1)
Leg3 = np.array(filtered_leg_final)

"""
Creating a function that returns the DFT magnitude
"""
def DFT(image_array):
    M, N = image_array.shape
    x = np.fft.fft2(image_array,[2*M, 2*N])
    x1 = np.log(x)
    x2 = np.abs(x1)
    x3 = np.fft.fftshift(x2)
    final = x3.astype(np.uint8)
    return final

"""
Plotting the different DFT magnitudes
"""
plt.subplot(131)
plt.suptitle("DFT magnitudes of Head images", y = 0.75, size = "xx-large")
plt.imshow(DFT(Head1),cmap = 'gray')
plt.title("Original image")
plt.subplot(132)
plt.imshow(DFT(Head2), cmap = 'gray')
plt.title("3x3 median filtered image")
plt.subplot(133)
plt.imshow(DFT(Head3), cmap = 'gray')
plt.title("7x7 median filtered image")
plt.show()

plt.subplot(131)
plt.suptitle("DFT magnitudes of Leg images", y = 0.95, size = "xx-large")
plt.imshow(DFT(Leg1),cmap = 'gray')
plt.title("Original image")
plt.subplot(132)
plt.imshow(DFT(Leg2), cmap = 'gray')
plt.title("3x3 median filtered image")
plt.subplot(133)
plt.imshow(DFT(Leg3), cmap = 'gray')
plt.title("15x15 median filtered image")
plt.show()

"""
Exercise 1c
"""

"""
To create the Noch filter we first have to create the distance function. The 
distance function has the image and the center coordinates (u_k, v_k) as input,
and returns the distance computations D_k and D_nk.
negative 
"""

def Distance_func(Image, u_k, v_k):
    Img = np.array(Image)
    M, N = Img.shape
    x = np.arange(0, 2*N)
    y = np.arange(0, 2*M)
    u,v = np.meshgrid(x,y)

    D = ((u - N - u_k)**2 + (v - M - v_k)**2)
    D_k = np.sqrt(D)

    D_n = ((u - N + u_k)**2 + (v - M + v_k)**2)
    D_nk = np.sqrt(D_n)
    return D_k, D_nk

"""
The notch filter itself follows the butterworth notch reject filter from
equation 4-145 from the textbook. Where the input of the function is the image
itself, the x and y coordinates of the moire patterns from the DFT magnitude specter,
the D_0 constant and n, the order of the filter.

The function starts by creating a empty list, this list will contain all
the notch pairs in the end. Then it turns the image into an array and saves its shape.
Then we itterate through the list containing the x and y coordinates of the moire
patterns, with the failsafe np.atleast_1d, that will guarantee us that the itteration
happens atleast one time, this is in case there only is one coordinate.
We then calculate the centre coordinate (v_k, u_k) corresponding to our moire patterns.
With theese coordinates we can calculate the distance computations from the distance function
we created earlier.

When we have the distance functions we can apply the butterworth noch reject filter following
equation 4-154 in the textbook.

The function then returns the distance computations and the transfer function.
"""
def Noch_filter(Image, x, y, D_0, n):
    h = []                                  
    img = np.array(Image)
    M, N = img.shape
    for i in range(len(np.atleast_1d(x))):
        v_k = M - y[i]
        u_k = N - x[i]
        D_k, D_nk = Distance_func(Image, u_k, v_k)
        p1 = (D_0/D_k)**n
        p2 = (D_0/D_nk)**n
        h_i = (1/(1 + p1) * (1/(1 + p2)))
        h.append(h_i)
    H = np.ones((h[0].shape))
    for i in range(len(np.atleast_1d(x))):
        H *= h[i]
    return H, D_k, D_nk

"""
Creating lists of x and y coordinates corresponding to the moire patterns from
the DFT magnitude plots. 
"""

x_head = [537, 519]
y_head = [166, 247]

x_leg = [419]
y_leg = [904]

"""
Creating the Butterworth notch reject filter transfer functions corresponding
to the moire patterns, the constant D_0 and the order n is found through trial
and error.
"""
D_0head = 30
n_head = 5
H_head, D_kh, D_nkh = Noch_filter(head_img, x_head, y_head, D_0head, n_head)

D_0leg = 15
n_leg = 5
H_leg, D_kl, D_nkl = Noch_filter(leg_img, x_leg, y_leg, D_0leg, n_leg)

"""
Creating a function that applies the notch reject filter to the image.
"""

def filter_applier(image, H):
    """
    Fourier transforming image and multiplying with the Noch filter
    """
    img = np.array(image)
    M,N = img.shape
    img_fft = np.fft.fft2(img, [2*M, 2*N])
    img_shift = np.fft.fftshift(img_fft)
    F = np.log(1 + np.abs(img_shift))
    G = img_shift*H
    mag = F*H
    """
    Inverse Fourier transforming the image multiplied with the Noch filter
    """
    G_shift = np.fft.fftshift(G)
    G_ifft = np.fft.ifft2(G_shift)
    g = np.real(G_ifft)[0:M, 0:N]


    return F, G,mag, g

"""
Applying the filter applier function to the images and their corresponding notch reject
filters
"""

F_head, G_head, NG_head, g_head= filter_applier(head_img, H_head)
F_leg, G_leg, NG_leg, g_leg = filter_applier(leg_img, H_leg)

"""
Plotting the DFT magnitude of the images together with the Noch filtered
images
"""
plt.subplot(121)
plt.imshow(F_head.astype(np.uint8), cmap = 'gray')
plt.title('DFT magnitude head image')
plt.subplot(122)
plt.imshow(NG_head.astype(np.uint8), cmap = 'gray')
plt.title('DFT magnitude Noch filtered head image')
plt.show()

plt.subplot(121)
plt.imshow(F_leg.astype(np.uint8), cmap = 'gray')
plt.title('DFT magnitude leg image')
plt.subplot(122)
plt.imshow(NG_leg.astype(np.uint8), cmap = 'gray')
plt.title('DFT magnitude Noch filtered leg image')
plt.show()

"""
Exercise 1d
"""

"""
Plotting the images together with the noch filtered images and median filtered
images
"""
plt.subplot(131)
plt.imshow(Head1, cmap = 'gray')
plt.title('Original head image')
plt.axis('off')
plt.subplot(132)
plt.imshow(Head3, cmap = 'gray')
plt.title('Median filtered head image with N = 7')
plt.axis('off')
plt.subplot(133)
plt.imshow(g_head, cmap = 'gray')
plt.title('Notch filtered head image, $D_0 = 30$, n = 5')
plt.axis('off')
plt.show()

plt.subplot(131)
plt.imshow(Leg1, cmap = 'gray')
plt.title('Original leg image')
plt.axis('off')
plt.subplot(132)
plt.imshow(Leg3, cmap = 'gray')
plt.title('Median filtered leg image with N = 15')
plt.axis('off')
plt.subplot(133)
plt.imshow(g_leg, cmap = 'gray')
plt.title('Notch filtered leg image, $D_0 = 15$, n = 5')
plt.axis('off')
plt.show()

"""
Exercise 2a
"""

"""
Importing image files and converting them to arrays
"""
sat1 = Image.open(r'C:\\Users\\tormh\\OneDrive\\Skrivebord\\Jobb\\Tor\\Codes\\Codes\\KODER_IMAGE_PROCESSING\\Oblig1\\sat1.tiff')
sat2 = Image.open(r'C:\\Users\\tormh\\OneDrive\\Skrivebord\\Jobb\\Tor\\Codes\\Codes\\KODER_IMAGE_PROCESSING\\Oblig1\\sat2.tiff')

sat1_array = np.array(sat1)
sat2_array = np.array(sat2)
   
"""
Creating a function that calculates the std of NxN non overlapping tiles.
If there is not enough indexes for a NxN tile, the function creates a N x rest, 
rest x N tile or rest x rest tile. This happens if the array is'nt divisible with
the desired tile size.
"""
def std(image_array, mask_size):
    std_list = []
    M,N = image_array.shape
    for i in range(0, M, mask_size):
        for j in range(0, N, mask_size):
            mask = image_array[i:i+mask_size, j:j+mask_size]
            std = np.std(mask)
            std_list.append(std)
    return std_list

"""
To normalize the array we check the dtype of the numbers in the array. in this case the dtype
is uint16. This means that the maximum size a number can have is 2^16 - 1. If we devide the 
array itself with the maximum number, the array will span between 0 and 1. after the arrays 
have been normalized, we plot the histograms of the std's calculated in the previous sub task.
"""

c = 2**16 - 1

normalized_sat1 = sat1_array/c
normalized_sat2 = sat2_array/c

std_sat1 = std(normalized_sat1, 16)
std_sat2 = std(normalized_sat2, 16)

est_std1 = np.min(std_sat1)
est_std2 = np.min(std_sat2)

print(est_std1, est_std2)

plt.subplot(121)
plt.hist(std_sat1, bins=100)
plt.title("Sat1")
plt.subplot(122)
plt.hist(std_sat2, bins = 100)
plt.title("Sat2")
plt.show()

"""
Creating two different inverse filters H(u,v), both are following the atmospheric turbulence 
formula, but they handle the values off H(u,v) close to zero differently. One of the filters
have implemented a cut off frequencey. The other one does not, but instead of cut off frequencey,
a Butterworth lowpass filter will also be applied to handle the values of H(u,v) close to zero.
"""
normalizing_factor = 2**16 - 1

norm_sat1 = sat1_array/normalizing_factor
norm_sat2 = sat2_array/normalizing_factor

"""
Creating inverse filter with cut off frequencey
"""
def inverse_filter_co(image_array, k, threshould):
    M,N = image_array.shape
    x = np.arange(-M/2, M/2)
    y = np.arange(-N/2,N/2)
    u,v = np.meshgrid(y,x)
    H = np.exp(-k*(u**2 + v**2)**(5/6))
    H[np.abs(H) < threshould] = threshould
    return H

"""
Creating a inverse filter without cut off frequencey
"""
def inverse_filter(image_array, k):
    M,N = image_array.shape
    x = np.arange(-M/2, M/2)
    y = np.arange(-N/2,N/2)
    u,v = np.meshgrid(y,x)
    H = np.exp(-k*(u**2 + v**2)**(5/6))
    return H

"""
Fourier transforming the images, determining values for k and dividing the 
image with the inverse filter with a cut off frequencey
"""

N_sat1 = np.fft.fftshift(np.fft.fft2(norm_sat1))
N_sat2 = np.fft.fftshift(np.fft.fft2(norm_sat2))

k_sat1 = 0.004 
k_sat2 = 0.005 

H_co_sat1 = inverse_filter_co(N_sat1, k_sat1, 0.1)
H_co_sat2 = inverse_filter_co(N_sat2, k_sat2, 0.1)

F_co_sat1 = N_sat1/H_co_sat1
F_co_sat2 = N_sat2/H_co_sat2

f_co_sat1 = np.abs(np.fft.ifft2(F_co_sat1))
f_co_sat2 = np.abs(np.fft.ifft2(F_co_sat2))

"""
Creating a Butterworth low pass filter
"""

def Butterworth_lowpass(image_array, D_0, n):
    M,N = image_array.shape
    x = np.arange(0,M)
    y = np.arange(0, N)
    u,v = np.meshgrid(y,x)

    D = np.sqrt((u - (N/2))**2 + (v - (M/2))**2)

    Y = np.ones([M,N])
    Y = 1/(1 + (D/D_0)**(2*n))

    return Y

"""
Dividing the Fourier transformed image with the inverse filter without a cut off
frequencey and applying the Butterworth low pass filter to the inverse filtered
image 
"""
H_sat1 = inverse_filter(N_sat1, k_sat1)
H_sat2 = inverse_filter(N_sat2, k_sat2)

F_sat1 = N_sat1/H_sat1
F_sat2 = N_sat2/H_sat2

D_sat1 = Butterworth_lowpass(norm_sat1, 60, 29)  # 60,9
D_sat2 = Butterworth_lowpass(norm_sat2, 60, 29)

f_sat1 = np.abs(np.fft.ifft2(F_sat1*D_sat1))
f_sat2 = np.abs(np.fft.ifft2(F_sat2*D_sat2))

"""
Plotting the original image together with the two types off inverse filtered images
"""

"""
Sat1
"""

plt.subplot(132)
plt.imshow(sat1_array, cmap = 'gray')
plt.title("Original image")
plt.axis('off')
plt.subplot(131)
plt.imshow(f_sat1, cmap = 'gray')
plt.title("Inverse butterworth filtered image with: k = 0.004, n = 29")
plt.axis('off')
plt.subplot(133)
plt.imshow(f_co_sat1, cmap = 'gray')
plt.title('Inverse filterd image with: k = 0.004, cut off = 0.1')
plt.axis('off')
plt.show()

"""
Sat2
"""
plt.subplot(132)
plt.imshow(sat2_array, cmap = 'gray')
plt.title("Original image")
plt.axis('off')
plt.subplot(131)
plt.imshow(f_sat2, cmap = 'gray')
plt.title("Inverse butterworth filtered image with: k = 0.005, n = 29")
plt.axis('off')
plt.subplot(133)
plt.imshow(f_co_sat2, cmap = 'gray')
plt.title('Inverse filterd image with: k = 0.005, cut off = 0.1')
plt.axis('off')
plt.show()

"""
Exercise 2c
"""

"""
Creating a Wiener filter with the image array, k and K as inputs. The filter
Fourier transform the image array and applies the inverse filter with cut off
frequencey to the transformed image and calulating abs(H). Then, using the 
equation 5-85 from the book and performing an inverse Fourier transform to
the wiener filter. and returning the new filtered image.
"""
def Wiener_filter_co(Image_array, k, K):
    G = np.fft.fftshift(np.fft.fft2(Image_array))
    H = inverse_filter_co(G, k, 0.1)
    H_abs = np.abs(H)**2

    F = ((1/H)*(H_abs/(H_abs + K)))*G
    f = np.abs(np.fft.ifft2(np.fft.fftshift(F)))
    return f

"""
Applying the wiener filter to the images
"""
wiener_sat1 = Wiener_filter_co(norm_sat1, k_sat1 , 4e-6)
wiener_sat2 = Wiener_filter_co(norm_sat2, k_sat2, 4e-6)

"""
Plotting the original image with the Wiener filtered image
"""
plt.subplot(131)
plt.imshow(sat1_array, cmap = 'gray')
plt.title("Original sat1 image")
plt.axis('off')
plt.subplot(132)
plt.imshow(wiener_sat1, cmap = 'gray')
plt.title("Wiener filtered sat1 image")
plt.axis('off')
plt.subplot(133)
plt.imshow(f_sat1, cmap = 'gray')
plt.title("Inverse butterworth filtered image with: k = 0.004, n = 29")
plt.axis('off')
plt.show()

plt.subplot(131)
plt.imshow(sat2_array, cmap = 'gray')
plt.title("Original sat2 image")
plt.axis('off')
plt.subplot(132)
plt.imshow(wiener_sat2, cmap = 'gray')
plt.title("Wiener filtered sat2 image")
plt.axis('off')
plt.subplot(133)
plt.imshow(f_co_sat2, cmap = 'gray')
plt.title('Inverse filterd image with: k = 0.005, cut off = 0.1')
plt.axis('off')
plt.show()
