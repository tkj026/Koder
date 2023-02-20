"""
Importing necessary packages
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import cv2

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Exercise 1a
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
PATH_PATCH = 'C:\\Users\\tormh\\OneDrive\\Skrivebord\\Jobb\\Tor\\Codes\\Codes\\KODER_IMAGE_PROCESSING\\Oblig2\\patch.png'
PATH_WINDMILL1 = 'C:\\Users\\tormh\\OneDrive\\Skrivebord\\Jobb\\Tor\\Codes\\Codes\\KODER_IMAGE_PROCESSING\\Oblig2\\windmills1.png'
PATH_WINDMILL2 = 'C:\\Users\\tormh\\OneDrive\\Skrivebord\\Jobb\\Tor\\Codes\\Codes\\KODER_IMAGE_PROCESSING\\Oblig2\\windmills2.png'
"""
Saving image and turning it into an array
"""
patch_image = Image.open(PATH_PATCH)

patch = np.array(patch_image)/255

"""
Splitting image array into its R,G,B,A components and normalizing them
"""
R = patch[:,:,0]
G = patch[:,:,1]
B = patch[:,:,2]
A = patch[:,:,3]

"""
Creating a function that calculates the magnitude of gradients and theta
"""
def gradients(comp):
    """
    Creating arrays that calculate the derivative in x-direction and y-direction
    """
    der_x = np.array([[1, 0, -1], [2, 0,-2], [1, 0, -1]])
    der_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    """
    Convolving the derivative arrays with each component of the image
    """
    G_x = convolve2d(comp, der_x,  mode = 'same', boundary = 'symm')
    
    G_y = convolve2d(comp, der_y,  mode = 'same', boundary = 'symm')

    """
    Using the pythagoran theorem to calculate the magnitude of the derivatives
    and the arctan of the ratio between the sides to calculate the angle
    """
    G = np.sqrt(G_x**2 + G_y**2)
    
    theta = np.mod(np.arctan2(G_y,G_x)*180/np.pi, 180)

    return G, theta, G_x, G_y

"""
Calculating the gradients for the R,G,B components of the image
"""
R_g = gradients(R)[0]
G_g = gradients(G)[0]
B_g = gradients(B)[0]

R_gx = gradients(R)[2]
G_gx = gradients(G)[2]
B_gx = gradients(B)[2]

R_gy = gradients(R)[3]
G_gy = gradients(G)[3]
B_gy = gradients(B)[3]

"""
Creating a function that finds what gradient has the highest value
out of the R,G,B gradients
"""

def max_grad(R_g,G_g,B_g):    
    """
    Saving the shape of the gradient arrays
    """
    M,N = R_g.shape

    """
    Creating an empty array with the same shape
    as the gradient arrays
    """
    max_grad = np.array([M,N])
    
    """
    Comparing each pixel in the gradient arrays for in the R,G,B component
    and saving the pixel with the highest value to the empty array created
    earlier
    """
    T = np.maximum(R_g, G_g)
    max_grad = np.maximum(T, B_g)
    
    return max_grad

"""
Calculating max gradient magnitude and direction and max gradients in
x-direction and y-direction
"""

Max_gradient_magnitude = max_grad(R_g,G_g,B_g)
gradient_direction = gradients(Max_gradient_magnitude)[1]
Max_gradientx_magnitude = max_grad(R_gx,G_gx, B_gx)
Max_gradienty_magnitude = max_grad(R_gy,G_gy, B_gy)


"""
Plotting the separate derivatives
"""
plt.subplot(121)
plt.imshow(Max_gradientx_magnitude, cmap = 'gray')
plt.title("Gradient in x direction")
plt.axis('off')
plt.subplot(122)
plt.imshow(Max_gradienty_magnitude, cmap = 'gray')
plt.title("Gradient in y direction")
plt.axis('off')
plt.show()

"""
Plotting the original gradient magnitude and direction
"""
plt.subplot(121)
plt.imshow(Max_gradient_magnitude, cmap = 'gray')
plt.title("Gradient magnitude")
plt.axis('off')
plt.subplot(122)
plt.imshow(gradient_direction, cmap = 'gray')
plt.title("Gradient direction")
plt.axis('off')
plt.show()

"""
Creating a function that separates the image into bins, by itterating a mask
through the image, the function then calculates the maximum gradient and direction
for each pixel inn the bin. With the image itself as input.
"""

def bins(image):
    """
    splitting the image into its colour components
    """
    R = image[:,:,0]
    G = image[:,:,1]
    B = image[:,:,2]
    
    """
    Saving the shape of the image
    """
    M,N = R.shape

    """
    Setting a step size for the mask
    """
    x_lim = 8
    y_lim = 8
    
    """
    Creating empty list that will hold the gradients for the colour components
    """
    R_grad = []
    G_grad = []
    B_grad = []

    """
    Itterating through the 2d array
    """
    for i in range(0, M, x_lim):
        for j in range(0, N, y_lim):

            """
            masking the colour components (creating the 8x8 cells)
            """
            r_g = gradients(R[i:i + x_lim, j:j + y_lim])[0]
            g_g = gradients(G[i:i + x_lim, j:j + y_lim])[0]
            b_g = gradients(B[i:i + x_lim, j:j + y_lim])[0]

            """
            Appending the masks to the empty lists
            """
            R_grad.append(r_g)
            G_grad.append(g_g)
            B_grad.append(b_g)

    """
    Creating two more empty lists that will hold the matricies with
    the calculated maximum gradients and their directions
    """
    maximum_grad = []
    angles = []

    """
    Itterating through the list that contains the 8x8 cells
    """
    for i in range(len(R_grad)):
        
        """
        Calculating the maximum gradients by using
        the max_grad function created earlier
        """

        maximum_gradient = max_grad(R_grad[i], G_grad[i], B_grad[i])
        maximum_grad.append(maximum_gradient)
    
    """
    Doing the same as above but instead of calculating the
    maximum gradient, the direction is calculated
    """
    for i in range(len(R_grad)):
        angle = gradients(maximum_grad[i])[1]
        angles.append(angle)

    """
    Returning the lists that contain the bins with their
    maximum gradient and direction values
    """
    return maximum_grad, angles

"""
Calculating the maximum grad and direction for every bin
"""
maximum_gradient, angles = bins(patch)


"""
Creating a function that calculates the histogram of gradients
"""
def HOG(gradients, angles, bin_number):

    """
    Creating an empty list
    """
    bins = []
    
    """
    Itterating through the lists that contains the max_grad and 
    direction matricies
    """
    for i in range(len(gradients)):
        """
        Saving each bin that contain the max_grad and direction
        values
        """
        x = gradients[i]
        y = angles[i]
        """
        Saving shape of the matricies
        """
        M, N = x.shape
        
        """
        Creating the bin width constant and a zero array that
        represent the bin centers
        """
        W = 180/bin_number
        Bin = np.zeros(bin_number)

        """
        Iterating through the matracies
        """    
        for j in range(M):
            for k in range(N):
                """
                Saving the gradient value and direction of each pixel
                """ 
                grad_val = x[j,k]
                angle_val = y[j,k]

                """
                Calculating the first index the bin will have
                """
                first_bin_index = int(np.floor(angle_val/W))
                
                """
                Calculating the second index the bin will have while
                considering the 180 degree wrapping
                """
                second_bin_index = 0

                if first_bin_index >= bin_number -1:
                    second_bin_index = 0
                else:
                    second_bin_index = first_bin_index +1
                
                """
                Calculating the wheight value for each index
                """
                first_bin_value_percent = 1 - (angle_val - (W*first_bin_index))/20
                second_bin_value_percent = 1 - first_bin_value_percent

                """
                Assigning the wighted value to the corresponding
                indicies in the histogram
                """
                Bin[first_bin_index] += grad_val*first_bin_value_percent
                Bin[second_bin_index] += grad_val*second_bin_value_percent

        """
        Appending the Histogram to the empty list and
        resetting the histogram values for the next bin
        """
        bins.append(Bin)
    """
    Returning the list that contains the HOG of all the bins
    """    
    return bins

"""
Calculating the histogram of gradients for the patch image
"""
f = HOG(maximum_gradient, angles, 9)

"""
Turning the list of HOGs into an flattened array
"""
r = np.array(f).flatten()

"""
Plotting the flattened array
"""
plt.plot(r/np.linalg.norm(r+0.0001))
plt.title("Feature vector exerise 1a")
plt.show()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Exercise 1b
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Open images
patch = cv2.imread(PATH_PATCH,1)
img1 = cv2.imread(PATH_WINDMILL1,1)
img2 = cv2.imread(PATH_WINDMILL2,1)

# Create HOG model for patch image
cellSize = (8,8)
cellsPerBlock = (3,6)
blockSize = (24,48)
winSize = (patch.shape[1],patch.shape[0])
blockStride = (1,1)
nBins = 9
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nBins)
# Compute HOG descriptor for patch
fVector = hog.compute(patch)

def HOG_features(img, threshould_value):
    # Padding of windmill image
    vPad = 24 # vertical padding
    hPad = 12 # horisontal padding
    padImg = cv2.copyMakeBorder(img, vPad, vPad, hPad, hPad, cv2.BORDER_REPLICATE)
    # Create HOG model for windmill image
    winSize = (padImg.shape[1], padImg.shape[0])
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nBins)
    # Compute HOG descriptors for windmill image
    winStride = (1,1)
    nFeatures = cellsPerBlock[0] * cellsPerBlock[1] * nBins
    fVectors = hog.compute(padImg, winStride).reshape(padImg.shape[1]-blockSize[0]+1,
    padImg.shape[0]-blockSize[1]+1,
    nFeatures).transpose((1,0,2))


    f_t = fVector
    f_ij = fVectors

    r,t,y = f_ij.shape
    p = np.dot(f_ij, f_t).reshape(r,t)


    M,N = p.shape
    p_t = np.copy(p)
    for i in range(M):
        for j in range(N):
            if p_t[i,j] <= threshould_value:
                p_t[i,j] = 0
            else:
                p_t[i,j] = p_t[i,j]
    return p, p_t, fVector

wm1_p, wm1_pt, fVector = HOG_features(img1, 0.41)
wm2_p, wm2_pt, fVector1 = HOG_features(img2, 0.41)


plt.plot(fVector)
plt.title("HOG feature vector")
plt.show()

plt.subplot(211)
plt.suptitle("windmill1.png", y = 0.95)
plt.imshow(wm1_p)
plt.title("Detected windmills")
plt.axis('off')
plt.subplot(212)
plt.imshow(wm1_pt)
plt.title("Detected windmills, with thresholding value = 0.41")
plt.axis('off')
plt.show()

plt.subplot(211)
plt.suptitle('Windmill2.png', y = 0.95)
plt.imshow(wm2_p)
plt.title("Detected windmills")
plt.axis('off')
plt.subplot(212)
plt.imshow(wm2_pt)
plt.title("Detected windmills, with thresholding value = 0.41")
plt.axis('off')
plt.show()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Exercise 1c
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

windmills1_image = Image.open(PATH_WINDMILL1)
windmills2_image = Image.open(PATH_WINDMILL2)

windmills1 = np.array(windmills1_image)
windmills2 = np.array(windmills2_image)
"""
Function that splits the rgb image into its components
"""
def image_split(image):
    R = image[:,:,0]
    G = image[:,:,1]
    B = image[:,:,2]
    return R, G, B

"""
Splitting the images into its color components
"""
wm1_R, wm1_G, wm1_B = image_split(windmills1)
wm2_R, wm2_G, wm2_B = image_split(windmills2)

"""
Creating gaussian filter, using equation (11-67) from page 883 in textbook
"""
def gaussian(window_size, sigma):
    X = np.arange(-window_size/2 +1, window_size/2 +1)
    Y = np.arange(-window_size/2 +1, window_size/2 +1)
    x,y = np.meshgrid(X,Y)

    G = 1/(2*np.pi*sigma**2) * np.exp(-(x**2 + y**2)/(2*sigma**2))
    return G

"""
Function that performs the gaussian pyramid
"""
def gaussian_pyramid(img, k, sigma):
    """
    - Creating an array of all the k values
    - Creating an empty list to hold the images
    """
    k_arr = np.array([sigma, k*sigma])
    img_list = []
    img_list.append(img)
    
    """
    Iterating through the k array
    """
    for i in k_arr:
        """
        Convolving the last image in the image list with the gaussian blur kernel
        """
        s = convolve2d(img_list[-1], gaussian(window_size = 5, sigma = i), mode = 'same', boundary='symm')
        """
        Creating an image array with half the rows and columns from the previous image
        """
        new_img = s[1::2, 1::2]

        """
        Appending the image with shape M/2,N/2 to the image list
        """        
        img_list.append(new_img)
    
    return img_list    

"""
Applying the gaussian pyramid to the RGB components of the images
"""

wm1_R_ss = gaussian_pyramid(wm1_R, k = 1, sigma = 1)
wm1_G_ss = gaussian_pyramid(wm1_G, k = 1, sigma = 1)
wm1_B_ss = gaussian_pyramid(wm1_B, k = 1, sigma = 1)

wm2_R_ss = gaussian_pyramid(wm2_R, k = 1, sigma = 1)
wm2_G_ss = gaussian_pyramid(wm2_G, k = 1, sigma = 1)
wm2_B_ss = gaussian_pyramid(wm2_B, k = 1, sigma = 1)

"""
Turning the three RGB components into one color image
"""
def grayscale_to_color(R,G,B):
    img_list = []
    for i in range(len(R)):
        x = (R[i],G[i],B[i])
        color_img = np.dstack(x).astype(np.uint8)
        img_list.append(color_img)

    return img_list

wm1_gp_unscaled = grayscale_to_color(wm1_R_ss, wm1_G_ss, wm1_B_ss)
wm2_gp_unscaled = grayscale_to_color(wm2_R_ss, wm2_G_ss, wm2_B_ss)


"""
A function that pad the images to the same size as the original image
Only for plotting
Source: https://stackoverflow.com/questions/59241216/padding-numpy-arrays-to-a-specific-size
"""
def pad(img, xlen, y_len, constant_values):

    M,N = img.shape

    x_min = (xlen - M)//2
    x_max = xlen - x_min - M

    y_min = (y_len - N)//2
    y_max = y_len - y_min - N

    return np.pad(img, pad_width=[(x_min, x_max), (y_min, y_max)], mode='constant', constant_values = constant_values)

"""
Resizing the images to the original shape
"""
def resize(img, constant_values):
    new_list = []
    for i in range(len(img)):
        img[i] = pad(img[i], 192,576, constant_values)
        new_list.append(img[i])
    return new_list


wm1_R_ss1 = resize(wm1_R_ss, 255)
wm1_G_ss1 = resize(wm1_G_ss, 255)
wm1_B_ss1 = resize(wm1_B_ss, 255)

wm2_R_ss1 = resize(wm2_R_ss,255)
wm2_G_ss1 = resize(wm2_G_ss,255)
wm2_B_ss1 = resize(wm2_B_ss,255)


wm1_gp_scaled = grayscale_to_color(wm1_R_ss1,wm1_G_ss1,wm1_B_ss1)
wm2_gp_scaled = grayscale_to_color(wm2_R_ss1,wm2_G_ss1,wm2_B_ss1)



"""
Plotting the gaussian blurred images
"""

plt.subplot(311)
plt.suptitle('Windmills1.png', y = 0.95)
plt.imshow(wm1_gp_scaled[0])
plt.title("Original image")
plt.axis('off')
plt.subplot(312)
plt.imshow(wm1_gp_scaled[1])
plt.title("First image blur")
plt.axis('off')
plt.subplot(313)
plt.imshow(wm1_gp_scaled[2])
plt.title("Second image blur")
plt.axis('off')
plt.show()

plt.subplot(311)
plt.suptitle('windmills2.png', y = 0.95)
plt.imshow(wm2_gp_scaled[0])
plt.title("Original image")
plt.axis('off')
plt.subplot(312)
plt.imshow(wm2_gp_scaled[1])
plt.title("First image blur")
plt.axis('off')
plt.subplot(313)
plt.imshow(wm2_gp_scaled[2])
plt.title("Second image blur")
plt.axis('off')
plt.show()


def DOG(img, k, sigma):
    """
    - Creating an array of all the k values
    - Creating an empty list to hold the images
    """
    k_arr = np.array([sigma, k*sigma])
    img_list = []
    img_list.append(img)
    
    """
    Iterating through the k array
    """
    for i in k_arr:
        """
        Convolving the last image in the image list with the gaussian blur kernel
        """
        s = convolve2d(img_list[-1], gaussian(window_size = 5, sigma = i), mode = 'same', boundary='symm')
        """
        Calculating difference of gaussians
        """
        dog = img_list[-1] - s
        """
        Creating an image array with half of the rows and column as the previous image
        """
        new_img = dog[1::2, 1::2]

        """
        Appending the image with shape M/2,N/2 to the image list
        """        
        img_list.append(new_img)
    
    return img_list

"""
Performing difference of gaussian on the windmills images
where we calculate the mean value of the mask for windmill1 and min 
value for the mask for windmill2
"""
wm1_R_blurred = convolve2d(wm1_R, gaussian(window_size = 5, sigma = 0.5), mode = 'same', boundary='symm')
wm1_G_blurred = convolve2d(wm1_G, gaussian(window_size = 5, sigma = 0.5), mode = 'same', boundary='symm')
wm1_B_blurred = convolve2d(wm1_B, gaussian(window_size = 5, sigma = 0.5), mode = 'same', boundary='symm')

wm2_R_blurred = convolve2d(wm2_R, gaussian(window_size = 5, sigma = 0.55), mode = 'same', boundary='symm')
wm2_G_blurred = convolve2d(wm2_G, gaussian(window_size = 5, sigma = 0.55), mode = 'same', boundary='symm')
wm2_B_blurred = convolve2d(wm2_B, gaussian(window_size = 5, sigma = 0.55), mode = 'same', boundary='symm')

wm1_R_dog = DOG(wm1_R_blurred, k = np.sqrt(2), sigma = 0.5)
wm1_G_dog = DOG(wm1_G_blurred, k = np.sqrt(2), sigma = 0.5)
wm1_B_dog = DOG(wm1_B_blurred, k = np.sqrt(2), sigma = 0.5)


wm2_R_dog = DOG(wm2_R_blurred, k = np.sqrt(1/2), sigma = 0.55)
wm2_G_dog = DOG(wm2_G_blurred, k = np.sqrt(1/2), sigma = 0.55)
wm2_B_dog = DOG(wm2_B_blurred, k = np.sqrt(1/2), sigma = 0.55)   

dog_wm1 = grayscale_to_color(wm1_R_dog, wm1_G_dog, wm1_B_dog)
dog_wm2 = grayscale_to_color(wm2_R_dog, wm2_G_dog, wm2_B_dog)

plt.subplot(311)
plt.suptitle('Windmill1.png, k = 2^(1/2), sigma = 0.5', y = 0.95)
plt.imshow(dog_wm1[0])
plt.title("Original image")
plt.axis('off')
plt.subplot(312)
plt.imshow(dog_wm1[1])
plt.title("First blurred difference")
plt.axis('off')
plt.subplot(313)
plt.imshow(dog_wm1[2])
plt.title("Second blurred difference")
plt.axis('off')
plt.show()


plt.subplot(311)
plt.suptitle('Windmill2.png, k = 1/2^(1/2), sigma = 0.55', y = 0.95)
plt.imshow(dog_wm2[0])
plt.title("Original image")
plt.axis('off')
plt.subplot(312)
plt.imshow(dog_wm2[1])
plt.title("First blurred difference")
plt.axis('off')
plt.subplot(313)
plt.imshow(dog_wm2[2])
plt.title("Second blurred difference")
plt.axis('off')
plt.show()

"""
Exercise 1d
"""

patch = cv2.imread(PATH_PATCH,1)

p_wm1_dog1 = HOG_features(dog_wm1[0], threshould_value= 0.33)
p_wm1_dog2 = HOG_features(dog_wm1[1], threshould_value= 0.33)
p_wm1_dog3 = HOG_features(dog_wm1[2], threshould_value= 0.33)

p_wm2_dog1 = HOG_features(dog_wm2[0], threshould_value= 0.33)
p_wm2_dog2 = HOG_features(dog_wm2[1], threshould_value= 0.33)
p_wm2_dog3 = HOG_features(dog_wm2[2], threshould_value= 0.33)

p_wm1_pyr1 = HOG_features(wm1_gp_unscaled[0], threshould_value= 0.33)
p_wm1_pyr2 = HOG_features(wm1_gp_unscaled[1], threshould_value= 0.33)
p_wm1_pyr3 = HOG_features(wm1_gp_unscaled[2], threshould_value= 0.33)

p_wm2_pyr1 = HOG_features(wm2_gp_unscaled[0], threshould_value= 0.33)
p_wm2_pyr2 = HOG_features(wm2_gp_unscaled[1], threshould_value= 0.33)
p_wm2_pyr3 = HOG_features(wm2_gp_unscaled[2], threshould_value= 0.33)

"""
Gaussian pyramid, windmills detected
"""
plt.subplot(321)
plt.suptitle("Gaussian pyramid Approach for Windmills1.png", y = 0.95)
plt.imshow(p_wm1_pyr1[0])
plt.title("Detected windmills original image")
plt.axis('off')
plt.subplot(323)
plt.imshow(p_wm1_pyr2[0])
plt.title("Detected windmills first difference")
plt.axis('off')
plt.subplot(325)
plt.imshow(p_wm1_pyr3[0])
plt.title("Detected windmills second difference")
plt.axis('off')


plt.subplot(322)
plt.imshow(p_wm1_pyr1[1])
plt.title("Detected windmills original image, with threshould value = 0.33")
plt.axis('off')
plt.subplot(324)
plt.imshow(p_wm1_pyr2[1])
plt.title("Detected windmills first difference, with threshould value = 0.33")
plt.axis('off')
plt.subplot(326)
plt.imshow(p_wm1_pyr3[1])
plt.title("Detected windmills second difference, with threshould value = 0.33")
plt.axis('off')
plt.show()

plt.subplot(321)
plt.suptitle("Gaussian pyramid aproach for Windmills2.png", y = 0.95)
plt.imshow(p_wm2_pyr1[0])
plt.title("Detected windmills")
plt.axis('off')
plt.subplot(323)
plt.imshow(p_wm2_pyr2[0])
plt.title("Detected windmills")
plt.axis('off')
plt.subplot(325)
plt.imshow(p_wm2_pyr3[0])
plt.title("Detected windmills")
plt.axis('off')

plt.subplot(322)
plt.imshow(p_wm2_pyr1[1])
plt.title("Detected windmills, with with threshould value = 0.33")
plt.axis('off')
plt.subplot(324)
plt.imshow(p_wm2_pyr2[1])
plt.title("Detected windmills, with threshould value = 0.33")
plt.axis('off')
plt.subplot(326)
plt.imshow(p_wm2_pyr3[1])
plt.title("Detected windmills, with threshould value = 0.33")
plt.axis('off')
plt.show()


"""
Difference of gaussian, windmills detected
"""
plt.subplot(321)
plt.suptitle("DOG Approach for Windmills1.png", y = 0.95)
plt.imshow(p_wm1_dog1[0])
plt.title("Detected windmills original image")
plt.axis('off')
plt.subplot(323)
plt.imshow(p_wm1_dog2[0])
plt.title("Detected windmills first difference")
plt.axis('off')
plt.subplot(325)
plt.imshow(p_wm1_dog3[0])
plt.title("Detected windmills second difference")
plt.axis('off')


plt.subplot(322)
plt.imshow(p_wm1_dog1[1])
plt.title("Detected windmills original image, with threshould value = 0.33")
plt.axis('off')
plt.subplot(324)
plt.imshow(p_wm1_dog2[1])
plt.title("Detected windmills first difference, with threshould value = 0.33")
plt.axis('off')
plt.subplot(326)
plt.imshow(p_wm1_dog3[1])
plt.title("Detected windmills second difference, with threshould value = 0.33")
plt.axis('off')
plt.show()

plt.subplot(321)
plt.suptitle("DOG aproach for Windmills2.png", y = 0.95)
plt.imshow(p_wm2_dog1[0])
plt.title("Detected windmills")
plt.axis('off')
plt.subplot(323)
plt.imshow(p_wm2_dog2[0])
plt.title("Detected windmills")
plt.axis('off')
plt.subplot(325)
plt.imshow(p_wm2_dog3[0])
plt.title("Detected windmills")
plt.axis('off')

plt.subplot(322)
plt.imshow(p_wm2_dog1[1])
plt.title("Detected windmills, with with threshould value = 0.33")
plt.axis('off')
plt.subplot(324)
plt.imshow(p_wm2_dog2[1])
plt.title("Detected windmills, with threshould value = 0.33")
plt.axis('off')
plt.subplot(326)
plt.imshow(p_wm2_dog3[1])
plt.title("Detected windmills, with threshould value = 0.33")
plt.axis('off')
plt.show()

