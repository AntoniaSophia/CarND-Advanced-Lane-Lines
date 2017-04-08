import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def warp(img,warp_Matrix):
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, warp_Matrix, img_size,flags=cv2.INTER_LINEAR)
    return warped


def undistort(img):
    return cv2.undistort(img, mtx, dist, None, mtx)

def mag_thresh(img, convertGray, sobel_kernel=3, mag_thresh=(0, 255)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    if convertGray == True:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    # 2) Take the gradient in x and y separately
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude 
    mag_sobel = np.sqrt(sobel_x*sobel_x + sobel_y*sobel_y)
    abs_sobel = np.absolute(mag_sobel)

    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a binary mask where mag thresholds are met
    # 6) Return this mask as your binary_output image
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1 
    return binary_output

def colorGradient30(img, s_thresh=(180,255), sx_thresh=(30, 120)):
    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]

    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    thresh_min = sx_thresh[0]
    thresh_max = sx_thresh[1]
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Threshold color channel
    s_thresh_min = s_thresh[0]
    s_thresh_max = s_thresh[1]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary


def colorGradient(img, s_thresh=(180, 255), sx_thresh=(30, 120)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    # Sobel x
    sobel_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    #abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    #scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Sobel y
    sobel_y = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1) # Take the derivative in y
    #abs_sobely = np.absolute(sobely) # Absolute y derivative to accentuate lines away from vertical
    #scaled_sobel = np.uint8(255*abs_sobely/np.max(abs_sobely))

    mag_sobel = np.sqrt(sobel_x*sobel_x + sobel_y*sobel_y)
    #abs_sobel = np.absolute(mag_sobel)
    abs_sobel = np.absolute(sobel_x)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( sxbinary, np.zeros_like(sxbinary), s_binary))
    return color_binary


def rgb2bw(rgb, thresh=10):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    
    s_binary = np.zeros_like(r)
    s_binary[(r >= thresh)] = 255
    
    print(np.amax(r))

    return (np.logical_or(r,b)*255)

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def pipeline(img,warp_Matrix):
    img_undistort = undistort(img)
    colorGrad = colorGradient(img_undistort,(170,220),(22,100))
    unwarped = warp(colorGrad,warp_Matrix)
    grayImage = rgb2bw(unwarped)
    return grayImage



# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "../camera_cal/camera_calibration_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]
M = dist_pickle["M"]
img = cv2.imread('../test_images/test5.jpg')

# now calculate M somehow.....

img_pipelined = pipeline(img,M)
height = int(img_pipelined.shape[0]/2)
print(height)

histogram = np.sum(img_pipelined[height:,:], axis=0)
plt.plot(histogram)

#np.bincount(img_pipelined[int(img_pipelined.shape[0]/2):720,:])
#histogram = np.sum(img_pipelined[img_pipelined.shape[0]/2:,:], axis=0)
#plt.plot(histogram)



f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(img_pipelined, cmap='gray')
ax2.set_title('Pipelined Image', fontsize=30)
plt.show()  


