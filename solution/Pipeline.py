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

def colorGradient(img, s_thresh=(180, 255), sx_thresh=(30, 120)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
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


def pipeline(img,warp_Matrix):
	img_undistort = undistort(img)
	colorGrad = colorGradient(img_undistort)
	unwarped = warp(colorGrad,warp_Matrix)
	return unwarped



# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "../camera_cal/camera_calibration_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]
M = dist_pickle["M"]
img = cv2.imread('../test_images/test5.jpg')

# now calculate M somehow.....

img_pipelined = pipeline(img,M)


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(img_pipelined)
ax2.set_title('Pipelined Image', fontsize=30)
plt.show()	


