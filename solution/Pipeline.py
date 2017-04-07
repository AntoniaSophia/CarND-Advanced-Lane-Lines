import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "../camera_cal/camera_calibration_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

img = cv2.imread('../test_images/test5.jpg')

def undistort(img):
	return cv2.undistort(img, mtx, dist, None, mtx)


def pipeline(img):
	img_undistort = undistort(img)
	return img_undistort

img_pipelined = pipeline(img)


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(img_pipelined)
ax2.set_title('Pipelined Image', fontsize=30)
plt.show()	


