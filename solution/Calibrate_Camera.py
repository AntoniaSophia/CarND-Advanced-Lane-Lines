import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import pickle


# Event listener, if button of mouse is clicked
def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    #print the x,y coordinates of mouse
    print(ix, iy)
    return



nx = 9 # the number of inside corners in x
ny = 6 # the number of inside corners in y

# define the left bottom corner and right bottom corner of the warped image
leftLaneCenter = 300
rightLaneCenter = 900

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((ny*nx,3), np.float32)
objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)



# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('../camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
        #print(corners)
        # Draw and display the corners
        #cv2.drawChessboardCorners(img, (9,6), corners, ret)
        #write_name = 'corners_found'+str(idx)+'.jpg'
        #cv2.imwrite(write_name, img)
        #cv2.imshow('img', img)
        #cv2.waitKey(5000)



def calculateWarp(img, mtx, dist):
    # Use the OpenCV undistort() function to remove distortion
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # Convert undistorted image to grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    # Search for corners in the grayscaled image

    # define the 4 coordinates (right top, right bottom, left bottom, left top) in the source image
    # those points are just derived from clicking on an image of a straight road track straight_lines1.jpg
    src = np.float32(
        [[703 , 460],
         [1104 , 718],
         [207 , 718],
         [578 , 460]])





    # define the 4 desired coordinates (right top, right bottom, left bottom, left top) in the source image
    dst = np.float32(
        [[rightLaneCenter,0],
         [rightLaneCenter,718],
         [leftLaneCenter,718],
         [leftLaneCenter,0]])

    # Compute the perspective transform, M, given source and destination points
    M = cv2.getPerspectiveTransform(src, dst)

    # Compute the inverse perspective transform
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    # Warp an image using the perspective transform, M
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    # Return the resulting image and matrix
    return warped, M , Minv


#cv2.destroyAllWindows()
calcDistortion = True
calcWarp = True

# Test undistortion on an image
if calcDistortion == True:
    #img = cv2.imread('calibration_wide/test_image.jpg')
    img = cv2.imread('../camera_cal/calibration13.jpg')

    img_size = (img.shape[1], img.shape[0])

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)


    #dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    #cv2.imwrite('../camera_cal/test_undistored_calibration13.jpg',dst)



if calcWarp == True:
    img = cv2.imread('../test_images/straight_lines1.jpg')
    img_dist = cv2.undistort(img, mtx, dist, None, mtx)
    fig = plt.figure(1)
    #To print out position of Mouse if left button clicked
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.imshow(img_dist)
    plt.show()    
    unwarped_image, perspective_M , perspective_Minv = calculateWarp(img,mtx, dist)






# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
# Visualize undistortion
if  calcDistortion == True and calcWarp == True:
    img = cv2.imread('../test_images/straight_lines1.jpg')
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    unwarped_image, perspective_M, perspective_Minv = calculateWarp(img, mtx, dist)

    polygon = Polygon([[703 , 460], [1104 , 718], [203 , 718], [578 , 460]], closed=True, fill=True, linewidth=2,edgecolor='r',facecolor='none')
    rectangle = patches.Rectangle((300,2),600,718,linewidth=2,edgecolor='r',facecolor='none')


    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dst)
    ax2.add_patch(polygon)
    ax2.set_title('Undistorted Image', fontsize=30)
    ax3.imshow(unwarped_image)
    ax3.add_patch(rectangle)
    ax3.set_title('Unwarped Image', fontsize=30)
    plt.show()

    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    dist_pickle["rvecs"] = rvecs
    dist_pickle["tvecs"] = tvecs
    dist_pickle["M"] = perspective_M    
    dist_pickle["Minv"] = perspective_Minv
    dist_pickle["rightLaneCenter"] = rightLaneCenter
    dist_pickle["leftLaneCenter"] = leftLaneCenter
 
    pickle.dump( dist_pickle, open( "../camera_cal/camera_calibration_pickle.p", "wb" ) )
