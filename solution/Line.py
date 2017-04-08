import pickle
import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os.path

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self,orientation):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = [] 
        #y values for detected line pixels
        self.ally = []

        self.coeff_history = []

        # orientation can be left or right lane
        self.orientation = orientation

    def processLanePts(self, x_pts, y_pts,img_shape):
        print('-------------------------------------')
        print('Process pts on Line ' , self.orientation)
        #print('Processing number of x_pts points' , len(x_pts))
        #print('Processing number of y_pts points' , len(y_pts))
        # Fit a second order polynomial to each
        lane_fit = np.polyfit(y_pts, x_pts, 2)


        # Generate x and y values for plotting
        ploty = np.linspace(0, img_shape[0]-1, img_shape[0] )
        lane_fitx = lane_fit[0]*ploty**2 + lane_fit[1]*ploty + lane_fit[2]

        #print("Coeff a" ,lane_fit[0])
        #print("Coeff b" ,lane_fit[1])
        #print("Coeff c" ,lane_fit[2])       

        if len(self.coeff_history)>0:
            mean_coeff = np.mean(self.coeff_history, axis=0)
        else:
            #first run! 
            self.coeff_history.append(lane_fit)
            mean_coeff = np.mean(self.coeff_history, axis=0)

        #print("Mean Coeff a" ,mean_coeff[0])
        #print("Mean Coeff b" ,mean_coeff[1])
        #print("Mean Coeff c" ,mean_coeff[2])

        relative_coeff_a_change = abs((lane_fit[0] - mean_coeff[0])/mean_coeff[0])
        relative_coeff_b_change = abs((lane_fit[1] - mean_coeff[1])/mean_coeff[1])
        relative_coeff_c_change = abs((lane_fit[2] - mean_coeff[2])/mean_coeff[2])

        if relative_coeff_a_change > 0.1 or relative_coeff_b_change > 0.1 or relative_coeff_c_change > 0.1:
            # Points seem to be invalid
            self.detected = False

            # skip values!
            print('Frame seems to be invalid!!!')
        else:   # valid points found!!
            self.detected = True

            print('Frame seems to be valid!!!')
            # step 1: remove first frame coeffs if more than threshold items available
            if (len(self.coeff_history) >= 5):
                self.coeff_history.pop(0)

            # step 2: append newly found coeffs
            self.coeff_history.append(lane_fit)

            # step 3: append x/y points
            self.allx = x_pts
            self.ally = y_pts

            # step 4: set current fit polynomial coefficients
            self.current_fit = lane_fit

            # step 5: set current fit polynomial coefficients
            self.diffs = [(lane_fit[0] - mean_coeff[0]) , (lane_fit[1] - mean_coeff[1]), (lane_fit[2] - mean_coeff[2])]

            # step 6: set the curvature radius
            # TODO
            #self.radius_of_curvature = None 
    
            #step 7: distance in meters of vehicle center from the line
            # TODO
            #self.line_base_pos = None 



# Define a class to receive the characteristics of each line detection
class LeftLine(Line):
    def __init__(self):
         Line.__init__(self, "left")


# Define a class to receive the characteristics of each line detection
class RightLine(Line):
    def __init__(self):
         Line.__init__(self, "right")

class EgoLane():
    def __init__(self):
        self.leftline = LeftLine()
        self.rightline = RightLine()

    def pipeline(self, frame):
        #print("pipeline")

        img = frame.currentImg

        img_undistort = frame.camera.undistort(img)

        colorGrad = self.colorGradient(img_undistort,(170,220),(22,100))
        warped = frame.camera.warp(colorGrad)
        maskedImage = frame.camera.maskInnerAreaOfInterest(warped)
        grayImage = frame.camera.rgbConvertToBlackWhite(maskedImage)
        
        # plt.imshow(grayImage)
        # plt.title('window fitting results')
        # plt.show()

        self.histoCurvatureFit(grayImage)
        coloredLaneImage = self.displayLane(img)

        unwarped = frame.camera.unwarp(coloredLaneImage)
        return unwarped


    def displayLane(self,img):
        overlay_img = np.zeros_like(img)

        ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
        left_fitx = self.leftline.current_fit[0]*ploty**2 + self.leftline.current_fit[1]*ploty + self.leftline.current_fit[2]
        right_fitx = self.rightline.current_fit[0]*ploty**2 + self.rightline.current_fit[1]*ploty + self.rightline.current_fit[2]

        if self.leftline.detected == True:
            for x1,y1 in zip(left_fitx.astype(int),ploty.astype(int)):
                #print(y1)
                cv2.circle(overlay_img,(x1,y1),2,(255,255, 0),2)

        if self.rightline.detected == True:
            for x1,y1 in zip(right_fitx.astype(int),ploty.astype(int)):
                cv2.circle(overlay_img,(x1,y1),2,(255,255, 0),2)

        if self.rightline.detected == True and self.leftline.detected == True:
            for x1,y1,x2,y2 in zip(left_fitx.astype(int),ploty.astype(int),right_fitx.astype(int),ploty.astype(int)):
                cv2.line(overlay_img,(x1,y1),(x2,y2),(255,0, 0),2)

        return overlay_img/255

    def processFrame(self, frame):
        print("Processing egolane")
        return self.pipeline(frame)

    def colorGradient(self, img, s_channel_thresh=(180, 255), sobel_x_thresh=(30, 120)):
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
        sxbinary[(scaled_sobel >= sobel_x_thresh[0]) & (scaled_sobel <= sobel_x_thresh[1])] = 1
        
        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_channel_thresh[0]) & (s_channel <= s_channel_thresh[1])] = 1
        # Stack each channel
        # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
        # be beneficial to replace this channel with something else.
        color_binary = np.dstack(( sxbinary, np.zeros_like(sxbinary), s_binary))
        return color_binary
    
    def histoCurvatureFit(self, binary_warped):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        height = int(binary_warped.shape[0]/2)
        histogram = np.sum(binary_warped[height:,:], axis=0)

        #histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 

        self.leftline.processLanePts(leftx, lefty, binary_warped.shape)


        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        self.rightline.processLanePts(rightx, righty, binary_warped.shape)


        # # Create an image to draw on and an image to show the selection window
        # out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # window_img = np.zeros_like(out_img)
        # # Color in left and right line pixels
        # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        # left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        # left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        # left_line_pts = np.hstack((left_line_window1, left_line_window2))
        # right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        # right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        # right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # fill the lane with red
        # for x1,y1,x2,y2 in zip(left_fitx.astype(int),ploty.astype(int),right_fitx.astype(int),ploty.astype(int)):
        #     cv2.line(window_img,(x1,y1),(x2,y2),(255,0, 0),2)
        #     cv2.circle(window_img,(x1,y1),2,(255,255, 0),2)
        #     cv2.circle(window_img,(x2,y2),2,(255,255, 0),2)
        

        # Draw the lane onto the warped blank image in green color
        #cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        #cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))

        #result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        # plt.imshow(result/255)
        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')
        # plt.xlim(0, 1280)
        # plt.ylim(720, 0)
        # result = (result/255)


# Define a class to receive the characteristics of each line detection
class Frame():
    def __init__(self):
        self.currentImg = None
        self.currentEgoLaneOverlay = None
        self.egoLane = EgoLane()
        self.camera = None

    def loadImageFromFile(self, filename):
        self.currentImg = cv2.imread(filename)
        self.currentImg = cv2.cvtColor(self.currentImg, cv2.COLOR_RGB2BGR) 

    def processCurrentFrame(self):
        self.currentEgoLaneOverlay = None
        self.currentEgoLaneOverlay = self.egoLane.processFrame(self)

    def displayCurrentImage(self, overlay=True):

        if overlay == True and self.currentEgoLaneOverlay != None:
            print("Show Overlay")
            
            img_pipelined = np.uint8(255*self.currentEgoLaneOverlay/np.max(self.currentEgoLaneOverlay))
            result = cv2.addWeighted(self.currentImg.astype(int), 1, img_pipelined.astype(int), 0.5, 0,dtype=cv2.CV_8U)
            
            plt.imshow(result)
        else:
            print("No Overlay!")
            plt.imshow(self.currentImg)

        plt.title('Input Image')
        plt.show()

    def receiveFrame(self, img):
        self.currentImage = copy.copy(img)

    def initializeCamera(self, fileName='../camera_cal/camera_calibration_pickle.p'):
        self.camera = Camera(fileName)



class Camera():
    def __init__(self,fileName):
        self.mtx = None
        self.dist = None
        self.M = None
        self.Minv = None
        self.rightLaneCenter = None
        self.leftLaneCenter = None
        self.calibrationFileName = fileName

        if os.path.isfile(fileName) == True:
            self.loadCameraCalibration()
        else:
            print("No Camera calibration found.... Exiting().....")
            #TODO: auto-calibrate
            exit()

    def loadCameraCalibration(self):
        # Read in the saved camera matrix and distortion coefficients
        # These are the arrays you calculated using cv2.calibrateCamera()
        dist_pickle = pickle.load( open(self.calibrationFileName , "rb" ) )
        self.mtx = dist_pickle["mtx"]
        self.dist = dist_pickle["dist"]
        self.M = dist_pickle["M"]
        self.Minv = dist_pickle["Minv"]
        self.rightLaneCenter = dist_pickle["rightLaneCenter"]
        self.leftLaneCenter = dist_pickle["leftLaneCenter"]    

    def warp(self, img):
        img_size = (img.shape[1], img.shape[0])
        warped = cv2.warpPerspective(img, self.M, img_size,flags=cv2.INTER_LINEAR)
        return warped

    def unwarp(self, img):
        img_size = (img.shape[1], img.shape[0])
        unwarped = cv2.warpPerspective(img, self.Minv, img_size,flags=cv2.INTER_LINEAR)
        return unwarped

    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

    def maskInnerAreaOfInterest(self, img,maskrange=150):

        imgshape = img.shape
        leftLaneArea = [self.leftLaneCenter-maskrange, self.leftLaneCenter+maskrange]
        rightLaneArea = [self.rightLaneCenter-maskrange, self.rightLaneCenter+maskrange]

        # remove left area
        contours = np.array( [ [0,0], [leftLaneArea[0],0], [leftLaneArea[0],imgshape[0]], [0,imgshape[0]] ] )
        cv2.fillPoly(img, pts =[contours], color=(0,0,0))

        # remove right area
        contours = np.array( [ [imgshape[1],0], [rightLaneArea[1],0], [rightLaneArea[1],imgshape[0]], [imgshape[1],imgshape[0]] ] )
        cv2.fillPoly(img, pts =[contours], color=(0,0,0))

        return img

    def rgbConvertToBlackWhite(self, rgb, thresh=10):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        
        s_binary = np.zeros_like(r)
        s_binary[(r >= thresh)] = 255

        return (np.logical_or(r,b)*255)


def testLine():
    testFrame = Frame()
    testFrame.initializeCamera()

    testFrame.loadImageFromFile('../test_images/test1.jpg')
    testFrame.processCurrentFrame()
    testFrame.displayCurrentImage()

    testFrame.loadImageFromFile('../test_images/test2.jpg')
    testFrame.processCurrentFrame()
    testFrame.displayCurrentImage()


testLine()
