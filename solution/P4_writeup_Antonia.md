#**Advanced Lane Finding**


**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


[//]: # (Image References)

[image0]: ./../camera_cal/calibration1.jpg "calibration1.jpg"
[image1]: ./../output_images/undistored_calibration1.jpg "Undistorted calibration1.jpg"
[image2]: ./../test_images/straight_lines1.jpg "Straight Line"
[image3]: ./../output_images/straight_lines1_warp_points.jpg "Straight Line Warp Points"
[image4]: ./../output_images/warp_example.png "Warp Example"
[image5]: ./../test_images/test4.jpg "test4.jpg"
[image6]: ./../output_images/undistort_test4.jpg "Undistorted test4.jpg"
[image7]: ./../docu/Curvature.jpg "Curvature.jpg"
[image8]: ./../docu/Curvature_1.jpg "Curvature_1.jpg"
[image10]: ./../docu/Class_Diagram.JPG "Class Diagram"
[image11]: ./../docu/Smoothing.JPG "Smoothing concept"

[image20]: ./../test_images/img_temp_1.png
[image21]: ./../output_images/challenge_contrastIncrease.jpg
[image22]: ./../output_images/challenge_colorGradient.jpg
[image23]: ./../output_images/challenge_maskAreaOfInterest.jpg
[image24]: ./../output_images/challenge_searchingWindow.jpg
[image25]: ./../output_images/challenge_finalResult.jpg

[image26]: ./../test_images/img_temp_134.png
[image27]: ./../output_images/bridge_contrastIncrease.jpg

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/571/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

* link to Line.py [Pipeline and video processing](https://github.com/AntoniaSophia/CarND-Advanced-Lane-Lines/blob/master/solution/Line.py)
* link to Calibrate_Camera.py [Calibrate the Camera](https://github.com/AntoniaSophia/CarND-Advanced-Lane-Lines/blob/master/solution/Calibrate_Camera.py)  
* link to Jupyter monitor which shows calibration and warp calculation [Notebook](https://github.com/AntoniaSophia/CarND-Advanced-Lane-Lines/blob/master/solution/Advanced_Lane_Lines.ipynb)
* link to HTML output of the Jupyter monitor which shows calibration and warp calculation [Notebook HTML](https://github.com/AntoniaSophia/CarND-Advanced-Lane-Lines/blob/master/output_images/Advanced_Lane_Lines.html)
* link to the annotated output video of the project_video.mp4 at [Project video](https://github.com/AntoniaSophia/CarND-Advanced-Lane-Lines/blob/master/output_videos/project_video.mp4)
* link to the annotated output video of the challenge_video.mp4 at [Challenge video](https://github.com/AntoniaSophia/CarND-Advanced-Lane-Lines/blob/master/output_videos/challenge_video.avi)

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file [Calibrate the Camera](https://github.com/AntoniaSophia/CarND-Advanced-Lane-Lines/blob/master/solution/Calibrate_Camera.py) or in the Jupyter notebook [Notebook](https://github.com/AntoniaSophia/CarND-Advanced-Lane-Lines/blob/master/solution/Advanced_Lane_Lines.ipynb) in the cells 2,3,4 and escpecially 5. 
What I've done is basically to load all chessboard images and let 9x6 corners inside the board been identfied. 

I start by preparing "object points" (variable `objpoints` in cell 2), which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objpoints` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  The variable `imgpoints` (see also cell 2) will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

The original image looks like: ![Orginial chessboard][image0]
The corresponding undistorted image looks like ![Undistorted chessboard][image1]

In the file [Calibrate the Camera](https://github.com/AntoniaSophia/CarND-Advanced-Lane-Lines/blob/master/solution/Calibrate_Camera.py) I continue to calculate the warp starting at cell 6: I added a click-event listener and used an image containing a straight line:
![Straight Line Image][image2]

Afterwards I've identified four points in that image by simply clicking on the pixels to identify the warp points
![Straight Line Image][image3]



The code for my perspective transform is contained in cell 8 of the Juypter notebook and containes a function called calculateWarp(). This function calculateWarp() takes as inputs an image (`img`), as well the distortion matrix of the camera.  I chose the hardcode the source and destination points in the following manner:

```
leftLaneCenter = 300
rightLaneCenter = 900

upper_right_click = (703 , 460)
lower_right_click = (1104 , 718)
lower_left_click = (207 , 718)
upper_left_click = (578 , 460)

src = np.float32(
    [upper_right_click,
     lower_right_click,
     lower_left_click,
     upper_left_click])

dst = np.float32(
    [[rightLaneCenter,0],
     [rightLaneCenter,img_dist.shape[0]],
     [leftLaneCenter,img_dist.shape[0]],
     [leftLaneCenter,0]])

```
This resulted in the following source and destination points:


| Source        | Destination   | 
|:-------------:|:-------------:| 
| 703 , 460      | 300, 0        | 
| 1104 , 718     | 300, 720      |
| 207 , 718      | 900, 720      |
| 578 , 460      | 900, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Warp Example][image4]

The resulting values I store in a pickle file, see cell 10 in the Jupyter notebook [Notebook](https://github.com/AntoniaSophia/CarND-Advanced-Lane-Lines/blob/master/solution/Advanced_Lane_Lines.ipynb)

###Pipeline (single images)

The file Line.py [Pipeline and video processing](https://github.com/AntoniaSophia/CarND-Advanced-Lane-Lines/blob/master/solution/Line.py) contains a class structure which I used for processing images and the pipeline.
![ClassDiagram][image10]
A class Frame contains a class Camera which keeps all relevant Camera functions like loadCalibration(), undistort(), warp(), unwarp(),... Additionally the Frame contains a class EgoLane which itself contains each an object LeftLine and RightLine. Both objects LeftLine and RightLine inherit from a common base class called Line.
The pipeline is implemented in the class EgoLane starting from line 291 contains all in all 13 steps

```
        #1.Step: take the modified image after contrastIncrease()
        img = frame.modifiedImg

        #2. Step: undistort this image
        img_undistort = frame.camera.undistort(img)

        #3. Step: apply the color gradient
        colorGrad = self.colorGradient(img_undistort,(170,220),(22,100))

        #4. Step: Warp the image
        warped = frame.camera.warp(colorGrad)

        #5.Step: mask the area of interest
        maskedImage = frame.camera.maskAreaOfInterest(warped)

        #6.Step: convert to black/white
        grayImage = frame.camera.rgbConvertToBlackWhite(maskedImage)
        
        #7.Step: in case we have nothing detected yet --> detect newly
        #   in case we have already detected lines --> detect from this base
        if self.leftline.detected == True and self.leftline.detected == True:
            histoCurvatureFitImage = self.nextFramehistoCurvatureFit(grayImage)
            #histoCurvatureFitImage = self.histoCurvatureFit(grayImage)
        else:
            histoCurvatureFitImage = self.histoCurvatureFit(grayImage)


        # 8.Step Now display the found lines and plot them on top of the original image
        coloredLaneImage = self.displayLane(frame.currentImg)
       

        #9.Step Now add a small resized image of the curvature calculation in the upper middle of the original image
        grayImage = np.uint8(grayImage)
        gray2color = cv2.cvtColor(grayImage,cv2.COLOR_GRAY2RGB ,3)
        gray2color = cv2.addWeighted(coloredLaneImage.astype(np.float32)*255, 1, (gray2color.astype(np.float32))*255, 1, 0)
        resized_image = cv2.resize(gray2color,None,fx=0.3, fy=0.3, interpolation = cv2.INTER_AREA)

        #10.Step Unwarp the whole image
        unwarped = frame.camera.unwarp(coloredLaneImage)*255
        src_mask = mask = 255 * np.ones(resized_image.shape, resized_image.dtype)
        unwarped = cv2.seamlessClone(resized_image.astype(np.uint8), unwarped.astype(np.uint8), src_mask.astype(np.uint8), (640,200), cv2.NORMAL_CLONE)


        #11.Step: add additional text left/right which might be interesting
        if displayText==True:
            new_image = np.zeros_like(unwarped)

            text1 = "Left Lane Dropout Counter: " + str(self.leftline.number_of_subsequent_invalid)
            text1a = "Left Lane Points found: " + str(len(self.leftline.allx))
            text2 = "Right Lane Dropout Counter: " + str(self.rightline.number_of_subsequent_invalid)
            text2a = "Right Lane Points found: " + str(len(self.rightline.allx))
            text3 = "Curvature radius left: " + str(self.leftline.radius_of_curvature) + " (m)"
            text4 = "Curvature radius right: " + str(self.rightline.radius_of_curvature) + " (m)"

            center_deviation = round((self.leftline.line_base_pos*xm_per_pix*100 + self.rightline.line_base_pos*xm_per_pix*(-100))/2,2)

            if center_deviation >=0:
                text5 = "Vehicle is left of center " + str(center_deviation) + ' (cm)'
            else:
                text5 = "Vehicle is right of center " + str(center_deviation) + ' (cm)'

            text6 = text5
            text7 = "RESTART! " 
            text8 = "ADAPTIVE!!"

            cv2.putText(new_image,text1,(50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,255),1,cv2.LINE_AA)
            cv2.putText(new_image,text1a,(50,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,255),1,cv2.LINE_AA)
            cv2.putText(new_image,text2,(900,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,255),1,cv2.LINE_AA)
            cv2.putText(new_image,text2a,(900,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,255),1,cv2.LINE_AA)
            cv2.putText(new_image,text3,(50,90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,255),1,cv2.LINE_AA)
            cv2.putText(new_image,text4,(900,90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,255),1,cv2.LINE_AA)
            cv2.putText(new_image,text5,(50,110), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,255),1,cv2.LINE_AA)
            cv2.putText(new_image,text6,(900,110), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,255),1,cv2.LINE_AA)
            
            if (self.leftline.number_of_subsequent_invalid == 5 or self.rightline.number_of_subsequent_invalid == 5):
                cv2.putText(new_image,text7,(540,60), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv2.LINE_AA)

            if adaptive == True:
                cv2.putText(new_image,text8,(540,80), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv2.LINE_AA)                

            result = cv2.addWeighted(unwarped.astype(np.float32)*255, 1, (new_image.astype(np.float32))*255, 1, 0)
        else:
            result = unwarped/255

        #12. Detect whether we should change preprocessing of that image in order to get more pixels
        if len(self.leftline.allx) < 500:
            adaptive = True
            self.leftline.reset()
        elif len(self.rightline.allx) < 500:
            adaptive = True
            self.rightline.reset()
        else:
            if self.leftline.number_of_subsequent_valid > 0:
                adaptive = False            

        #13.Step finally return the result
```


####1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

The original image ![Test image test4.jpg][image5]

The corresponding undistorted image ![Undistorted test image test4.jpg][image6]


####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

There are basically three methods which I use sequentially:
- line 813: contrastIncrease(): This methods also contains a conversion into HSV color space and masks areas which are too dark for lines (threshold is 170)
- line 437: colorGradient(): The color gradient contains a conversion into HSV color space followed by a Sobel-Operator in x-direction and an appropriate scaling+thresholding
- line 783: maskAreaOfInterest(): Masks the area of interest where I'm searching for the line to appear (cutting left side, right side and middle of the image) 

Here are examples applied to the following original image:
![original image][image20]

The method contractIncrease() creates the following result:
![contractIncrease][image21]

The method colorGradient() creates the following result:
![colorGradient][image22]

The method maskAreaOfInterest() creates the following result:
![maskAreaOfInterest][image23]

In cases where it is too dark and the method contractIncrease() would otherwise produce too less points for polynomial calculation I used an "adaptive mode" which only gets active in cases of dark color conditions
```
mask = cv2.adaptiveThreshold(img_g,170,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
```

Example on the challenge video directly under the bridge:
![contractIncreaseBridge][image26]: 

And this produced the final output image of the function contractIncrease()
![contractIncreaseBridge][image27]: 


####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

see above in the pipeline Step 4. An example you see in the Juyper notebook ![Warp Example][image4]


####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

There are three methods which identify the lane-line pixels and try to calculate the polynomial fit:
- line 471: histoCurvatureFit() : This calculates the most probable lane-line pixels with the technique of sliding window search in case there was no previous lane found
- line 600: nextFramehistoCurvatureFit() : This calculates the most probable lane-line pixels with the technique of sliding window search in case there was already a previous lane found
- line 83: processLanePts() : takes the output of the above two functions and performs a polynomial fit with 2nd order polynomial

Here is an example of the sliding window search (orange = left line pixels, blue = right line pixels, green = sliding window search , yellow = polynomial fit curve)
![slidingWindowSearch][image24]:

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The base class Line has a function called calculateCurvature(). You can find this function in the file Line.py at line 224
In this function the following steps are carried out:
- take the quadratic coefficient after the polynomial fit (the curvature only depends on the quadratic coefficient)
- generate some fake data in order to execute a polynomial fit on that. Distinguish between left line and right line!
- execute the polynomial fit
- define maximum y-value, corresponding to the bottom of the image
- define conversion ratios between pixel space and real world space (in meters)
- execute the polynomial fit again, but this time in world space
- calculate the radius according to the following formula
![Radius Calculation][image7]
![Radius Calculation_1][image8]

```
def calculateCurvature(self):
    #1.Step: Generate some fake data to represent lane-line pixels
    ploty = np.linspace(0, 719, num=720)# to cover same y-range as image
    quadratic_coeff = self.current_fit[0] 

    #2.Step: For each y position generate random x position within +/-50 pix
    # of the line base position in each case (x=300 for left, and x=900 for right)
    if self.orientation == 'left':
        points_x = np.array([300 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) 
                                  for y in ploty])
    else:
        points_x = np.array([900 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) 
                                    for y in ploty])

    points_x = points_x[::-1]  # Reverse to match top-to-bottom in y

    #3.Step: Execute the polynomial fit: Fit a second order polynomial to pixel positions in each fake lane line
    points_x_fit = np.polyfit(ploty, points_x, 2)
    fitx = points_x_fit[0]*ploty**2 + points_x_fit[1]*ploty + points_x_fit[2]

    #4. Step:  Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    #5. Step: Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    #6. Step Fit new polynomials to x,y in world space
    fit_cr = np.polyfit(ploty*ym_per_pix, fitx*xm_per_pix, 2)

    #7. Step: Calculate the new radii of curvature
    curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])

    #8. Step: return the result
    return curverad.astype(int)
```


The position of the vehicle with respect to center is calculated with the following steps:
- after polyfit calculate the x-values at the bottom on the image
- depending on the line (LeftLine or RightLine) calculate the deviation from the lane center in pixels
- transform the pixel deviation into real world space, convert to centimeters and take the mean between the LeftLine deviation and the RightLine deviation

```
#1.Step: calculate the bottom x-values after polyfit
center_line_point = lane_fit[0]*720**2 + lane_fit[1]*720 + lane_fit[2]

#2. Step: distance in meters of vehicle center from the line
if self.orientation is 'left':
    self.line_base_pos = self.center_line_point - 300
else:
    self.line_base_pos = 900 - self.center_line_point                 

#3. Step Transform into the real-world space and take the mean between the leftline deviation and the rightline deviation
center_deviation = round((self.leftline.line_base_pos*xm_per_pix*100 + self.rightline.line_base_pos*xm_per_pix*(-100))/2,2)
```

The code is split across the file Line.py, e.g line 118, 200-205, 348


####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 393-431 code in function displayLane().  Here is an example of my result on a test image:
![challengeFinalResult][image25]


---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my project video result](./../output_videos/project_video.mp4)
And finally here's a [link to my challenge video result](./../output_videos/challenge_video.mp4)

The project video looks quite good, whereas the challenge video has some problems driving under the brigde in the shadow. I should have invested more time in finding out on how to solve the problems....See at least the next section for some discussion on ideas.

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Writing the overall class structure, the pipeline and a "first shot" was pretty easy. However the problems started when I tried to get my solution running for different conditions.
I was playing around with Sobel, Thresholds, Magnitudes, HistogramEqualization, Color Spaces,.... - but at the end I feel there is still a lot of work to do in order to make this pipeline work for most common conditions.

What I've done so far:
- creating an extensible class structure in Line.py
- coefficient smoothing (take the average in the polynomial coefficients of the last n occurences of lines), see picture
![coefficientSmooting][image11]
- implementing a method contrastIncrease() for thresholding in order to filter preferably white and yellow color 
- implemented contractIncrease() function which is able to adapt to different conditions when e.g. the numbers of pixels for polynomial fit is too less 
- next curvature fit (line 600: nextFramehistoCurvatureFit()) in order to start with the already found polynomial fit and increase search speed as well as having a better starting point for the sliding window search
- implemented a restart() mechanism in order to initiate a kind of "healing" process when lines are not found anymore (e.g. the polynomial fit history was bad)

Further work:
- my pipeline is not good when conditions are dark (lots of shadows!) or very bright (contractIncrease() doesn't deal with this yet)
- implement a real shadow removal, I got the following hints for techniques to achieve it (see below)
- problems after the first shadow area when no line points have been detected   docu/img_overlay_586.png
- improve contrastIncrease() to be much more adaptive - not only for the situation which I covered (when no pixels are found at all)


########################################



Further hints I got from my mentor:
- https://infoscience.epfl.ch/record/111781/files/piecewise_shadows.pdf
- more complex - http://aqua.cs.uiuc.edu/site/files/cvpr11_shadow.pdf
- ML on the bird's eye view https://carnd-forums.udacity.com/questions/33788268/an-experiment-using-deeplearning-for-advanced-lane-finding

