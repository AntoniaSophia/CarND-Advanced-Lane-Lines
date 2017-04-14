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

[image1]: ./../output_images/undistorted_calibration13.png "Model"
[image2]: ./../test_images/straight_lines1.jpg "Straight Line"
[image3]: ./../output_images/straight_lines1_warp_points.jpg "Straight Line Warp Points"
[image4]: ./../output_images/warp_example.png "Warp Example"
[image10]: ./../docu/Class_Diagram.JPG "Class Diagram"
[image11]: ./../docu/Smoothing.JPG "Smoothing concept"

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

![Undistorted chessboard][image1]

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
The pipeline is implemented in the class EgoLane starting from line 279 contains all in all 13 steps

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
![alt text][image2]


####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

see above in the pipeline Step 4. An example you see in the Juyper notebook ![Warp Example][image4]


####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./../results/project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?


coefficient smoothing
class structure which is easily extensible
histogram adaptation after YUV transformation on the Y-channel did not work so far
contrastIncrease (thresholding in order to filter white and yellow color)
next curvature fit
double the line finding process ("overlay of the overlay") lines 646,647 - consumes nearly twice processing time, but gives very good results


Further work:
- implement a real shadow removal, I got the following hints for techniques to achieve it 
- problems after the first shadow area when no line points have been detected   docu/img_overlay_586.png
- improve contrastIncrease
Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  



Hints I got from my mentor:
- https://infoscience.epfl.ch/record/111781/files/piecewise_shadows.pdf
- more complex - http://aqua.cs.uiuc.edu/site/files/cvpr11_shadow.pdf
- ML on the bird's eye view https://carnd-forums.udacity.com/questions/33788268/an-experiment-using-deeplearning-for-advanced-lane-finding

