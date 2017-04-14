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
[image3]: ./../test_images/straight_lines1_warp_points.jpg "Straight Line Warp Points"
[image4]: ./../test_images/warp_example.png "Warp Example"
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

The resulting values I store in a pickle file, see cell 10 in the Jupyter notebook

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]


####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

see above 


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
-- https://infoscience.epfl.ch/record/111781/files/piecewise_shadows.pdf
-- more complex - http://aqua.cs.uiuc.edu/site/files/cvpr11_shadow.pdf
- ML on the bird's eye view https://carnd-forums.udacity.com/questions/33788268/an-experiment-using-deeplearning-for-advanced-lane-finding
