#**Behavioral Cloning**


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image4]: ./docu/model.png "Model"
[image5]: ./docu/model_summary.png "Model summary"
[image6]: ./docu/data_distribution.png "Data distribution"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

* link to model.py [Model File](https://github.com/AntoniaSophia/Behavioral-Cloning_Project3/blob/master/solution/model.py)
* link to drive.py [Drive Control](https://github.com/AntoniaSophia/Behavioral-Cloning_Project3/blob/master/solution/drive.py)  
* link to final network model.h5 [Network file](https://github.com/AntoniaSophia/Behavioral-Cloning_Project3/blob/master/solution/model.h5)
* link to the best initial model (see also "Solution Design approach") [TOP initial model](https://github.com/AntoniaSophia/Behavioral-Cloning_Project3/blob/master/solution/model_TOP_FOR_RETRAIN.h5)
* link to the video of the first track videos_track1.mp4 [Video Track 1](https://github.com/AntoniaSophia/Behavioral-Cloning_Project3/blob/master/solution/videos/videos_track1.mp4)
* link to the video of the first track videos_track1.mp4 [Video Track 2](https://github.com/AntoniaSophia/Behavioral-Cloning_Project3/blob/master/solution/videos/videos_track2.mp4)
* link to the HD video of the first track 'Videos Track1 Fullhd-1.m4v' [Video Track 1 HD](https://github.com/AntoniaSophia/Behavioral-Cloning_Project3/blob/master/solution/videos/Videos%20Track1%20Fullhd-1.m4v)
* link to the HD video of the first track 'Videos Track2 Fullhd-1.m4v' [Video Track 2 HD](https://github.com/AntoniaSophia/Behavioral-Cloning_Project3/blob/master/solution/videos/Videos%20Track2%20Fullhd-1.m4v)
* link to the exploration of the training data and network [Exploration of test data](https://rawgit.com/AntoniaSophia/Behavioral-Cloning_Project3/master/solution/docu/Behavioral_Cloning.html)

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* P3_writeup_Antonia.md (this file) summarizing the results
* videos_track1.mp4 which shows the car driving autonomously on the first track
* videos_track2.mp4 which shows the car driving autonomously on the second track


####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```


####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 24 and 64 (model.py lines 201-208) 
The model includes RELU layers to introduce nonlinearity (code lines 201, 203, 205), and the data is normalized in the model using a Keras lambda layer (code line 198). 

Actually is it the standard NVidia network which was shown during the Udacity lessons. 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 202, 204, 206). 

Other approaches to avoid overfitting are:
- training data of the tracks clockwise and anti-clockwise 
- limited epochs (maximum 3-4)
- 20% validation data (see model.py line 185)

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 189-190). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer with standard parameters, so the learning rate was not tuned manually (see model.py line 220).
I didn't experience at all with different networks or modifying the NVidia network. I adjusted parameters like:
- number of epochs
- rates of augmented data
- dropout layers (where and which dropout rate)
- color spaces (at the RGB worked best for me)

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and clockwise/anticlockwise driving direction. Also certain curves of the second track have been recorded 3-4 times seperately in order to get this tricky curves trained sufficiently

For details about how I created the training data, see the next section. 




###Model Architecture and Training Strategy

####1. Solution Design Approach

I used a 2-level approach:
- first try to find a model with limited data which is able to detect lanes and lane markings correctly ("initial model")
- secondly I trained this model using heavy random shadowing and translation, but kept the convolutional layers which already have proven to be able to have understood how to drive 

So actually the first model identified edges, lane markings, obstacles, curves and knows that it shall keep in the middle of the road.
This knowledge is stored in the weights of the convolutional layers of this initial layer and thus it works perfectly fine to cut only the last non-convolutional layers and train them to identify shadows and other random noise (e.g. translation)
Actually I hope this understanding is really correct - but it seemed to work... ;-)

Find the "initial model" at [Best Initial Model](https://github.com/AntoniaSophia/Behavioral-Cloning_Project3/blob/master/solution/model_TOP_FOR_RETRAIN.h5)

This initial model had a low mean squared error both on the training set and on the validation set. This implied that the model was not overfitting. 

The initial model contains dropout layers in order to reduce overfitting (model.py lines 202, 204, 206), the retraining model also contained a dropout layer of flattening of the model (see model.py line 264). 

Other approaches to avoid overfitting are:
- training data of the tracks clockwise and anti-clockwise 
- limited epochs (maximum 3-4)
- separate 20% of the validation data (see model.py line 185)

The final step was to run the simulator to see how well the car was driving around track one. This already worked out pretty well with the initial model. There were a few spots where the vehicle fell off the track in the second track. In order to improve the driving behavior in these cases, I just recorded additional training data containing only those parts and repeating this 3-4 times.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

I simply chose the original NVidia network without any change. However I added some dropout layers in the convolutional part and one dropout after flattening. All in all I experienced that using dropouts was a real nightmare: I found out that it makes only sense in case you have many weights involved, otherwise the network get unstable were quickly. Ok, this is more or less clear! On the other hand I also experienced that the border between "stable driving behavior" and "vehicles misses a curve" is extremely narrow. At the end the dropout rates which worked best were between 0.05 and 0.15 only. Shouldn't dropouts have better effects!?

Also the retraining model is identical, only difference is that the dropout layers of the initial model have been removed of course. This is kind of Frankenstein code starting from line 322...

The model architecture (see model.py lines 196-217) consisted of a convolution neural network with the following layers and layer sizes:
![Model as a table][image5]

Here is a simple graphical visualization of the architecture:


![Model as a small graph][image4]

####3. Creation of the Training Set & Training Process

Training data from both tracks have been taken. In order to capture good driving behavior I took:
- 3 tracks in the original direction
- 2 track in the other direction
- training of recovery (how to get back to center) --> unfortunately I recognized that I also recorded the phase where I drove towards the hard shoulder also which is a silly mistake of course...
- in the second track 3 curves got a special treatment as I recorded them a couple of times as they made trouble
- as additional data source I used the Udacity data 

At the end I had around 33000 images in total. 

The following HTML file shows exploration of the test data including center pictures, flipped pictured, shadowed pictures and the distribution of the angles among the whole test set.

[Exploration of test data](https://rawgit.com/AntoniaSophia/Behavioral-Cloning_Project3/master/solution/docu/Behavioral_Cloning.html)

The following image shows the distribtion of the angles of the test data:
![data distribution][image6]


I used the following techniques in order to augment the test data (see the function image_pipeline in line 101):
- random show works well
- translation also works well
- left camera + right camera plus correction factor

What didn't work at all:
- chosing a different color space than RBG
- darkening the whole image (not only random shadowing)
- creation of too much augmented data also lead to worse results from a certain ratio (maximum 1:2 between real data:augmented data )


General remark:
From my point of view the training data is not really good - I'm not an expert in gaming so I had a tough time driving around escpecially the second track. I was often to harsh in the angle and sometimes I just drove like a drunken pingiun.


Finally I randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3. The reason is simply heuristic: more epochs didn't produce better results and made the difference between training loss and validation even bigger which is and indicator for overfitting. With 3 epochs both values have been pretty close.
I used an adam optimizer so that manually training the learning rate wasn't necessary.

Actually I'm really surprised that the absolute accuracy value doesn't allow a prediction whether the model is good or bad!


####4. Let's give an overall feedback

Thank god I'm done with it.... ;-)

Definitely it was a lot of fun working on that stuff and I'm really proud to succeed in the second track also!!

Again I really learned a lot of things, but on the other hand I feel that even more questions arise after this project:
- how can a neural network be validated? My only criteria was that I tried to observe if the network has understood the rules of center driving
- why are some attempts crazy and some others pretty good?
- there seems to be so much of random inside the approach, is it just because I'm a beginner or is this normal?
- accuracy has nearly no meaning, rather the opposite: networks with the smallest loss had the worst results according to my observation - really!?
- using dropouts was a real nightmare: I found out that it makes only sense in case you have many weights involved, otherwise the network get unstable were quickly. On the other hand I also experienced that the border between "stable driving behavior" and "vehicles misses a curve" is extremely narrow. At the end the dropout rates which worked best were between 0.05 and 0.15 only. Shouldn't dropouts have better effects?
- are there any kind of best practices on how to get to good models in a systematic way? At least for me it was a high amount of try&error

It was really great fun to work on this project!!

