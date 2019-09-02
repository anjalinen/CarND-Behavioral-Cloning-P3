# **Behavioral Cloning** 

## Project Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

*  model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* run1.mp4 showing the video of the autonomous mode running on track 1

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths 15 (model.py lines 57-63) 

The model includes RELU layers to introduce nonlinearity (code line 59), and the data is normalized in the model using a Keras lambda layer (code line 58). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 60). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually (model.py line 66).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and adjusted the steering angle for the left at 0.4 and the right at -0.2

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to make sure that it drives in autonomous mode with going off the road.

My first step was to use a convolution neural network model similar to the NVIDIA mode, I thought this model might be appropriate because it has been tested before and works on autonomous cars. I tried a few iterations and was able to reduce it to less layers.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set (80% and 20%). I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model to add a dropout layer and adjusted the epochs.

I had to tweak the cropping to avoid the car going off the road after the bridge.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 57-63) consisted of a convolution neural network with the following layers and layer sizes:

* Input 64x64x3
* process image with 32x32x3
* Convulational2D with 15x3x3
* Dropout with 0.4
* Maxpooling2D with 2x2
* Flatter
* Dense


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded 1.5 laps on track one using center lane driving. Here is an example image of center lane driving:

![center lane](./data/IMG/center_2019_08_24_16_55_45_525.jpg)

![left lane](./data/IMG/left_2019_08_24_16_55_45_525.jpg)

![right lane](./data/IMG/left_2019_08_24_16_55_45_525.jpg)

I randomly shuffled the data set and put 20% of the data into a validation set. 

The final video of the autonomous mode can be found here:
![video](./run1.mp4)
