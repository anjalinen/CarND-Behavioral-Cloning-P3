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

My model consists of a convolution neural network with filter sizes from 24 to 64 and depths 15 (model.py lines 65-76) 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 65). 

#### 2. Attempts to reduce overfitting in the model

The weights were kept constant to reduce overfitting. Also different data augmentation techniques like flipping images and using left and right images helps with reducing overfitting.
The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually (model.py line 77).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and adjusted the steering angle for the left at 0.2 and the right at -0.2

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to make sure that it drives in autonomous mode with going off the road.

My first step was to use a convolution neural network model similar to the NVIDIA model, I thought this model might be appropriate because it has been tested before and works on autonomous cars. I tried a few iterations and was able to reduce it to less layers.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set (80% and 20%). I initially did the training with only 5 epochs and then again with 10 epochs. The epochs are relatively a small number to avoid overfitting.

I had to tweak the cropping to avoid the car going off the road after the bridge.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 65-77) consisted of a convolution neural network with the following layers and layer sizes:

![model architecture](./nvidia.png)

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded 2 laps on track one using center lane driving. Here is an example image of center lane driving:

![center lane](./train_data/center_2019_09_01_02_47_57_972.jpg)

![left lane](./train_data/left_2019_09_01_02_47_57_972.jpg)

![right lane](./train_data/right_2019_09_01_02_47_57_972.jpg)

I randomly shuffled the data (line 35) set and put 20% of the data into a validation set. 

The final video of the autonomous mode can be found here:

[![Watch on Youtube](./train_data/center_2019_09_01_02_48_57_171.jpg)](https://youtu.be/OI-w--BwRSk)
