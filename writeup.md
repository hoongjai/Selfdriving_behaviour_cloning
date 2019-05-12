# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)
[image2]: ./examples/center_2019_05_11_03_45_51_061.jpg "Center Image"
[image3]: ./examples/center_2019_05_11_03_46_52_120_recovery1.jpg "Recovery Image"
[image4]: ./examples/center_2019_05_11_03_46_55_033_recovery2.jpg "Recovery Image"
[image5]: ./examples/center_2019_05_11_03_46_57_147_recovery3.jpg "Recovery Image"
[image6]: ./examples/center_2019_05_11_03_46_52_120_recovery1.jpg "Normal Image"
[image7]: ./examples/flipped_image.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
I am using the famous Nvidia Convoluation Neural Network which was recommended and proven working.
I added a dropout rate of 0.25 after layer 6 to solved the overfitting issue.

I splitted the data set to training data and validation data to ensure that the model is not learning/memorize the dataset, which give inaccurate validation.

The layer architecture:
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 70, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 33, 158, 24)       1824      
_________________________________________________________________
activation_1 (Activation)    (None, 33, 158, 24)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 15, 77, 36)        21636     
_________________________________________________________________
activation_2 (Activation)    (None, 15, 77, 36)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 6, 37, 48)         43248     
_________________________________________________________________
activation_3 (Activation)    (None, 6, 37, 48)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 4, 35, 64)         27712     
_________________________________________________________________
activation_4 (Activation)    (None, 4, 35, 64)         0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 2, 33, 64)         36928     
_________________________________________________________________
activation_5 (Activation)    (None, 2, 33, 64)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 4224)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               422500    
_________________________________________________________________
activation_6 (Activation)    (None, 100)               0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 100)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
activation_7 (Activation)    (None, 50)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
activation_8 (Activation)    (None, 10)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 559,419
Trainable params: 559,419
Non-trainable params: 0
_________________________________________________________________

#### 2. Attempts to reduce overfitting in the model
My model contained dropout function to reduce overfitting. The implementation can be found in model.py build_model() function.

#### 3. Model parameter tuning
I used adam optimizer where manual tuning of learning rate is not required.
I use Epoch = 5 for the training and the training time was approximately 25mins x 5 = 125mins = 2 hours.
  
#### 4. Appropriate training data
Training data was specially selected to keep the vehicle stay on track. To do this, I cropped the top and bottom of the images to remove the sky and car engine cover.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach
My first step was to use Nvidia Convolution Neural Network. I thought this model might be appropriate because it is widely used and proven to be working on behaviour cloning.
    
I found that my model had an overfitting issue. To fix the overfitting issue I added a dropout function after layer 6.
Few more tricks that I used to improved my model:
- Cropped the images to removed distractions like sky, engine cover, tree, building and etc.
- Augmented the images by flipping the image to take opposite sign measurement (images shown below).
- Transfer learning where the model was train by loading previous good model.
- Generator to train in batch to reduce memory comsumption.

#### 2. Final Model Architecture
The finial model architecture consisted of following layeras and layer sizes.
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 70, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 33, 158, 24)       1824      
_________________________________________________________________
activation_1 (Activation)    (None, 33, 158, 24)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 15, 77, 36)        21636     
_________________________________________________________________
activation_2 (Activation)    (None, 15, 77, 36)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 6, 37, 48)         43248     
_________________________________________________________________
activation_3 (Activation)    (None, 6, 37, 48)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 4, 35, 64)         27712     
_________________________________________________________________
activation_4 (Activation)    (None, 4, 35, 64)         0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 2, 33, 64)         36928     
_________________________________________________________________
activation_5 (Activation)    (None, 2, 33, 64)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 4224)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               422500    
_________________________________________________________________
activation_6 (Activation)    (None, 100)               0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 100)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
activation_7 (Activation)    (None, 50)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
activation_8 (Activation)    (None, 10)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 559,419
Trainable params: 559,419
Non-trainable params: 0
_________________________________________________________________

#### 3. Creation of the Training Set & Training Process
At first I used the training data provided by Udacity and it worked well until I encounter the first sharp corner after the bridge... 
Next I drove the simulator myself and collected two laps of data on track one using center lane driving. Here is an example image of center lane driving:
![alt text][image2]

Next I recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recovery from leaving the track. These images below show what a recovery looks like:
![alt text][image3]
![alt text][image4]
![alt text][image5]
        
Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would help taking the opposite curve measurement. 
For example, here were images that has then been flipped:
![alt text][image6]
![alt text][image7]

Final step was randomly shuffled the data set and put 15% of the data into a validation set. 

I used this training data for training the model while the validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5. I used an adam optimizer so that manually training the learning rate wasn't necessary.
