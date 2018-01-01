# **Behavioral Cloning** 

## Self-Driving-Car Project3
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* deep_cnn_generator.py containing the script to create and train the model
* deep_cnn_generator.ipynb containing the visualization of loss
* deep_cnn_generator.html containing the html format of the ipynb code
* drive.py for driving the car in autonomous mode
* gen_video.sh for creating videos from images using `ffmpeg`
* data.mp4 the video of training data
* run2.mp4 the recorded video of autonomous mode
* model_deep_cnn_submitted.h5 containing a trained convolution neural network 
* report.md or report.html summarizing the results. I can view the videos in this md file using `retext` but github cannot display videos. You could watch the videos using the html file or just open videos using a media plyer.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing

```sh
python drive.py model_deep_cnn_submitted.h5
```

#### 3. Submission code is usable and readable

The deep_cnn_generator.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 or 3x3 filter sizes and depths between 24 and 64 (deep_cnn_generator.py lines 77-83) 

The model includes RELU layers to introduce nonlinearity (code line 77-83), and the data is normalized in the model using a Keras lambda layer (code line 75). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 80, 82, 84, 87, 89). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 94-96). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I also flipped left right of all images to augment the data set.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the [nvidia deep cnn](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) I thought this model might be appropriate because nvidia adoppted it for real autonomous driving.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that there is a dropout layer after some deep convolutional layers and fully connceted layers.

Then I subsampled the dataset, _i.e.,_, using a subset of original dataset. In details, I used every 6th image of the original image data set, because several images that are continous contain similar information. This subsampling method can also reduce training time.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I recorded more data:

    * two counter clockwise laps and one clockwise lap of center lane driving
    * one lap of recovery driving from the sides
    * one lap focusing on driving smoothly around curves


At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (deep_cnn_generator.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:

| layer         | sizes         |
|:-------------:|:-------------:|
| lambda        | 3, 120*360    |
| cropping2d    | (70,25),(0,0) |
| Conv2D        | 24, 5x5       |
| Conv2D        | 36, 5x5       |
| Conv2D        | 48, 5x5       |
| Dropout       |               |
| Conv2D        | 64, 3x3       |
| Dropout       |               |
| Conv2D        | 64, 3x3       |
| Dropout       |               |
| Flatten       |               |
| FullyConnected| 100           |
| Dropout       |               |
| FullyConnected| 50            |
| Dropout       |               |
| FullyConnected| 10            |
| FullyConnected| 1             |


#### 3. Creation of the Training Set & Training Process


To capture good driving behavior, I recorded:

    * one counter clockwise lap of center lane driving
    * one clockwise lap of center lane driving
    * driving from right side to center
    * driving from left side to center
    * driving smooth turns
    * one counter clockwise lap of center lane driving

I created a video using `ffmpeg` from recorded centering images. Here is the video:
<div style="text-align:center"><video width="400" controls><source src ="./data.mp4" | absolute_url}}' />Your browser does not support HTML5 video.</video></div>

After the collection process, I had 9633(x6 using 3 cameras and flipping left right) of data points. I then preprocessed this data by normalization, mean centering, and cropping top environments and bottom car hoods.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the fact that validation loss was larger than training loss after epoch 5. I used an adam optimizer so that manually training the learning rate wasn't necessary.

The following is the recorded result:

<div style="text-align:center"><video width="400" controls><source src ="./run2.mp4" | absolute_url}}' />Your browser does not support HTML5 video.</video></div>
