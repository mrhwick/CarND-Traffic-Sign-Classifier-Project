# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./internet-test-images/children_crossing.jpg "Children Crossing"
[image5]: ./internet-test-images/no_entry.jpg "Traffic Sign 2"
[image6]: ./internet-test-images/no_truck_passing.jpg "Traffic Sign 3"
[image7]: ./internet-test-images/right_turn.jpg "Traffic Sign 4"
[image8]: ./internet-test-images/stop.jpg "Traffic Sign 5"

## Rubric Points

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Dataset Exploration

After loading the provided dataset, I aggregated a few key metrics about the dataset. The training set is quite large, with 34799 color image entries. The test set is slightly smaller, with only 12630 color image entries. All of the provided images are already formatted to be 32x32 pixels in shape, with 3 color channels. The classification key provided in the `sign_names.csv` file indicated that there are 43 classes of traffic sign in this dataset.

I created a histogram of the training set by classification label. This revealed that the training set was unbalanced in regards to the number of examples provided for each of the classification labels.

![image](https://cloud.githubusercontent.com/assets/865759/25776214/89e749bc-3286-11e7-8e5c-a2055a58b00b.png)

I also plotted some of the images themselves to get an idea of what they look like in the given format. Here is one example:

![image](https://cloud.githubusercontent.com/assets/865759/25776447/63adb104-328c-11e7-8baf-1c93b925d897.png)

### Dataset Preprocessing

One obvious preprocessing step is to convert the color images into a more compact format that retains the visual features of the images. Grayscale is a good candidate, and it is the conversion which I used in this project. Below is an example image after grayscale conversion:

![image](https://cloud.githubusercontent.com/assets/865759/25776486/3aa6cb64-328d-11e7-8cc9-534a51cb9a2d.png)

The provided training set distribution has clear imbalances between the number of examples for each classification. These imbalances need to be reduced without reducing the available training data. I chose to do this by augmenting the training set to include new copies of slightly-altered images based off of the original set. The first augmentation was performed by rotating the images, and I performed rotations to only the images which needed additional examples to bring the distribution into balance.

Here is the class distribution after the rotation augmentaion:

![image](https://cloud.githubusercontent.com/assets/865759/25776494/83497c2c-328d-11e7-8eab-239d9e1b1800.png)

The next preprocessing step that I performed was further augmentation to generate more new images. This time, the augmentation step generated new images by small translations about the original base images. These translations would provide deviation from the original training set to allow the model to discern features independently of location within the image.

Here is the class distribution after the translation augmentation:

![image](https://cloud.githubusercontent.com/assets/865759/25776517/389c3a74-328e-11e7-9d6b-2f00969ebd3e.png)

### Model Architecture

My final model consisted of the following layers:

|Layer| Description|
|:-------------------------:|:-----------------------------:|
|Input| 32x32x1 Grayscale Image|
|Convolution 5x5| 1x1 stride, valid padding, outputs 28x28x6|
|RELU Activation||
|Max Pooling| 2x2 stride, outputs 14x14x6|
|Convolution 5x5| 1x1 stride, valid padding, outputs 10x10x16|
|RELU Activation||
|Max Pooling| 2x2 stride, outputs 5x5x16|
|Flatten & Concatenate Conv1 and Conv2| Output a single vector of size 1576|
|Fully Connected| Outputs 120|
|RELU Activation||
|Fully Connected| Outputs 84|
|RELU Activation||
|Fully Connected| Outputs 43|

I adapted the LeNet architecture for this problem. I took a few suggestions from [this paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). Particularly the suggestion to concatenate the outputs of each of the two convolutions and send them both into the fully connected layers prior to the classification logits.

# TODO: FINISH THE REST


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


