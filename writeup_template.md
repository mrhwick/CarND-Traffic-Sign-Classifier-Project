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

You're reading it! and here is a link to my [project code](https://github.com/mrhwick/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

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

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used softmax cross entropy as my cost function, and an AdamOptimizer. This minimized the mean of cross entropy over 20 epochs of training, with a batch size of 128 examples. I used a variable learning rate, with the rate beginning at `0.0015` and reducing to `0.0005` after a validation accuracy of 94% was achieved during training.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 93.7% 
* test set accuracy of 92.0%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I chose the LeNet architecture, because it was familiar to me and was designed to solve a similar classification problem using image data as input.

* What were some problems with the initial architecture?
The original architecture did not distiguish well between features of small size. The model performed well enough on classes of signs which had distinction based on large sized features such as the difference between a stop sign and a no-entry sign. This did not hold for classes of signs which had distinction based on small sized features, such as the bicycle-crossing and children-crossing signs.

I also noticed that my training regime would often reach a position of relatively higher accuracy, and then regress back to some lower accuracy position soon thereafter. This indicated some problem related to the learning rate or variations in the distribution of examples within the training set. I expect that there was some amount of overfitting to the frequent classes of examples prior to including augmentation in the data preprocessing steps.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

One major improvement that I made was to concatenate the outputs of both of the convolution layers and submit them as input into the fully-connected layers. This was an intuitive step to provide the fully-connected layers with information about both the large and small sized features of the input. This improved my overall performance, as well as improving the training time necessary to reach convergence upon a relative maxima in accuracy.

Another major improvement I made was to drop down the learning rate after a certain threshold of validation accuracy was achieved. This allowed my learner to stop moving across the maxima, moving more slowly once the accuracy maxima is discovered.

* Which parameters were tuned? How were they adjusted and why?

I tuned some of the layer sizes in the fully-connected layers to accomodate for more information coming from the concatenated convolution outputs. I also tuned the learning rate in a single adaptation once the validation accuracy reached a specified threshold.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

A convolution layer is useful for distinguishing the features of each class of traffic sign regardless of their absolute position within the input. The resolution of each convolution allowed the model to discover features at their respective input scale, with the first convolution discovering quite small features and the second convolution discovering larger features.

Although I did not implement dropout in my model, dropout would decrease overfitting to particular inputs by disallowing certain random groups of nodes within the model from acting together. These "pathways" of mutual activation within the model would be likely, and cause overfitting because a single representation of the class is sufficient to cover the training set. With dropout implemented, the model would be forced to learn multiple representations of the same correct classification filters throughout its structure.

If a well known architecture was chosen:
* What architecture was chosen?

I chose to implement LeNet and then adapt it to the needs of this project's application.

* Why did you believe it would be relevant to the traffic sign application?

I believed that LeNet would be relevant because it involved image recognition/classification. I knew there would be some adjustment necessary because the handwriting classification problem relies more on distinction of a single set of important features (pen strokes). The traffic sign classification problem, on the other hand, includes a much larger amount of variation in the input data structure. In adapting the LeNet model architecture to the traffic sign classification problem, I made careful changes to allow the classifier to act upon multiple groups of features. This adaptation was suggested by the provided [paper authored by Pierre Sermanet and Yann LeCun](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf).

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

The model's accuracy on all three sets indicates that the model performs well on examples to which it has been exposed as well as generalizing well to examples which it has not experienced before in training. The test set accuracy in particular give us good confidence that this model generalizes well to new examples in the same format. I would have liked to pay more attention to differences between training accuracy and validation accuracy to attempt reducing overfitting.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

In each of these images the larger features are fairly easy to distinguish, which works well for my model. In the case of ![alt-text][image-7], there is a particular difficulty of distinguishing its smallest features from the smallest features found on other signs such as the "bicycle-crossing" and "road-work" signs. I would have liked to have trained the model on more examples of these particular classes in order to reduce this ambiguity.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Children Crossing      		| Bicycles Crossing   									| 
| No Entry     			| No Entry 										|
| No Truch Passing					| No passing for vehicles over 3.5 metric tons 											|
| Turn Right Ahead	      		| Turn Right Ahead					 				|
| Stop			| Stop      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This is a less accurate prediction average than the model performed on the test set. The ambiguity between the model's representations of the "children crossing" and the "bicycles crossing" classes is shown in this inaccuracy.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 26th cell of the Ipython notebook.

For the first image, the model is very sure that this is a "Bicycles Crossing" sign (probability of 0.98), and the image does not contain a "Bicycles Crossing" sign. As you can see, the similar traffic signs "Children Crossing" (the correct class) and "Road Work" were the next-highest ranked predictions. Implementing dropout may have helped my model better distinguish between these nearby classes. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .98         			| Bicycles crossing   									| 
| .01     				| Children crossing 										|
| .00					| Road work											|
| .00	      			| Beware of ice/snow					 				|
| .00				    | Right-of-way at the next intersection      							|

In the second image, the model is completely certain that this is a "No entry" sign (probability of 1.00), and it was correct in this prediction. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| No Entry   									| 
| .00     				| Turn Right Ahead 										|
| .00					| Stop											|
| .00	      			| End of all speed and passing limits					 				|
| .00				    | Speed Limit (20km/h)      							|

In the third image, the model is completely certain that this is a "No passing for vehicles over 3.5 metric tons" sign (probability of 1.00), and it was correct in this prediction. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| No passing for vehicles over 3.5 metric tons   									| 
| .00     				| End of no passing by vehicles over 3.5 metric tons 										|
| .00					| Ahead only											|
| .00	      			| No passing					 				|
| .00				    | Speed Limit (80km/h)      							|

In the fourth image, the model is completely certain that this is a "Turn right ahead" sign (probability of 1.00), and it was correct in this prediction. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Turn right ahead   									| 
| .00     				| Road Work 										|
| .00					| Ahead Only											|
| .00	      			| Keep Left					 				|
| .00				    | Right-of-way at the next intersection      							|

In the fifth image, the model is reasonably certain that this is a "Stop" sign (probability of 0.71), and it was correct in this prediction. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .71         			| Stop   									| 
| .26     				| Turn Right Ahead 										|
| .02					| End of all speed and passing limits											|
| .00	      			| Speed Limit (30km/h)					 				|
| .00				    | Yield      							|


