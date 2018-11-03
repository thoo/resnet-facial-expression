![MacDown logo](Images/Facial.png)


## Table of Contents
1. [Introduction](#1-introduction)
2. [Data Preprocessing](#2-data-preprocessing)
3. [Shallow CNN Model](#3-shallow-cnn-model)
4. [Ensemble Models](#4-ensemble-models)
5. [Wider CNN Model](#5-wider-cnn-model)
6. [Residual Neural Network](#6-residual-neural-network)
7. [Web Application](#7-web-application)


## 1 Introdcution

**Deep learning** has revitalized computer vision and enhanced the accuracy of machine interpretation on images in recent years. Especially convolutional neural networks (**CNN**) is so far the go-to machine learning algorithm  with [great preformance](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html). In this project, I build **CNN** models to classify the facial expression into six facial emotion categories such as happy (😄) , sad (😢), fear (😨) , surprise (😲), neutral (😐) and angry (😠).

~~~
label_dict={
 1: ['Angry', '😠'],
 2: ['Fear', '😨'],
 3: ['Happy', '😄'],
 4: ['Sad', '😢'],
 5: ['Surprise', '😲'],
 6: ['Neutral', '😐']
 }
~~~

The dataset is taken from [Kaggle Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/leaderboard).  I train three different **CNN** models: 2-layer model, 3-layer model with dropout to prevent overfitting, and 34-layer residual networks model.

Facial expression recognition can bring the relationship between the human and machine closer. It has gained significant interest in social behavioral science, clinical and security fields.

## 2 Data Preprocessing
The dataset 35,887 images with the resolution of 48x48 pixels. Each image is flatten to a single row with 2,304 columns and so there are 35,887 rows with 2,304 columns in a single **'csv'** file. The following figure is the preview of first 16 images with the class label below each image.

<p align="center">
<img src="Images/original_data.png"  align="middle"/>
<h5 align="center">Figure 1. First 16 images with their class label</h4>
</p>

As you can see from the first few images, this dataset needs some preprocessing. The third image from the left on the first row is even hard for a human to label the facial expression. Here we are only concerns about facial expression. We are only accounting for the facial landmarks. For instance, the second image from the left on the first row can be cropped around the face and we have all the information for the facial expression recognition.

I will realign and crop the face out on some of the images as first preprocessing steps.
Figure 2 illustrates how the process is carried out.  This image is not in the dataset. This is just for demonstration. The first on the left is the initial preprocess data and I extract the facial region using [openface preprocessing alignment script](https://github.com/cmusatyalab/openface/blob/master/util/align-dlib.py) based on **dlib** library. The middle image in the Figure 2. is a post-process image and the last image is how the machine interprets an image. You can see the darker color around the important landmarks and their locations can give the vital information about facial expression. The computer is comparing and analyzing these facial features and it is reasonable to align these landmarks from images to images so that `eye`, `nose` and `mouth` align for all images. This process is demonstrated in Figure 4.

**`Get_face.ipynb`** file contains the preprocessing steps in details while **`Get_face-result.ipynb`** shows the visualization of some post-process images.

<p align="center">
<img src="Images/preprocess.png"  align="middle"/>
<h5 align="center">Figure 3. Extract the facial region.</h4>
</p>

<p align="center">
<img src="Images/rotation.png"  align="middle"/>
<h5 align="center">Figure 4. Align the facial region.</h4>
</p>


The original dataset contains seven different classes including **'disgust'**. However, when we look at the distribution among different categories, **'disgust'** only accounts account for about 1.5% of the total dataset as shown in Figure 2. Therefore, I am going to drop this category. Here the `modified dataset` is the dataset after the data preprocessing step.
<p align="center">
<img src="Images/Percentage.png"  align="middle"/>
<h5 align="center">Figure 5. Data distribution among categories</h4>
</p>


<p align="center">
<img src="Images/labels_dist_comb.png"  height=500 align="middle"/>
<h5 align="center">Figure 6. Total Number of categories</h4>
</p>

## 3 Shallow CNN Model
The first convolutional neural networks model is a 4-layer simple model. There are two convolutional layers and two densely  (fully) connected layers. For the simplicity, there is no regularization and dropout to avoid overfitting in this model.  The average accuracy for this simple model is around **62 %** between 3000 to 10000 iterations.

| Layer Name    | Output Size     | Process |
|:-------------:|:---------------| :-------------|
| conv1         | 24 x 24     	 |       filter =5x5 , stride =2, channels =16 |
| conv1         | 12 x 12     	 |       filter =5x5 , stride =2, channels =36 |
| dc1           | 128     	  |
| dc2           | 6     	 |       softmax,max pool |
| **Flops** [ Ignore biases] ||1.1 x 10^6 |

<p align="center">
<img src="Images/layout2.png"  align="middle"/>
<h5 align="center">Figure 6. Architecture of a simple CNN</h4>
</p>

This is a very shallow network and relatively very easy to train. Here is the loss curve and confusion matrix for this model. As we can conclude from this model, the prediction accuracy is high on `Happy`,`Surprise` and `Neutral` category. This CNN algorithm has hard time classifying `Angry`, `Sad` and `Fear` category and are quite often gotten confused with `Neutral` as illustrated in the confusion matrix. This model can be reimplemented by running `First Convolutional Neural Net-fv.ipynb`.



<p align="center">
<img src="Images/curve.png"  align="middle"/>
<h5 align="center">Figure 7. Loss curve and accuracy on the validation dataset on the left and the confusion matrix on the right figure.</h4>
</p>

## 4 Ensemble Models
Instead of applying the best single model to predict the categories, I will use multiple models and combine all the predictions together and determine the prediction based on the highest score. So when I train the model in `First Convolutional Neural Net-fv.ipynb`, I save the model every 1000 iterations. Now each model will give a six probabilistic predictions with respect to each category on an image and I will combine these prediction per category across 9 models. By judging from Figure , ensemble learning has boosted the accuracy on categories such that `Sad`, `Fear` and `Angry`. As expected the accuracy of the prediction is increased by ~2% with 63.7% accuracy.

<p align="center">
<img src="Images/Ensemble.png"  align="middle"/>
<h5 align="center">Figure 8. Confusion matrix for ensemble models</h4>
</p>

## 5 Wider CNN Model
In the previous model, we don't consider dropout and regularization to avoid overfitting. In this model, we go one layer deeper and also use 30% dropout.


| Layer Name    | Output Size     | Process |
|:-------------:|:---------------| :-------------|
| conv1         | 24 x 24     	 |       filter =5x5 , conv\_stride=1,max\_pool \_stride =2, channels =32 |
| conv2         | 12 x 12     	 |       filter =5x5 , conv\_stride=1,max\_pool \_stride =2, channels =64 |
| conv3         | 6 x 6     	 |       filter =5x5 , conv\_stride=1,max\_pool \_stride =2, channels =64 |
| dc1           | 128     	  |
| dc2           | 6     	 |       softmax,max pool |
| **Flops** [ Ignore biases] ||1.2 x 10^7 |

The second model is slightly better than the first one with an average accuracy of around **63.6%** compared to **~62 %** from the first model. But in terms of the amount of computation, only 1.6% increasement is not a good improvement. Therefore, I look into the ensemble learning and the accuracy of the ensemble learning is about **65.9 %**. It is about 2.2% improvement on accuracy compared to the first shallow CNN model. Accuracy on differentiating between `Sad` and `Neutral` gains a boost in this model.

<p align="center">
<img src="Images/Ensemble2.png"  align="middle"/>
<h5 align="center">Figure 7. Confusion matrix for ensemble deeper CNN models</h4>
</p>

This model is demonstrated in
`Second Convolutional Neural Net Model.ipynb`. Let take a look at a few images in which the algorithm couldn't give the correct prediction. It is even hard for human to predict the category of some of the images shown in Figure 8.

<p align="center">
<img src="Images/Wrong_prediction1.jpg"  align="middle"  />
<h5 align="left">Figure 8. Images where the model predicted incorrectly. In the caption below each image, the label is on the left and the prediction is on the right. </h4>
</p>

## 6 Residual Neural Network
`Deep Convolutional Neural Net Model.ipynb` contains how to setup and run 34-layer residual neural network introduced by [Kaiming He](https://github.com/KaimingHe/deep-residual-networks/commits/master).

<p align="center">
<img src="Images/curve3.png"  align="middle"/>
<h5 left="center">Figure 9. Loss curve and accuracy on the validation dataset on the left and the confusion matrix on the right figure.</h4>
</p>

## 7 Web Application
For the deployment, I need three major dependencies: *tensorflow*, *dlib*, and *opencv*. To meet all the package requirements, I packed all python dependencies and scripts into a docker image and deploy it to Google Container Registry. Then, transform the docker image to a web application using Google App Engine. The web application can be accessed at https://facial-expression-recognition.appspot.com . One can upload the image and the application looks for faces in the image by detecting the facial landmark with `dlib`. Then, use residual neural network to predict the emotional state and plot a probability bar graph.
