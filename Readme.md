![MacDown logo](/Volumes/SAM_USB/gsync_FE/Images/Facial.png)


## Table of Contents
1. [Introduction](#1-introduction)
2. [Data Preprocessing](#2-data-preprocessing)
3. [Strong and Emphasize](#Strong-and-Emphasize)


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

Facial expression recognition can bring the relationship between the human and machine closer. It has gained significant interest in socail behavioral science, clinical and security fields. 

## 2 Data Preprocsessing
The dataset 35,887 images with the resolution of 48x48 pixels. Each image is flatten to a single row with 2,304 columns and so there are 35,887 rows with 2,304 columns in a single **'csv'** file. The following figure is the preview of first 16 images in binary colormap with the class label below each image. 

<p align="center">
<img src="/Volumes/SAM_USB/gsync_FE/Images/original_data.png"  align="middle"/>
<h5 align="center">Figure 1. First 16 images with their class label</h4>
</p>

As you can see from the first few images, this dataset needs some preprocessing. The third image from the left on the first row is even hard for a human to label the facial expression. Here we are only concerns about facial expression. We are only accounting for the facail landmarks. For instance, the second image from the left on the first row can be cropped around the face and we have all the information for the facial expression recognition. 

I will realign and crop the face out on some of the images as first preprocessing steps. 
Figure 2 illustrates how the process is carried out.  This image is not in the dataset. This is just for demonstration. The first on the left is the initial preprocess data and I extract the facial region using [openface preprocessing alignment script](https://github.com/cmusatyalab/openface/blob/master/util/align-dlib.py) based on **dlib** library. The middle image in the Figure 2. is a post-process image and the last image is how the machine interprets an image. You can see the darker color around the important landmarks and their locations can give the vital information about facial expression. The computer is comparing and analyzing these facial features and it is reasonable to align these landmarks from images to images so that `eye`, `nose` and `mouth` align for all images. This process is demonstrated in Figure 4.

**`Get_face.ipynb`** file contains the preprocessing steps in details while **`Get_face-result.ipynb`** shows the visualization of some post-process images.

<p align="center">
<img src="/Volumes/SAM_USB/gsync_FE/Images/preprocess.jpg"  align="middle"/>
<h5 align="center">Figure 3. Extract the facial region.</h4>
</p>

<p align="center">
<img src="/Volumes/SAM_USB/gsync_FE/Images/rotation.png"  align="middle"/>
<h5 align="center">Figure 4. Align the facial region.</h4>
</p>


The original dataset contains seven different classes including **'disgust'**. However, when we look at the distribution among different categories, **'disgust'** only accounts account for about 1.5% of the total dataset as shown in Figure 2. Therefore, I am going to drop this category. Here the `modified dataset` is the dataset after the data preprocessing step. 
<p align="center">
<img src="/Volumes/SAM_USB/gsync_FE/Images/Percentage.png"  align="middle"/>
<h5 align="center">Figure 5. Data distribution among categories</h4>
</p>


<p align="center">
<img src="/Volumes/SAM_USB/gsync_FE/Images/labels_dist_comb.png"  height=500 align="middle"/>
<h5 align="center">Figure 6. Total Number of categories</h4>
</p>

## 3 Shallow Model
The first convolutional neural networks model is a 4-layer simple model. There are two convolutional layers and two densely  (fully) connected layers. For the simplicity, there is no regularization and dropout to avoid overfitting in this model.  The average accuracy for this simple model is around **62 %** between 3000 to 10000 iterations. 

| Layer Name    | Output Size     | Process |
|:-------------:|:---------------| :-------------|
| conv1         | 24 x 24     	 |       filter =5x5 , stride =2, channels =16 |
| conv1         | 12 x 12     	 |       filter =5x5 , stride =2, channels =36 |
| dc1           | 128     	  | 
| dc2           | 6     	 |       softmax,max pool |
| **Flops** [ Ignore biases] ||1.1 x 10^6 |

<p align="center">
<img src="/Volumes/SAM_USB/gsync_FE/Images/layout2.png"  align="middle"/>
<h5 align="center">Figure 6. Architecture of a simple CNN</h4>
</p>

This is a very shallow network and relatively very easy to train. Here is the loss curve and confusion matrix for this model. As we can conclude from this model, the prediction accurcy is high on `Happy`,`Surprise` and `Neutral` category. This CNN algorithm has hard time classifying `Angry`, `Sad` and `Fear` category and are quite often gotten confused with `Neutral` as illustrated in the confusion matrix. This model can be reimplemented by running `First Convolutional Neural Net-fv.ipynb`.  



<p align="center">
<img src="/Volumes/SAM_USB/gsync_FE/Images/curve.png"  align="middle"/>
<h5 align="center">Figure 7. Loss curve and accuracy on the validation dataset on the left and the confusion matrix on the right figure.</h4>
</p>

## 4 Ensemble Models
Instead of applying the best single model to predict the categories, I will use multiple models and combine all the predictions together and determine the prediction based on the highest score. So when I train the model in `First Convolutional Neural Net-fv.ipynb`, I save the model every 1000 iterations. Now each model will give a six probabilistic preditions with respect to each category on an image and I will combine these prediction per category across 9 models. By juding from Figure , ensemble learning has boosted the accuracy on categories such that `Sad`, `Fear` and `Angry`. As expected the accuracy of the prediction is increased by ~2% with 63.7% accuracy.  

<p align="center">
<img src="/Volumes/SAM_USB/gsync_FE/Images/Ensemble.png"  align="middle"/>
<h5 align="center">Figure 8. Confusion matrix for ensemble models</h4>
</p>

## 5 
