Inventory Monitoring at Distribution Centers

as computer vision advances, 
the concept in the project is to identify the number of items in each bin so that the system can 
track the inventory and make sure that delivery consignments have the correct number of 
items.


## Project Set Up and Installation

I used AWS Sagemaker studio to work on this project to use the service you should create a domain on sagemaker and open studio and then upload notebook and py files to start

## Dataset

### Overview
The data acquired is a set of images of bins with items in, and arranged in folders each folder 
contain images with the same number of items vary from one item in the bin to 5 items in the 
bin.
the data name is Amazon Bin Image Dataset that can be downloaded using the json file included with the notebook
the most class with images is the class with three items in the bin

### Access
after downloading the data using json file then uploaded to s3 container so it can be accessed with the estimator to train the model on it.

## Model Training

I created a CNN class using Torch library and initialized a model in the main function with the number of 
classes of 5 as the dataset.

## Machine Learning Pipeline

create a train.py that is used to create the model and train it then using notebook instance to create tuner to get the best set of hyperparameters then use the hyperparametes in an estimator to train the model with multi-instances on 100 epochs. then deployed the model to endpoint and invoked it and get results 

## Standout Suggestions
used Hyperparameter Tuning and multi-instance then deployed the model to endpoint and invoked it and get results
