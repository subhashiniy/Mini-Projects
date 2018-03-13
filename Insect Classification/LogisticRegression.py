#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 12:20:42 2018

@author: subha
"""

# organize imports
from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import accuracy_score
import h5py
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import average_precision_score
import os
import json
import pickle
#import seaborn as sns
#import matplotlib.pyplot as plt

# load the user configs
with open('/home/subha/InsectData/Conf/conf.json') as f:    
  config = json.load(f)

# config variables
test_size     = config["test_size"]
seed      = config["seed"]
features_path   = config["features_path"]
labels_path   = config["labels_path"]
results     = config["results"]
classifier_path = config["classifier_path"]
train_path    = config["train_path"]
num_classes   = config["num_classes"]
classifier_path = config["classifier_path"]

# import features and labels
h5f_data  = h5py.File(features_path, 'r')
h5f_label = h5py.File(labels_path, 'r')

features_string = h5f_data['dataset_1']
labels_string   = h5f_label['dataset_1']

features = np.array(features_string)
labels   = np.array(labels_string)

h5f_data.close()
h5f_label.close()

# verify the shape of features and labels
#print (" features shape: {}".format(features.shape))
#print ("labels shape: {}".format(labels.shape))

#print ("training started...")
# split the training and testing data
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(features),
                                                                  np.array(labels),
                                                                  test_size=test_size,
                                                                  random_state=seed)

#print ("splitted train and test data...")
#print ("train data  : {}".format(trainData.shape))
#print ("test data   : {}".format(testData.shape))
#print ("train labels: {}".format(trainLabels.shape))
#print ("test labels : {}".format(testLabels.shape))


model = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
model.fit(trainData, trainLabels)

# evaluate the model of test data
preds = model.predict(testData)
pickle.dump(model, open(classifier_path, 'wb'))
#print(labels)
# display the confusion matrix
print ("Confusion matrix")

# get the list of training lables
labels = sorted(list(os.listdir(train_path)))

# plot the confusion matrix
cm = confusion_matrix(testLabels, preds)
#sns.heatmap(cm,annot=True,cmap="Set2")
#plt.show()
print(cm)
print("accuracy:")
print(accuracy_score(testLabels,preds))
