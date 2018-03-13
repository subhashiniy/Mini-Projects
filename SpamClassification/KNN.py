#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 16:29:53 2018

@author: subha
"""

import os

import numpy as np
import pandas as pd

import string
import codecs

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


from scipy import sparse as sp

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_array, check_X_y, check_is_fitted
from sklearn.utils.sparsefuncs import csc_median_axis_0
from sklearn.utils.multiclass import check_classification_targets
# save the classifier


# Get Ham Data from ham file
data = []
target = []
DATA_DIR = '/home/subha/Desktop/enron'
target_names = ['ham', 'spam']
 
def get_data(DATA_DIR):
	subfolders = ['enron6']# % i for i in range(1,7)]
 
	data = []
	target = []
	for subfolder in subfolders:
		# spam
		spam_files = os.listdir(os.path.join(DATA_DIR, subfolder, 'spam'))
		for spam_file in spam_files:
			with open(os.path.join(DATA_DIR, subfolder, 'spam', spam_file), encoding="latin-1") as f:
				data.append(f.read())
				target.append(1)
 
		# ham
		ham_files = os.listdir(os.path.join(DATA_DIR, subfolder, 'ham'))
		for ham_file in ham_files:
			with open(os.path.join(DATA_DIR, subfolder, 'ham', ham_file), encoding="latin-1") as f:
				data.append(f.read())
				target.append(0)	
	
	return data, target
ham = []
for filename in os.listdir('/home/subha/enron1/ham'):
    
    with open(os.path.join('/home/subha/enron1/ham', filename)) as f:
        content = f.read()
        ham.append(content)

# Remove punctuation from each ham email

clean_ham = []
for x in ham:
    clean_ham.append(''.join([c for c in x if c not in string.punctuation]))


clean_stop_ham = []

for email in clean_ham:
    clean_stop_ham.append (' '.join([x for x in email.split() if x.lower() not in stopwords.words('english')]))


# Get Spam Data from spam files inside spam folder

spam = []
for filename in os.listdir('/home/subha/enron1/spam'):
    
    with codecs.open(os.path.join('/home/subha/enron1/spam', filename),encoding='utf-8', errors='ignore') as f:
        content = f.read()
        spam.append(content)


clean_spam = []
for x in spam:
    clean_spam.append(''.join([c for c in x if c not in string.punctuation]))

# remove stopwords 

clean_stop_spam = []
for email in clean_spam:
    clean_stop_spam.append (' '.join([x for x in email.split() if x.lower() not in stopwords.words('english')]))


# create numpy arrays from spam & ham lists

spam_np = np.array(clean_stop_spam)
ham_np = np.array(clean_stop_ham)

# create label for spam and ham , depending on the shape

spam_label = np.repeat('spam',1500)
ham_label = np.repeat('ham',1500)

# create pandas series from those numpy arrays

spam_series = pd.Series(spam_np,index=spam_label)
ham_series = pd.Series(ham_np,index=ham_label)
full_series = pd.concat([spam_series,ham_series])

# create the final dataframe

emails = pd.DataFrame(full_series,columns=['email'])
emails['label'] = emails.index

# reset index and make it as a column

emails.reset_index(inplace=True)
emails.drop('index',axis = 1,inplace = True)

# create the bag of words transformer and fit it to email column and
# emails matrix sparse

bow_transformer = CountVectorizer().fit(emails['email'])
emails_bow = bow_transformer.transform(emails['email'])


class NearestCentroid(BaseEstimator, ClassifierMixin):    

    def __init__(self, metric='euclidean', shrink_threshold=None):
        self.metric = metric
        self.shrink_threshold = shrink_threshold

    def fit(self, X, y):     
        
      
       
        X, y = check_X_y(X, y, ['csr', 'csc'])
        is_X_sparse = sp.issparse(X)
        
        check_classification_targets(y)

        n_samples, n_features = X.shape
        le = LabelEncoder()
        y_ind = le.fit_transform(y)
        self.classes_ = classes = le.classes_
        n_classes = classes.size
        

        # Mask mapping each class to its members.
        self.centroids_ = np.empty((n_classes, n_features), dtype=np.float64)
        # Number of clusters in each class.
        nk = np.zeros(n_classes)

        for cur_class in range(n_classes):
            center_mask = y_ind == cur_class
            nk[cur_class] = np.sum(center_mask)
            if is_X_sparse:
                center_mask = np.where(center_mask)[0]         
            
                self.centroids_[cur_class] = X[center_mask].mean(axis=0)

        if self.shrink_threshold:
            dataset_centroid_ = np.mean(X, axis=0)

            # m parameter for determining deviation
            m = np.sqrt((1. / nk) - (1. / n_samples))
            # Calculate deviation using the standard deviation of centroids.
            variance = (X - self.centroids_[y_ind]) ** 2
            variance = variance.sum(axis=0)
            s = np.sqrt(variance / (n_samples - n_classes))
            s += np.median(s)  # To deter outliers from affecting the results.
            mm = m.reshape(len(m), 1)  # Reshape to allow broadcasting.
            ms = mm * s
            deviation = ((self.centroids_ - dataset_centroid_) / ms)
            # Soft thresholding: if the deviation crosses 0 during shrinking,
            # it becomes zero.
            signs = np.sign(deviation)
            deviation = (np.abs(deviation) - self.shrink_threshold)
            deviation[deviation < 0] = 0
            deviation *= signs
            # Now adjust the centroids using the deviation
            msd = ms * deviation
            self.centroids_ = dataset_centroid_[np.newaxis, :] + msd
        return self

    def predict(self, X):
        
        check_is_fitted(self, 'centroids_')

        X = check_array(X, accept_sparse='csr')
        return self.classes_[pairwise_distances(X, self.centroids_, metric=self.metric).argmin(axis=1)]


def detect_spam_ham(entry):
    
	email_chemss = tfidf_transformer.transform(bow_transformer.transform([entry]))
	#print(email_loaded_model.predict(email_chemss))
	return email_spam_detector_model.predict(email_chemss)[0]


if __name__ == '__main__':
    tfidf_transformer = TfidfTransformer().fit(emails_bow)
    tfidf_emails = tfidf_transformer.transform(emails_bow)
    clf = NearestCentroid(metric='euclidean')
    email_spam_detector_model = clf.fit(tfidf_emails,emails['label'])
    X, y = get_data(DATA_DIR)
    #splice=X[:100]
    #clean_data = []
    ''''for x in splice:
     clean_data.append(''.join([c for c in x if c not in string.punctuation]))
    for email in clean_data:
     clean_data.append (' '.join([x for x in email.split() if x.lower() not in stopwords.words('english')]))'''
    true = y[0:5900]
    test=X[0:5900]
    tp=0
    tn=0
    fp=0
    fn=0
    sum=0;
    for i in range (0,len(test)):
    pred = detect_spam_ham(test[i])
     if(pred=='spam'):
      if((y[i]==1)):
       tp+=1
      else:
       fp+=1
     if(pred=='ham'):
      if(y[i]==0):
       tn+=1
      else:
       fn=1
    print("Accuracy:",(tp+tn)/float(len(true)))
    print("Precision:",tp/(tp+fp))
    print("Recall:",tp/(tp+fn))
    print("True positive:",tp)
    print("True negative:",tn)
    print("False positive:",fp)
    print("False Negative:",fn)



