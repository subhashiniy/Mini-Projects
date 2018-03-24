#Text classification of 20 news group data
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.pipeline import Pipeline

#Loading the data set - training data.
twenty_train = fetch_20newsgroups(subset='train', shuffle=True) 

#twenty_train.target_names #prints all the categories



# Building a pipeline:  Extracting features from text files, TF-IDF, Training Naive Bayes (NB) classifier on training data.
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])


text_clf = text_clf.fit(twenty_train.data, twenty_train.target)

#Loading the data set - test data
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)

# Performance of NB Classifier
predicted = text_clf.predict(twenty_test.data)

print("The Accuracy is: ")
print(np.mean(predicted == twenty_test.target))



