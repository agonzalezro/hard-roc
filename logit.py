#!/usr/bin/python
# -*- coding: utf8 -*-
 
# SAMPLE SUBMISSION TO THE BIG DATA HACKATHON 13-14 April 2013 'Influencers in a Social Network'
# .... more info on Kaggle and links to go here
#
# written by Ferenc Huszár, PeerIndex
 
from sklearn import linear_model
from sklearn.metrics import auc, roc_curve
import numpy as np
 
###########################
# LOADING TRAINING DATA
###########################
 
trainfile = open('train_f09.csv')
header = trainfile.next().rstrip().split(',')
 
y_train = []
X_train = []
 
for line in trainfile:
    splitted = line.rstrip().split(',')
    label = int(splitted[0])
    y_train.append(label)
    X_train.append([float(item) for item in splitted[1:]])

trainfile.close()
 
y_train = np.array(y_train)

testfile = open('test_f09.csv')
#ignore the test header
testfile.next()
 
X_test = []
for line in testfile:
    splitted = line.rstrip().split(',')
    X_test.append([float(item) for item in splitted])

testfile.close()
from numpy import array
X_test = array(X_test)
y_train = array(y_train)
X_train = array(X_train)

from data import normalize
X_train, X_test = normalize(X_train, X_test)

###########################
# EXAMPLE BASELINE SOLUTION USING SCIKIT-LEARN
#
# using scikit-learn LogisticRegression module without fitting intercept
# to make it more interesting instead of using the raw features we transform them logarithmically
# the input to the classifier will be the difference between transformed features of A and B
# the method roughly follows this procedure, except that we already start with pairwise data
# http://fseoane.net/blog/2012/learning-to-rank-with-scikit-learn-the-pairwise-transform/
###########################
 
model = linear_model.LogisticRegression(fit_intercept=True, penalty='l2', C=25.1)
model.fit(X_train,y_train)
# compute AuC score on the training data (BTW this is kind of useless due to overfitting, but hey, this is only an example solution)
p_train = model.predict_proba(X_train)
p_train = p_train[:,1:2]

(fpr, tpr, thresholds) = roc_curve(y_train, p_train.T)
print 'AuC score on training data:',auc(fpr, tpr)
 
###########################
# READING TEST DATA
###########################
 
 
# transform features in the same way as for training to ensure consistency
# compute probabilistic predictions
p_test = model.predict_proba(X_test)
#only need the probability of the 1 class
p_test = p_test[:,1:2]
 
###########################
# WRITING SUBMISSION FILE
###########################
predfile = open('predictions.csv','w+')
 
print >>predfile,','.join(header)
for p in p_test:
    print >>predfile, ('%.5f'%p)
 
predfile.close()
