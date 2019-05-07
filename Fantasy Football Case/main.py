#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 19:24:05 2018

"""

#Note that this scipt only uses the model
#which got us the highest accuracy: ensemble methods
#Note that we used built in 
#score functions to compute our accuracies
#which were done in the 'classify' function
#pipelines including data prep in the functions classify and dataprep

from scipy import stats
from sklearn.metrics import fbeta_score, accuracy_score
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectFromModel
    
# linear algebra
    
import numpy as np   
    
# data processing   
import pandas as pd 
    
    # data visualization
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style
    
# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB

from dataprep import dataprep
from classify import classify
# Dataprep to be ran first
traindata = 'data/train.csv'# Load  the train data
testdata='data/test.csv'	#load test data				  
X_train, Y_train, X_test,train_df,test_df= dataprep(traindata,testdata) #Run this first


#Predicting labels to be ran second
predictedlabels,score=classify(X_train,Y_train,X_test)
print ('Total Accuracy: ', score)
	

