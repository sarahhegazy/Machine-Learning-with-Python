#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 19:17:10 2018

"""
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

def classify(X_train, Y_train, X_test):
	
    from sklearn import model_selection
    #Selecting models to use
    estimators = []
    model1 = LogisticRegression()
    estimators.append(('lr', model1))
    model3 = SVC()
    estimators.append(('svm', model3))
    
    # create the ensemble model
    ensemble = VotingClassifier(estimators)
    results = model_selection.cross_val_score(ensemble, X_train, Y_train, cv=10)
    print(results.mean())
    ultimate_model=ensemble.fit(X_train,Y_train)
    score=ultimate_model.score(X_train,Y_train)
    
    predictedLabels=ultimate_model.predict(X_test)

    return predictedLabels,score

