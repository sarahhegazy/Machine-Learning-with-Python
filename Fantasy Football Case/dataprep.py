#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 19:53:15 2018

@author: ingridkasbah
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

def dataprep(traindata,testdata):
    #Load the data and understand what's in it
    test_df = pd.read_csv(testdata)
    train_df = pd.read_csv(traindata)
    train_df.info()
    train_df.describe() #to visualize and understand the Data at hand
    
    train_df.head()
    
    #calculate the missing values
    total_NA = train_df.isnull().sum().sort_values(ascending=False)
    percentage_NA = pd.concat([total_NA, (round(train_df.isnull().sum()/train_df.isnull().count()*100, 1)).sort_values(ascending=False)], axis=1, keys=['Total', '%'])
    percentage_NA.head(5)
    
    train_df.columns.values
    
    
    #Check correlation between age/sex/and survival
    survived = 'survived'
    not_survived = 'not survived'
    fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
    women = train_df[train_df['Sex']=='female']
    men = train_df[train_df['Sex']=='male']
    ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)
    ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)
    ax.legend()
    ax.set_title('Female')
    ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)
    ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)
    ax.legend()
    _ = ax.set_title('Male')
    
    #survival/Pclass/Embarked
    FacetGrid = sns.FacetGrid(train_df, row='Embarked', size=4.5, aspect=1.6)
    FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None,  order=None, hue_order=None )
    FacetGrid.add_legend()
    
    sns.barplot(x='Pclass', y='Survived', data=train_df)
    
    grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
    grid.map(plt.hist, 'Age', alpha=.5, bins=20)
    grid.add_legend();
    
    #NEW FEATURE: ALONE OR NOT
    data = [train_df, test_df]
    for dataset in data:
        dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
        dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0
        dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
        dataset['not_alone'] = dataset['not_alone'].astype(int)
    train_df['not_alone'].value_counts()
    
    #number of relatives vs the survival
    axes = sns.factorplot('relatives','Survived', 
                          data=train_df, aspect = 2.5, )
    
    #save passenger id before dropping
    PassID = train_df['PassengerId']
    
    #Drop passenger id
    train_df = train_df.drop(['PassengerId'], axis=1)
    
    #New feature related to deck
    import re
    deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
    data = [train_df, test_df]
    
    for dataset in data:
        dataset['Cabin'] = dataset['Cabin'].fillna("U0")
        dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
        dataset['Deck'] = dataset['Deck'].map(deck)
        dataset['Deck'] = dataset['Deck'].fillna(0)
        dataset['Deck'] = dataset['Deck'].astype(int)
    # we can now drop the cabin feature
    train_df = train_df.drop(['Cabin'], axis=1)
    test_df = test_df.drop(['Cabin'], axis=1)
    
    
    
    #which is the most common embarked value?
    train_df['Embarked'].describe()
    
    common_value = 'S'
    data = [train_df, test_df]
    
    for dataset in data:
        dataset['Embarked'] = dataset['Embarked'].fillna(common_value)
        
        #Fare value change of type
    data = [train_df, test_df]
    
    for dataset in data:
        dataset['Fare'] = dataset['Fare'].fillna(0)
        dataset['Fare'] = dataset['Fare'].astype(int)
        
        
    #replace titles, or use Sarah's?     
    data = [train_df, test_df]
    titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Officer": 5, "Master":6,"Royalty":7}
    
    for dataset in data:
        # extract titles
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
        # replace titles with a more common title or as Rare
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Sir', 'Jonkheer', 'Dona','Don'], 'Royalty')
        dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Dr',\
                                                'Major', 'Rev'], 'Officer')    
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
        # convert titles into numbers
        dataset['Title'] = dataset['Title'].map(titles)
        # filling NaN with 0, to get safe
        dataset['Title'] = dataset['Title'].fillna(0)
    train_df = train_df.drop(['Name'], axis=1)
    test_df = test_df.drop(['Name'], axis=1)
    
    
    '''
    train_df["Age"].fillna(train_df.groupby(["Title","Pclass","Sex"])["Age"].transform("median"),inplace=True)
    test_df["Age"].fillna(test_df.groupby(["Title","Pclass","Sex"])["Age"].transform("median"),inplace=True)
    train_df["Age"].isnull().sum()
    test_df["Age"].isnull().sum()
    
    '''
    
    data = [train_df, test_df]
    
    for dataset in data:
        mean = train_df["Age"].mean()
        std = test_df["Age"].std()
        is_null = dataset["Age"].isnull().sum()
        # compute random numbers between the mean, std and is_null
        rand_age = np.random.randint(mean - std, mean + std, size = is_null)
        # fill NaN values in Age column with random values generated
        age_slice = dataset["Age"].copy()
        age_slice[np.isnan(age_slice)] = rand_age
        dataset["Age"] = age_slice
        dataset["Age"] = train_df["Age"].astype(int)
    train_df["Age"].isnull().sum()
    
    #Sex convert to numeric
    genders = {"male": 0, "female": 1}
    data = [train_df, test_df]
    
    for dataset in data:
        dataset['Sex'] = dataset['Sex'].map(genders)
    
    train_df['Ticket'].describe()
    train_df = train_df.drop(['Ticket'], axis=1)
    test_df = test_df.drop(['Ticket'], axis=1)
    
    #Embarked to numeric
    ports = {"S": 0, "C": 1, "Q": 2}
    data = [train_df, test_df]
    
    for dataset in data:
        dataset['Embarked'] = dataset['Embarked'].map(ports)
        
    corr=train_df.corr()
    
    
    #Categorizing age groups
    data = [train_df, test_df]
    for dataset in data:
        dataset['Age'] = dataset['Age'].astype(int)
        dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
        dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
        dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
        dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
        dataset.loc[ dataset['Age'] > 66, 'Age'] = 6
    
    train_df['Age'].value_counts()
    
    train_df.head(10)
    
    data = [train_df, test_df]

    #categorizing Fares
    for dataset in data:
        dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
        dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3
        dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4
        dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5
        dataset['Fare'] = dataset['Fare'].astype(int)
    
     #creating two new classes
    data = [train_df, test_df]
    for dataset in data:
        dataset['Age_Class']= dataset['Age']* dataset['Pclass']
        
    for dataset in data:
        dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['relatives']+1)
        dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)
    # Let's take a last look at the training set, before we start training the models.
    train_df.head(10)
    X_train = train_df.drop("Survived", axis=1)
    Y_train = train_df["Survived"]
    X_test  = test_df.drop("PassengerId", axis=1).copy()
    return X_train, Y_train, X_test, train_df, test_df
