import os
import sys
import glob
import numpy as np 
import pandas as pd 
import shutil 
import itertools
import random
import multiprocessing
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

scaler = StandardScaler()

fil = sys.argv[1]
num_splits = int(sys.argv[2])
num_repeats = int(sys.argv[3])

features = pd.read_csv(fil)
features = features.drop([features.columns[0]], axis=1)
labels = features[features.columns[-1]]
features = features.drop([features.columns[-1]], axis=1)
features = scaler.fit_transform(features)

model = KNeighborsClassifier()

params = {
    'n_neighbors': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
    'leaf_size': [2,3,4,5,6,7,8,9,10,20,30,40,50],
    'p': [1,2,3,4,5],
    'weights':['uniform', 'distance'],
    'algorithm':['auto']}

cv = RepeatedStratifiedKFold(n_splits=num_splits, n_repeats=num_repeats, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=params, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0, return_train_score=True)
grid_result = grid_search.fit(features, labels)

df = pd.DataFrame(grid_search.cv_results_)

df.to_csv('GridInfo.csv')