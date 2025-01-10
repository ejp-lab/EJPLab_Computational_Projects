import os
import sys
import glob
import numpy as np 
import pandas as pd 
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
import warnings

warnings.filterwarnings('ignore')

def CreateSelectKBestDictionary(data, classes, n_splits):

    col_names = data.columns.to_list()
    X_train_temp, X_test_temp, Y_train_temp, Y_test_temp = train_test_split(data, classes, stratify=classes, test_size=(1 / n_splits), random_state=42)

    out_dict = {}

    for number in range(2, len(data.columns)):
        selector = SelectKBest(f_classif, k=number)
        selector.fit_transform(X_train_temp, Y_train_temp)

        cols = list(selector.get_support(indices=True))
        cols_to_keep = [col_names[i] for i in cols]

        out_dict.update({number : cols_to_keep})

    return out_dict

def SetupAlgorithms():
    
    neigh = KNeighborsClassifier()
    # You can add other things here if you want. Just add to the lists below.

    algorithm_list = [neigh]
    algorithm_names = ['KNN']

    return algorithm_list, algorithm_names

def Preprocess(fil, num_splits, num_repeats ):

    length = int(num_splits * num_repeats)

    features = pd.read_csv(fil)
    features = features.drop([features.columns[0]], axis=1)
    labels = features[features.columns[-1]]
    features = features.drop([features.columns[-1]], axis=1)

    feature_selection_dictionary = CreateSelectKBestDictionary(features, labels, num_splits)
    keys = list(feature_selection_dictionary.keys())

    clfs, clf_names = SetupAlgorithms()
    for idx, clf in enumerate(clfs):

        out_dictionary = {}

        rskf = RepeatedStratifiedKFold(n_splits=num_splits, n_repeats=num_repeats, random_state=36851234)

        for key in keys:
            print(clf_names[idx] + ' : ' + str(key) + ' Features')
            tmp_cols = feature_selection_dictionary[key]

            avg_score = 0

            for train_index, test_index in rskf.split(features, labels):
                X_train, X_test = features.loc[list(train_index)], features.loc[list(test_index)]
                Y_train, Y_test = labels.loc[list(train_index)], labels.loc[list(test_index)]

                X_train = X_train[tmp_cols]
                X_test = X_test[tmp_cols]

                scaler = StandardScaler()
                scaler.fit(X_train)

                scaled_train_f = scaler.transform(X_train)
                scaled_test_f = scaler.transform(X_test)

                clf.fit(scaled_train_f, Y_train)
                clf_prediction = clf.predict(scaled_test_f)
                clf_score = clf.score(scaled_test_f, Y_test)

                avg_score += clf_score

            out_dictionary.update({key : avg_score / length})

        keys = list(out_dictionary.keys())
        values = list(out_dictionary.values())

        with open('KBestTotal_' + clf_names[idx] + '.csv','w') as h:
            h.write('num_features' + ',' + 'Accuracy' + '\n')
            for idx, key in enumerate(keys):
                h.write(str(key)) 
                h.write(',')
                h.write(str(values[idx]))
                h.write('\n')

    return True

if __name__ == "__main__":

    cwd = os.getcwd() + '/'

    input_file = sys.argv[1]
    number_splits = int(sys.argv[2])
    number_repeats = int(sys.argv[3])

    Preprocess(input_file, number_splits, number_repeats)
