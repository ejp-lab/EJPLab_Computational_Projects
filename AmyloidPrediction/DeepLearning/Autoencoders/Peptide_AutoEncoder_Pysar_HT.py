from tkinter import N
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout

import os
import glob
from matplotlib import cm
from matplotlib import colors
from scipy.stats import pearsonr
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import logging
import optuna
import pickle
from sklearn.preprocessing import StandardScaler
import random
import pdb
import glob

# Peptide_Autoencoder_Pysar_HT.py - Trains a pySAR autoencoder on peptide manifold. (DL_TFGPU)  
# To run: python Peptide_Autoencoder_Pysar_HT.py  
n = 152

# Setting Seeds and logging levels
tf.random.set_seed(420)
np.random.seed(420)
os.environ['PYTHONHASHSEED']=str(420)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)
warnings.filterwarnings('ignore')
tf.get_logger().setLevel(logging.ERROR)
#tf.compat.v1.disable_eager_execution()

# Using this to take testing data out of the generator
testing_set = pd.read_csv("Test_Set.csv")
testing_sequences = testing_set['Sequence'].to_list()
validation_set = pd.read_csv("Validation_Set.csv")
val_sequences = validation_set['Sequence'].to_list()

def generate_arrays_from_file(path, epochs=None, batch_size=1, shuffle=True):

    # Assumes files are pre scaled
    # random may need seeds set (Numpy is set for sure)
    # Randomizes every step of the way so the machine doesnt memorize
    # Assumes dataset was dumped into a folder with partition files
    # Runs forever unless validation steps or steps is specified

    files = glob.glob(path + "/*")
    
    if shuffle:
        random.shuffle(files)
    current_epoch = 0

    while 1:
        
        if shuffle:
            random.shuffle(files)
        
        # One "epoch" is after this for loop finishes
        # each loop is a batch
        for filee in files:

            df = pd.read_csv(filee, index_col=0)
            # taking the testing data out of the generator cause this wasnt split right
            if "Test" not in path and "Val" not in path:

                df = df.loc[~df['Sequence'].isin(testing_sequences)]
                df = df.loc[~df['Sequence'].isin(val_sequences)]

            df = df.drop('Sequence', axis=1)

            array = df.to_numpy()
            if shuffle:
                np.random.shuffle(array)

            # splits into arrays of size split
            # Will be more or less listed batch_size
            split_size = int(array.shape[0]/batch_size)
            if split_size == 0:
                yield (array, array)

            else:
                arrays = np.array_split(array, split_size)
                for array in arrays:
                    yield (array, array)

        current_epoch += 1

        if not epochs:
            continue

        if current_epoch == epochs:
            break

def objective(trial):

    params = {

    'arc' : trial.suggest_categorical('arc', [2,3,4,5]),

    # Initial learning rate
    'initial_learning_rate' : trial.suggest_float('initial_learning_rate', 1e-4, 1e-2 ,log=True),

    'drop' : trial.suggest_float('drop', 0.05, 0.5),
    'act_1' : trial.suggest_categorical('act_1' , ['relu','gelu','tanh','sigmoid','linear']),
    'act_2' : trial.suggest_categorical('act_2' , ['relu','gelu','tanh','sigmoid','linear']),
    'bottle' : trial.suggest_int('bottle', 2, 20),
    'batch_size': trial.suggest_int('batch_size', 64, 512, step=32)
    }

    # Autoencoder
    input = layers.Input(shape=(n,))

    # Encoder
    if params['arc'] == 2:

        encoder_reduced = layers.Dense(params['bottle'], activation="linear")(input)
        encoder_model = Model(input,encoder_reduced) # This actually ties this models weights to that of the Autoencoder which is weird

    # Decoder
        decoder = layers.Dense(n, activation="linear")(encoder_reduced)

    elif params['arc'] == 3:
        encoder = layers.Dense(64,activation=params['act_1'])(input)
        encoder = Dropout(params['drop'])(encoder)

        encoder_reduced = layers.Dense(params['bottle'], activation="linear")(encoder)
        encoder_model = Model(input,encoder_reduced) # This actually ties this models weights to that of the Autoencoder which is weird

    # Decoder
        decoder = layers.Dense(64, activation=params['act_1'])(encoder_reduced)
        decoder = Dropout(params['drop'])(decoder)
        decoder = layers.Dense(n, activation="linear")(decoder)

    elif params['arc'] == 4:
        encoder = layers.Dense(64,activation=params['act_1'])(input)
        encoder = Dropout(params['drop'])(encoder)
        encoder = layers.Dense(32,activation=params['act_2'])(encoder)
        encoder = Dropout(params['drop'])(encoder)

        encoder_reduced = layers.Dense(params['bottle'], activation="linear")(encoder)
        encoder_model = Model(input,encoder_reduced) # This actually ties this models weights to that of the Autoencoder which is weird

    # Decoder
        decoder = layers.Dense(32, activation=params['act_2'])(encoder_reduced)
        decoder = Dropout(params['drop'])(decoder)
        decoder = layers.Dense(64, activation=params['act_1'])(decoder)
        decoder = Dropout(params['drop'])(decoder)
        decoder = layers.Dense(n, activation="linear")(decoder)

    elif params['arc'] == 5:
        encoder = layers.Dense(128,activation='relu')(input)
        encoder = Dropout(params['drop'])(encoder)
        encoder = layers.Dense(64,activation=params['act_1'])(encoder)
        encoder = Dropout(params['drop'])(encoder)
        encoder = layers.Dense(32,activation=params['act_2'])(encoder)
        encoder = Dropout(params['drop'])(encoder)

        encoder_reduced = layers.Dense(params['bottle'], activation="linear")(encoder)
        encoder_model = Model(input,encoder_reduced) # This actually ties this models weights to that of the Autoencoder which is weird

    # Decoder
        decoder = layers.Dense(32, activation=params['act_2'])(encoder_reduced)
        decoder = Dropout(params['drop'])(decoder)
        decoder = layers.Dense(64, activation=params['act_1'])(decoder)
        decoder = Dropout(params['drop'])(decoder)
        decoder = layers.Dense(128, activation="relu")(decoder)
        decoder = Dropout(params['drop'])(decoder)
        decoder = layers.Dense(n, activation="linear")(decoder)

    early_stopping = EarlyStopping(monitor='val_loss',min_delta=0.001, patience=10, restore_best_weights=True)

    initial_learning_rate = params['initial_learning_rate']
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, 
    decay_steps = 10000, 
    decay_rate = 0.96, 
    staircase = True)

    # Autoencoder
    autoencoder = Model(input, decoder)
    autoencoder.compile(tf.keras.optimizers.Adam(learning_rate = lr_schedule),
                        loss='mse',)


    # Training Autoencoder
    history = autoencoder.fit_generator(
        generate_arrays_from_file(training_set, batch_size=params['batch_size']),
        validation_data = generate_arrays_from_file(val_set, batch_size=params['batch_size']),
        epochs=100,
        steps_per_epoch=1000,
        shuffle=True,
        verbose=1,
        validation_steps=len(val_sequences)/params['batch_size'],
        callbacks=[early_stopping]
    )

    
    preds = autoencoder.predict_generator(generate_arrays_from_file(val_set, epochs=1,shuffle=False, batch_size=params['batch_size']))
    labels = np.array([i for i in generate_arrays_from_file(val_set, epochs=1, shuffle=False, batch_size=params['batch_size'])])
    labels = np.concatenate(labels[:, 1])

    error = np.mean(np.square(preds - labels))

    print(error)
    return error

if __name__ == "__main__":

    # Custom for this script because pysar features are too many
    training_set = "PeptideManifoldWithPysarAutocorrelationFeatures_rescaled.csv"
    val_set = "Val_Pysar"
    test_set = "Test_Pysar"

    study = optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=25, n_jobs=1, show_progress_bar=True)
    best_params = study.best_trial.params
    
    pickle.dump(study, open('Pysar_Study','wb'))
    pickle.dump(best_params, open('best_params_autoencoder_Pysar.sav', 'wb'))

    with open("best_params_autoencoder_Pysar.sav","rb") as f:
        best_params = pickle.load(f)

    params = best_params

    input = layers.Input(shape=(n,))

    # Encoder
    if params['arc'] == 2:

        encoder_reduced = layers.Dense(params['bottle'], activation="linear")(input)
        encoder_model = Model(input,encoder_reduced) # This actually ties this models weights to that of the Autoencoder which is weird

    # Decoder
        decoder = layers.Dense(n, activation="linear")(encoder_reduced)

    elif params['arc'] == 3:
        encoder = layers.Dense(64,activation=params['act_1'])(input)
        encoder = Dropout(params['drop'])(encoder)

        encoder_reduced = layers.Dense(params['bottle'], activation="linear")(encoder)
        encoder_model = Model(input,encoder_reduced) # This actually ties this models weights to that of the Autoencoder which is weird

    # Decoder
        decoder = layers.Dense(64, activation=params['act_1'])(encoder_reduced)
        decoder = Dropout(params['drop'])(decoder)
        decoder = layers.Dense(n, activation="linear")(decoder)

    elif params['arc'] == 4:
        encoder = layers.Dense(64,activation=params['act_1'])(input)
        encoder = Dropout(params['drop'])(encoder)
        encoder = layers.Dense(32,activation=params['act_2'])(encoder)
        encoder = Dropout(params['drop'])(encoder)

        encoder_reduced = layers.Dense(params['bottle'], activation="linear")(encoder)
        encoder_model = Model(input,encoder_reduced) # This actually ties this models weights to that of the Autoencoder which is weird

    # Decoder
        decoder = layers.Dense(32, activation=params['act_2'])(encoder_reduced)
        decoder = Dropout(params['drop'])(decoder)
        decoder = layers.Dense(64, activation=params['act_1'])(decoder)
        decoder = Dropout(params['drop'])(decoder)
        decoder = layers.Dense(n, activation="linear")(decoder)

    elif params['arc'] == 5:
        encoder = layers.Dense(128,activation='relu')(input)
        encoder = Dropout(params['drop'])(encoder)
        encoder = layers.Dense(64,activation=params['act_1'])(encoder)
        encoder = Dropout(params['drop'])(encoder)
        encoder = layers.Dense(32,activation=params['act_2'])(encoder)
        encoder = Dropout(params['drop'])(encoder)

        encoder_reduced = layers.Dense(params['bottle'], activation="linear")(encoder)
        encoder_model = Model(input,encoder_reduced) # This actually ties this models weights to that of the Autoencoder which is weird

    # Decoder
        decoder = layers.Dense(32, activation=params['act_2'])(encoder_reduced)
        decoder = Dropout(params['drop'])(decoder)
        decoder = layers.Dense(64, activation=params['act_1'])(decoder)
        decoder = Dropout(params['drop'])(decoder)
        decoder = layers.Dense(128, activation="relu")(decoder)
        decoder = Dropout(params['drop'])(decoder)
        decoder = layers.Dense(n, activation="linear")(decoder)

    early_stopping = EarlyStopping(monitor='val_loss',min_delta=0.001, patience=10, restore_best_weights=True)

    initial_learning_rate = params['initial_learning_rate']
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, 
    decay_steps = 100000, 
    decay_rate = 0.96, 
    staircase = True)

    # Autoencoder
    autoencoder = Model(input, decoder)
    autoencoder.compile(tf.keras.optimizers.Adam(learning_rate = lr_schedule),
                        loss='mse',)


    # Training Autoencoder
    history = autoencoder.fit_generator(
        generate_arrays_from_file(training_set, batch_size=params['batch_size']),
        validation_data = generate_arrays_from_file(val_set, batch_size=params['batch_size']),
        epochs=100,
        steps_per_epoch=1000,
        shuffle=True,
        verbose=1,
        validation_steps=len(val_sequences)/params['batch_size'],
        callbacks=[early_stopping]
    )

    autoencoder.save("Best_HT_Autoencoder_Pysar.model")
    encoder_model.save("Best_HT_Encoder_Pysar.model")

    preds = autoencoder.predict_generator(generate_arrays_from_file(test_set, epochs=1,shuffle=False,batch_size=params['batch_size']))
    labels = np.array([i for i in generate_arrays_from_file(test_set, epochs=1,shuffle=False,batch_size=params['batch_size'])])[0][0]
    error = np.mean(np.square(preds - labels))

    print(f"The test error is: {error}")

    test_comp = encoder_model.predict_generator(generate_arrays_from_file(test_set, epochs=1, shuffle=False, batch_size=params['batch_size']))
    val_comp = encoder_model.predict_generator(generate_arrays_from_file(val_set, epochs=1, shuffle=False, batch_size=params['batch_size']))
    train_comp = encoder_model.predict_generator(generate_arrays_from_file(training_set, epochs=1, shuffle=False, batch_size=params['batch_size']))

    with open("Training_Set_Embeddings_Pysar.pickle", 'wb') as f:
        pickle.dump(train_comp,f)

    with open("Validation_Set_Embeddings_Pysar.pickle", 'wb') as f:
        pickle.dump(val_comp,f)

    with open("Testing_Set_Embeddings_Pysar.pickle", 'wb') as f:
        pickle.dump(test_comp,f)
    
