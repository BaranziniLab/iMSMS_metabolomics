import talos as ta
from talos.utils import hidden_layers

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import os
import multiprocessing as mp
import joblib
import time

from data_utility import get_processed_global_and_targeted_data, get_processed_global_data, get_processed_targeted_data
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
from paths import *


sample = sys.argv[1]
NCORES = int(sys.argv[2])


ITR = 1000

data_to_go, analyte_columns_selected = get_processed_global_and_targeted_data(sample)
test_size = 0.2
p = {'first_neuron': [500, 750, 1000],
     'second_neuron': [100, 150, 200, 250, 300, 500, 650, 750, 900, 1000],
     'activation': ['relu'],
     'last_activation': ['sigmoid'],
     'batch_size': [32],
     'epochs': [50],
     'dropout': [0.2],
     'lr': [0.001],
     'early_stopping': [tf.keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', verbose=0, patience=10, restore_best_weights=True)]
    }


def main():
    start_time = time.time()
    p = mp.Pool(NCORES)
    itr_array = np.arange(ITR)
    out_dict_list = p.map(train, itr_array)
    p.close()
    p.join()
    joblib.dump(out_dict_list, os.path.join(OUTPUT_PATH, "patient_classification", "dnn_models_total_analyte", sample, "data", "data_used_for_talos_optimized_dnn_models_for_global_compound_{}_sample.joblib".format(sample)))
    print("Completed in {} min".format(round((time.time() - start_time)/60, 2)))
    

def train(itr):
    df_train, df_test = train_test_split(data_to_go, test_size=test_size, stratify=np.array(data_to_go['Disease_label']))
    X_train = df_train[analyte_columns_selected].values
    y_train = df_train.Disease_label.values
    X_test = df_test[analyte_columns_selected].values
    y_test = df_test.Disease_label.values
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    scan_object = ta.Scan(x=X_train_norm, y=y_train, x_val=X_test_norm, y_val=y_test, model=dnn_model, params=p, 
                          experiment_name='dnn_models', fraction_limit=0.99)
    best_model = scan_object.best_model(metric='val_auc', asc=False)
    summary_df = scan_object.data
    summary_df = summary_df.sort_values('val_auc', ascending=False)
    out_dict = {}
    out_dict["train"] = df_train
    out_dict["test"] = df_test
    out_dict["X_train_norm"] = X_train_norm
    out_dict["X_test_norm"] = X_test_norm
    out_dict["scaler"] = scaler
    out_dict["model_index"] = itr+1
    summary_df.to_csv(os.path.join(OUTPUT_PATH, "patient_classification", "dnn_models_total_analyte", sample, "summary", "talos_for_global_compounds_{}_sample_index_{}.csv".format(sample, itr+1)), index=False, header=True)
    best_model.save(os.path.join(OUTPUT_PATH, "patient_classification", "dnn_models_total_analyte", sample, "models", "talos_best_dnn_model_for_global_compounds_{}_sample_index_{}.h5".format(sample, itr+1)))
    return out_dict


def dnn_model(X_train, y_train, X_val, y_val, params):
    neg, pos = np.bincount(y_train)
    total = neg + pos
    initial_bias = np.log([pos/neg])
    weight_for_0 = (1 / neg)*(total)/2.0
    weight_for_1 = (1 / pos)*(total)/2.0
    class_weight = {0: weight_for_0, 1: weight_for_1}
    input_dim = X_train.shape[-1]
    output_bias = tf.keras.initializers.Constant(initial_bias)
        
    model = Sequential()    
    model.add(Dense(params['first_neuron'], input_dim=X_train.shape[1], activation=params['activation']))
    model.add(Dropout(params['dropout']))
    model.add(Dense(params['second_neuron'], input_dim=X_train.shape[1], activation=params['activation']))
    model.add(Dense(1, activation=params['last_activation']))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=params['lr']), loss='binary_crossentropy', metrics=['AUC'])
    history = model.fit(X_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'], 
                        validation_data=(X_val, y_val), class_weight = class_weight,  verbose=0, callbacks=[params['early_stopping']])
    return history, model

if __name__ == "__main__":
    main()







