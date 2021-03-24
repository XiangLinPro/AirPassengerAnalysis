'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-03-03 19:59:38
LastEditors: ZhangHongYu
LastEditTime: 2021-03-24 08:49:59
'''
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import make_scoreraa
from sklearn.model_selection import StratifiedKFold
from DataReader import FeatureDictionary, DataParser
#from matplotlib import pyplot as plt
import config
from PNN import k_fold_cross_valid
from DataReader import load_data

pnn_params = {
    "embedding_size":8,
    "deep_layers":[32,32],
    "dropout_deep":[0.5,0.5,0.5],
    "deep_layer_activation":tf.nn.relu,
    "batch_norm":1,
    "batch_norm_decay":0.995,
    "verbose":True,
    "random_seed":config.RANDOM_SEED,
    "loss_type":"logloss",
    "deep_init_size":50,
    "use_inner":False
}

train_params = {
    "loss_type":"logloss",
    "learning_rate":0.01,
    "epochs":30,
    "optimizer_type":"sgd",
    "batch_size":4
}

if __name__ == '__main__':
    # load data
    dfTrain, X_train, y_train, X_submission = load_data()

    # folds
    folds = list(StratifiedKFold(n_splits=config.NUM_SPLITS, shuffle=True,
                                 random_state=config.RANDOM_SEED).split(X_train, y_train))
    k_fold_cross_valid(dfTrain, X_submission, folds, pnn_params, train_params)

