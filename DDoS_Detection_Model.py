import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, RepeatVector, Input, AveragePooling1D, ReLU, ELU, BatchNormalization
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.models import Model

from tensorflow.keras.utils import to_categorical
import smote_variants as sv
from sklearn.model_selection import KFold, RepeatedKFold, RandomizedSearchCV, RepeatedStratifiedKFold

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.decomposition import TruncatedSVD
from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from keras.layers.merge import concatenate

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

ddos_data = pd.read_csv('drive/My Drive/Colab Notebooks/DDoS/CICDDoS2019/DDoS-Model/4_Classes/4_Classes_NO_Port_IP/DDoS_Dataset_4_Classes_ETC_FS_10_Features.csv')
ddos_data = ddos_data.drop(['Unnamed: 0'], axis=1)

oversampler= sv.MulticlassOversampling(sv.distance_SMOTE())
X_samp, y_samp= oversampler.sample(X, y)

scaler = MinMaxScaler()
Scaled_X = scaler.fit_transform(X_samp)
y = to_categorical(y_samp)
Scaled_X = np.array(Scaled_X)
y = np.array(y)

def Composite_2_CNN_LSTM_32_Net():
    # input layer
    visible = Input(shape=(10,1))
    # first feature extractor
    norm_1 = BatchNormalization()(visible)
    conv1a = Conv1D(32, kernel_size=3, activation='relu', padding='same')(norm_1)
    pool1 = MaxPooling1D(pool_size=(2))(conv1a)

    # second feature extractor
    conv2a = Conv1D(32, kernel_size=3, activation='relu', padding='same')(norm_1)
    pool2 = MaxPooling1D(pool_size=(2))(conv2a)
    pool3 = MaxPooling1D(pool_size=(2))(norm_1)

    # merge feature extractors
    merge = concatenate([pool1, pool2, pool3])
    pool4 = AveragePooling1D(2)(merge) 
    lstm_1 = LSTM(64, return_sequences=True)(pool4)
    hidden_2 = Dense(128)(lstm_1)
    flat1 = Flatten()(hidden_2)
    # output
    output = Dense(4, activation='softmax')(flat1)
    model = Model(inputs=visible, outputs=output)
    return model

kfold = KFold(n_splits=5, shuffle=True)
cvscores_accuracy = []
cvscores_loss = []
for train, test in kfold.split(Scaled_X, y):
    # create model
    model = Composite_2_CNN_LSTM_32_Net()
    # Compile model
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=my_metrics)
    # Fit the model
    model_his = model.fit(Scaled_X[train], y[train], epochs=50, batch_size=23, verbose=2, validation_data=(Scaled_X[test], y[test]))
    # evaluate the model
    scores = model.evaluate(Scaled_X[test], y[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    print("%s: %.3f" % (model.metrics_names[0], scores[0]))
    cvscores_accuracy.append(scores[1] * 100)
    cvscores_loss.append(scores[0])
print("Mean %s Proposed Model: %.2f%% (+/- %.2f%%)" % (model.metrics_names[1], np.mean(cvscores_accuracy), np.std(cvscores_accuracy)))
print("Mean %s Proposed Model: %.3f (+/- %.3f)" % (model.metrics_names[0], np.mean(cvscores_loss), np.std(cvscores_loss)))
# model summay
model.summary()
