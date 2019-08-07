#!/usr/bin/env python
from sklearn.externals import joblib
import numpy as np
import pandas as pd

def get_sepsis_score(data, model):

    M1 = joblib.load('model-saved.pkl')
    s_m = np.load('septic_mean.npy', allow_pickle=True)
    ns_m = np.load('Nonseptic_mean.npy', allow_pickle=True)
    All = np.vstack((s_m, ns_m))
    maenAll = np.mean(All, axis=0)

    # Pre processing for sLinear Interpolate
    for column in range(data.shape[1]):
        col = data[:, column]
        value = col[~np.isnan(col)]
        indexVal = np.argwhere(~np.isnan(col))
        indexNaN = np.argwhere(np.isnan(col))
        if ((len(value) == 1) & (col.shape[0] > 1)):
            col[np.int(indexNaN[0])] = data[np.int(indexVal[0]), column]
        data[:, column] = col

    df = pd.DataFrame.from_records(data)

    # sLinear Interpolate and linear approach
    df.interpolate(method='slinear', inplace=True)
    df.interpolate(method='linear', inplace=True)

    ## impute rest of NaN value with mean Value
    data = np.array(df)
    for column in range(data.shape[1]):
        col = data[:, column]
        value = col[np.isnan(col)]
        if len(value) > 0:
            col[np.isnan(col)] = maenAll[column]
        data[:, column] = col

    df = pd.DataFrame.from_records(data)

    predicted = M1.predict(data)

    score = np.random.rand(len(data), 1)
    for i in range(len(data)):
        if predicted[i]==0:
         score[i] = 0.4
        else:
         score[i] = 0.6

    label = np.copy(predicted)

    return score, label

def load_sepsis_model():

    return None
