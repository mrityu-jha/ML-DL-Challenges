from sys import path
from typing import Counter
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
import joblib
import os 
from imblearn.over_sampling import SMOTE
from keras.utils.np_utils import to_categorical
from sklearn.decomposition import PCA

def path_of_preprocess( path_of_model, type_of_preprocess = 'scaling' ):
    print("Path of Preprocess:", os.path.join(os.path.dirname(path_of_model), 'preprocess_' + type_of_preprocess + '_' + os.path.splitext(os.path.basename(path_of_model))[0] + '.pkl'))
    return os.path.join(os.path.dirname(path_of_model), 'preprocess_' + type_of_preprocess + '_' + os.path.splitext(os.path.basename(path_of_model))[0] + '.pkl')

def basic( df, isTrain = True ):
    print( "Basic Preprocessing" )
#    df['city'] = df['city'].apply( lambda x : x.split( '[' )[0] if '[' in x else x )
    df['state'] = df['state'].apply( lambda x : x.split( '[' )[0] if '[' in x else x )
    df['prev_job_years'] = df['experience'] - df['current_job_years']
    if isTrain:
        risk_flag = df[['risk_flag']].copy()
    #    df.drop( ['risk_flag','Id', 'city', 'profession', 'current_job_years', 'current_house_years'], axis = 1, inplace = True )
        df.drop(['risk_flag', 'Id', 'experience', 'state'], axis=1, inplace=True)
        df = pd.get_dummies(df, drop_first=True)
        df = pd.concat([df, risk_flag], axis=1)
        return df
    else:
        df.drop(['id', 'experience', 'state'], axis=1, inplace=True)
    #    df.drop(['id', 'city', 'profession', 'current_job_years', 'current_house_years'], axis=1, inplace=True)
        df = pd.get_dummies(df, drop_first=True)
        return df

def dim_reduction( X, path_of_model, isTrain = True ):
    # if isTrain:
    #     print('PCA on Training Data')
    #     pca = PCA( n_components = 0.90 )
    #     X = pca.fit_transform(X)
    #     joblib.dump(pca, path_of_preprocess(path_of_model, 'PCA'))
    #     return X
    # else:
    #     print('PCA on Test Data')
    #     pca = joblib.load(path_of_preprocess(path_of_model, 'PCA'))
    #     X = pca.transform(X)
    #     return X
    return X


def min_max_scaler(X, path_of_model, isTrain=True ):
    if isTrain:
        print('Transforming Training Data')
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        joblib.dump(scaler, path_of_preprocess(path_of_model))
        return X
    else:
        print('Transforming Test Data')
        scaler = joblib.load(path_of_preprocess(path_of_model))
        X = scaler.transform(X)
        return X

def oversample( X, Y ):
    oversample = SMOTE()
    X, Y = oversample.fit_resample( X, Y )
    return X, Y

def svc(X, path_of_model, isTrain=True):
    print( "SVC Preprocessing for:", path_of_model )
    X = min_max_scaler( X, path_of_model, isTrain )
    X = dim_reduction( X, path_of_model, isTrain )
    return X


def rfc(X, path_of_model, isTrain=True):
    print("RFC Preprocessing for:", path_of_model)
    X = min_max_scaler(X, path_of_model, isTrain)
    X = dim_reduction(X, path_of_model, isTrain)
    return X


def adaboost(X, path_of_model, isTrain=True):
    print("ADABOOST Preprocessing for:", path_of_model)
    X = min_max_scaler(X, path_of_model, isTrain)
    X = dim_reduction(X, path_of_model, isTrain)
    return X

def xgbc(X, path_of_model, isTrain=True):
    print("XGBOOST Preprocessing for:", path_of_model)
    X = min_max_scaler(X, path_of_model, isTrain)
    X = dim_reduction(X, path_of_model, isTrain)
    return X

def ctbc(X, path_of_model, isTrain=True):
    print("CATBOOST Preprocessing for:", path_of_model)
    X = min_max_scaler(X, path_of_model, isTrain)
    X = dim_reduction(X, path_of_model, isTrain)
    return X

def neuralNet(X, Y, path_of_model, isTrain=True):
    print("NEURAL NET Preprocessing for:", path_of_model)
    if isTrain:
        print('Transforming Training Data')
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        joblib.dump(scaler, path_of_preprocess(path_of_model))
        Y = to_categorical( Y )
        return X, Y
    else:
        print('Transforming Test Data')
        scaler = joblib.load(path_of_preprocess(path_of_model))
        X = scaler.transform(X)
        if Y is not None:
            Y = to_categorical( Y )
            return X, Y 
        return X
