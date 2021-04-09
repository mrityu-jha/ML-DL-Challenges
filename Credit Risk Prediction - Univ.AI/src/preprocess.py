from sys import path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
import joblib
import os 

def path_of_preprocess( path_of_model ):
    print( "Path of Preprocess:", os.path.join( os.path.dirname( path_of_model ), 'preprocess_' + os.path.splitext(os.path.basename(path_of_model))[0] + '.pkl' ) )
    return os.path.join( os.path.dirname( path_of_model ), 'preprocess_' + os.path.splitext(os.path.basename(path_of_model))[0] + '.pkl' )

def basic( df, isTrain = True ):
    print( "Basic Preprocessing" )
    df['city'] = df['city'].apply( lambda x : x.split( '[' )[0] if '[' in x else x )
    df['state'] = df['state'].apply( lambda x : x.split( '[' )[0] if '[' in x else x )
    if isTrain:
        risk_flag = df[['risk_flag']].copy()
        df.drop( ['risk_flag','Id', 'city'], axis = 1, inplace = True )
        df = pd.get_dummies(df, drop_first=True)
        df = pd.concat([df, risk_flag], axis=1)
        return df
    else:
        df.drop(['id', 'city'], axis=1, inplace=True)
        df = pd.get_dummies(df, drop_first=True)
        return df


def svc(X, path_of_model, isTrain=True):
    print( "SVC Preprocessing for:", path_of_model )
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


def rfc(X, path_of_model, isTrain=True):
    print("RFC Preprocessing for:", path_of_model)
    if isTrain:
        print( 'Transforming Training Data' )
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        joblib.dump(scaler, path_of_preprocess(path_of_model))
        return X
    else:
        print( 'Transforming Test Data' )
        scaler = joblib.load(path_of_preprocess(path_of_model))
        X = scaler.transform(X)
        return X


def adaboost(X, path_of_model, isTrain=True):
    print("ADABOOST Preprocessing for:", path_of_model)
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
