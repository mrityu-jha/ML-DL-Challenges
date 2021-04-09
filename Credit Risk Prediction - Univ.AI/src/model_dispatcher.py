from os import name
import tensorflow as tf
import keras
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import load_model
from tensorflow.python.ops.gen_math_ops import xlog1py
import config
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RFC, AdaBoostClassifier as ABC

def svc( params = {} ):
    model = SVC( **params )
    print( model.get_params() )
    return model

def rfc( params = {} ):
    model = RFC( **params )
    print( model.get_params() )
    return model

def adaboost( params = {} ):
    model = ABC( **params )
    print( model.get_params() )
    return model

def return_model( num_fold = None, name_of_model = None ):
    if num_fold is not None and name_of_model is None:
        print( 'NUM FOLD: ', num_fold )
    elif num_fold is None and name_of_model is not None:
        print( 'Name of Model: ', name_of_model )
    elif num_fold is not None and name_of_model is not None:
        print( 'Atleast one of the arguments passed must be None')
        return None
    else:
        print( 'Atleast one of the arguments passed must be not None')
        return None

    model_dict = { 
        
        'svc' : svc( { 
            'C' : 1.0,
            'kernel' : 'rbf',
            'probability' : True,
            'verbose' : True,
            'random_state' : 0
        } ),

        'rfc' : rfc( {
            'n_estimators' : 100,
            'max_depth' : None,
            'min_samples_leaf' : 1,
            'verbose' : 1,
            'random_state' : 0
        } ),

        'adaboost': adaboost ({
            'base_estimator' : None,
            'n_estimators' : 100,
            'learning_rate' : 1,
            'algorithm' : 'SAMME.R',
            'random_state' : 0
        } )

    }

    if( num_fold == None ):
        try:
            return model_dict[name_of_model]
        except:
            print( 'Invalid Model Name passed' )
    else:
        try:
            print( 'MODEL SELECTED: ', config.MODELS[ num_fold - 1  ] )
            return model_dict[ config.MODELS[ num_fold - 1 ] ]
        except:
            print( 'The value of num_fold is invalid' )
            
