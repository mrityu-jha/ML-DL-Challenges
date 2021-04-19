import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout, BatchNormalization, Input
from keras.models import load_model, Sequential
from tensorflow.python.keras.models import Sequential
from tensorflow.python.ops.gen_math_ops import xlog1py
import config
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RFC, AdaBoostClassifier as ABC
from xgboost import XGBClassifier 
from catboost import CatBoostClassifier
from tensorflow.keras.regularizers import l1, l2, l1_l2
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

def xgbc( params = {} ):
    model = XGBClassifier( **params )
    print(model.get_params())
    return model

def ctbc( params = {} ):
    model = CatBoostClassifier( **params )
    print( model.get_params() )
    return model

def neuralNet(INPUT_SHAPE_FOR_neuralNet):
    model = Sequential()
    model.add(Input(shape=INPUT_SHAPE_FOR_neuralNet ) )
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='softmax'))
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
            'n_estimators' : 159,
            'max_depth' : 87,
            'min_samples_leaf' : 8,
            'verbose' : 1,
            'max_features' : 'sqrt',
            'bootstrap' : True,
            'min_samples_split' : 2,
            'random_state' : 42
        } ),

        'adaboost': adaboost ({
            'base_estimator' : None,
            'n_estimators' : 100,
            'learning_rate' : 1,
            'algorithm' : 'SAMME.R',
            'random_state' : 0
        } ),
        'xgbc': xgbc( {
            'colsample_bytree': 1.0, 
            'eta': 0.07500000000000001,
            'gamma': 1.0, 
            'max_depth': 17, 
            'min_child_weight': 1.0, 
            'n_estimators': 160, 
            'reg_alpha': 0.8, 
            'reg_lambda': 0.3279258666854785,
            'subsample': 0.5
        } ),
        
        'ctbc' : ctbc( {
            'iterations' : 4000,
            'learning_rate' : 0.01,
            'l2_leaf_reg' : 10,
            'depth' : 10,
            'rsm' : 0.98,
            'loss_function' : 'Logloss',
            'eval_metric' : 'AUC',
            'use_best_model' : True,
            'random_seed' : 42
        } ),

        'neuralNet' : neuralNet( 
            config.INPUT_SHAPE_FOR_neuralNet
        )
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
            
