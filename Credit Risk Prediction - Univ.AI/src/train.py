import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
import config
import model_dispatcher
import plot
import evaluation
from collections import defaultdict
import joblib

def return_split( Y ):
    skf = StratifiedKFold( n_splits = config.NUM_SPLITS, shuffle = True, random_state = 42 )
    split = 1
    for train_idx, val_idx in skf.split( np.zeros( Y.shape[0] ), Y ):
        print( 'Returning Split:', split )
        split += 1
        yield train_idx, val_idx

def get_model_name( num_fold ):
    print( 'Num-Fold:', num_fold, 'Model Name:', config.MODELS[ num_fold - 1 ] )
    return os.path.join(config.SAVE_MODEL, config.MODELS[num_fold - 1] + '_' + str(num_fold) + '.pkl')

def train():
    print( 'Starting Training' )
    data_frame = pd.read_csv(config.TRAIN_CSV_PATH)
    data_frame = config.preprocessing_function['basic'](data_frame, isTrain = True )
    Y = data_frame[['risk_flag']].copy()
    num_fold = 1
    results = defaultdict( list )

    for train_idx, val_idx in return_split( Y ):
        train_df = data_frame.iloc[train_idx]
        val_df = data_frame.iloc[val_idx]
        train_data = train_df.values
        X_train, Y_train = train_data[ :, :-1 ], train_data[ :, -1 ]
        val_data = val_df.values
        X_val, Y_val = val_data[:, :-1], val_data[:, -1]
        model_path = get_model_name( num_fold )
        X_train = config.preprocessing_function[config.MODELS[ num_fold - 1 ]]( X_train, model_path, isTrain = True )
        X_val = config.preprocessing_function[config.MODELS[ num_fold - 1 ]]( X_val, model_path, isTrain = False )
        print( 'X_train Shape:', X_train.shape, '\nY_train Shape:', Y_train.shape, '\nX_val Shape:', X_val.shape, '\nY_val Shape:', Y_val.shape )
        model = model_dispatcher.return_model( num_fold )   
        model.fit( X_train, Y_train )
        joblib.dump( model, model_path )
        train_score = evaluation.score_of_model( model, X_train, Y_train, 'Training' )
        val_score = evaluation.score_of_model( model, X_val, Y_val, 'Validation' )
    
        plot.plot_rocCurve( model, X_val, Y_val, num_fold )

        results[ get_model_name( num_fold ) ] = [ train_score, val_score ]
        num_fold += 1

        if( num_fold > len( config.MODELS ) ):
            break

    return results

if __name__ == '__main__':
    data = train()
