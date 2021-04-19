import pandas as pd
import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import StratifiedKFold
import config
import model_dispatcher
import plot
import evaluation
from collections import defaultdict
import joblib
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
np.random.seed( 42 )

def return_split( Y ):
    skf = StratifiedKFold( n_splits = config.NUM_SPLITS, shuffle = True, random_state = 42 )
    split = 1
    for train_idx, val_idx in skf.split( np.zeros( Y.shape[0] ), Y ):
        print( 'Returning Split:', split )
        split += 1
        yield train_idx, val_idx

def get_model_name( num_fold ):
    print( 'Num-Fold:', num_fold, 'Model Name:', config.MODELS[ num_fold - 1 ] )
    if(num_fold in config.FOLD_OF_neuralNet):
        return os.path.join(config.SAVE_MODEL, config.MODELS[num_fold - 1] + '_' + str(num_fold) )
    else:
        return os.path.join(config.SAVE_MODEL, config.MODELS[num_fold - 1] + '_' + str(num_fold) + '.pkl')


def return_callbacks(num_fold):
    print('Returning Callbacks')
    mc = ModelCheckpoint(
        filepath=get_model_name(num_fold),
        monitor='val_accuracy',
        mode='max',
        verbose=1,
        save_best_only=True,
        save_weights_only=False
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_accuracy',
        mode='max',
        factor=0.5,
        patience=3,
        verbose=1
    )

    return mc, reduce_lr


def return_opt(opt_name, learning_rate):
    print(opt_name.upper(), 'Optimizer Selected with learning rate:', learning_rate)
    opt_dict = {
        'adam': tf.keras.optimizers.Adam(learning_rate=learning_rate),
        'sgd': tf.keras.optimizers.SGD(learning_rate=learning_rate),
        'rms': tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
        'ftrl': tf.keras.optimizers.Adagrad(learning_rate=learning_rate),
        'nadam': tf.keras.optimizers.Nadam(learning_rate=learning_rate),
        'ada_delta': tf.keras.optimizers.Adadelta(learning_rate=learning_rate),
        'ada_grad': tf.keras.optimizers.Adagrad(learning_rate=learning_rate),
        'ada_max': tf.keras.optimizers.Adamax(learning_rate=learning_rate),
    }

    return opt_dict[opt_name]

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
        model = model_dispatcher.return_model( num_fold )   
        if( num_fold in config.FOLD_OF_neuralNet ):
            X_train, Y_train = config.preprocessing_function[config.MODELS[ num_fold - 1 ]]( X_train, Y_train, model_path, isTrain = True )
            X_val, Y_val = config.preprocessing_function[config.MODELS[ num_fold - 1 ]]( X_val, Y_val, model_path, isTrain = False )
            X_train, Y_train = config.preprocessing_function['oversample']( X_train, Y_train )
            print( 'X_train Shape:', X_train.shape, '\nY_train Shape:', Y_train.shape, '\nX_val Shape:', X_val.shape, '\nY_val Shape:', Y_val.shape )
            mc, reduce_lr = return_callbacks(
                num_fold
            )
            opt = return_opt( 'adam', learning_rate = 0.01 )
            model.compile(
                optimizer = opt,
                loss = 'categorical_crossentropy',
                metrics = ['accuracy']
            )
            history = model.fit(
                X_train,
                Y_train,
                validation_data=( X_val, Y_val ),
                epochs = 100,
                callbacks=[mc, reduce_lr],
                batch_size = config.BATCH_SIZE,
                class_weight = config.CLASS_WEIGHT
            )
            plot.plot_loss(history, num_fold )
            plot.plot_accuracy(history, num_fold )
            model = load_model( get_model_name( num_fold ) )
            train_score = evaluation.score_of_model_for_neuralNet( model, X_train, Y_train, 'Training' )
            val_score = evaluation.score_of_model_for_neuralNet( model, X_val, Y_val, 'Validation' )
            print("PREDICT", model.predict(X_val), model.predict(X_val).shape )
            Y_pred = model.predict_classes( X_val )
            print( "Y_PRED:", Y_pred )
            Y_val = np.argmax( Y_val, axis = -1 )
        else:
            X_train = config.preprocessing_function[config.MODELS[ num_fold - 1 ]]( X_train, model_path, isTrain = True )
            X_val = config.preprocessing_function[config.MODELS[ num_fold - 1 ]]( X_val, model_path, isTrain = False )
            X_train, Y_train = config.preprocessing_function['oversample']( X_train, Y_train )
          #  print( 'X_train Shape:', X_train.shape, '\nY_train Shape:', Y_train.shape, '\nX_val Shape:', X_val.shape, '\nY_val Shape:', Y_val.shape )
            if config.MODELS[ num_fold - 1 ] == 'ctbc':
                model.fit(X_train, Y_train, eval_set = ( X_val, Y_val ) )
            else:
                model.fit( X_train, Y_train )
                plot.plot_featureImportance( model, num_fold, train_df.columns )
            joblib.dump( model, model_path )
            train_score = evaluation.score_of_model( model, X_train, Y_train, 'Training' )
            val_score = evaluation.score_of_model( model, X_val, Y_val, 'Validation' )    
            Y_pred = model.predict( X_val )
            plot.plot_rocCurve( model, X_val, Y_val, num_fold )
            
        evaluation.classificationReport_of_model(Y_val, Y_pred)
        plot.plot_confusionMatrix( Y_val, Y_pred, num_fold )

        results[ get_model_name( num_fold ) ] = [ train_score, val_score ]
        num_fold += 1

        if( num_fold > len( config.MODELS ) ):
            break

    return results

if __name__ == '__main__':
    results = train()
    for num_fold in range( 1, len(results) + 1 ):
        print("MODEL: ", config.MODELS[num_fold-1])
        print("TRAIN SCORE:", results[get_model_name(num_fold)][0])
        print("VALIDATION SCORE:", results[get_model_name(num_fold)][1])
