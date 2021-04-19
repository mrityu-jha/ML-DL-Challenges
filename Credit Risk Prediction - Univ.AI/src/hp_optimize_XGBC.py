import numpy as np
import config
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from train import return_split, get_model_name

def optimize( X_train, X_val, Y_train, Y_val ):
    # Number of trees in random forest
    def objective(space):
        model = xgb.XGBClassifier(
            n_estimators = int( space[ 'n_estimators' ] ), 
            max_depth = int( space[ 'max_depth' ] ), 
            gamma = space[ 'gamma' ],
            reg_lambda = float( space[ 'reg_lambda' ] ), 
            reg_alpha = float( space[ 'reg_alpha' ] ),
            min_child_weight = int(space[ 'min_child_weight' ] ),
            colsample_bytree = int( space[ 'colsample_bytree' ] ),
            subsample = float( space['subsample'] ),
            objective = space['objective'],
            learning_rate = float( space['learning_rate'] )
        )

        evaluation = [ (X_train, Y_train), ( X_val, Y_val ) ]

        model.fit(X_train, Y_train,
                eval_set = evaluation, eval_metric = "auc",
                early_stopping_rounds = 10, verbose = False)

        pred = model.predict( X_val )
        accuracy = accuracy_score( Y_val, pred > 0.5)
        print("SCORE:", accuracy)
        return {'loss': -accuracy, 'status': STATUS_OK}

    space = {
        'max_depth':  hp.choice('max_depth', np.linspace( start = 1, stop = 50, num = 10, endpoint = True )),
        'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
        'learning_rate': hp.quniform('eta', 0.025, 0.5, 0.025),
        'reg_alpha': hp.quniform('reg_alpha', 0, 1, 0.2 ),
        'reg_lambda': hp.uniform('reg_lambda', 0, 1),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
        'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': hp.quniform( 'n_estimators', 100, 170, 1 ),
        'objective': 'binary:logistic',
        'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
        'seed': 42
    }

    trials = Trials()
    best_hyperparams = fmin(
        fn = objective,
        space = space,
        algo = tpe.suggest,
        max_evals = 100,
        trials = trials
    )
    print("The best hyperparameters are : ", "\n")
    print(best_hyperparams) 


if __name__ == '__main__':
    print('Starting Optimization for XGBOOST')
    num_fold = 1
    data_frame = pd.read_csv(config.TRAIN_CSV_PATH)
    data_frame = config.preprocessing_function['basic']( data_frame, isTrain = True )
    Y = data_frame[['risk_flag']].copy()
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
        X_train, Y_train = config.preprocessing_function['oversample']( X_train, Y_train )
        break
    optimize( X_train, X_val, Y_train, Y_val )