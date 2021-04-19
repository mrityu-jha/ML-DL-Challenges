from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC
import config
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

def optimize( X, Y ):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=10, stop=200, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {
            'model__n_estimators': n_estimators,
            'model__max_features': max_features,
            'model__max_depth': max_depth,
            'model__min_samples_split': min_samples_split,
            'model__min_samples_leaf': min_samples_leaf,
            'model__bootstrap': bootstrap
        }
    print(random_grid)

    model = RFC( verbose = 2 )
    pipeline_model = Pipeline( [( 'scaler', MinMaxScaler() ), ( 'model', model )] )
    rfc_random = RandomizedSearchCV(estimator=pipeline_model, param_distributions=random_grid,
                                    n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
    rfc_random.fit( X, Y )
    print( rfc_random.best_params_ )
    
    n_estimators = [ 159 ]
    # Number of features to consider at every split
    max_features = ['sqrt']
    # Maximum number of levels in tree
    max_depth = [87, 88]
    # Minimum number of samples required to split a node
    min_samples_split = [2]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [ 8, 9 ]
    # Method of selecting samples for training each tree
    bootstrap = [True]
    # Create the random grid
    random_grid = {
            'model__n_estimators': n_estimators,
            'model__max_features': max_features,
            'model__max_depth': max_depth,
            'model__min_samples_split': min_samples_split,
            'model__min_samples_leaf': min_samples_leaf,
            'model__bootstrap': bootstrap
        }
    print(random_grid)

    model = RFC( verbose = 2 )
    X, Y = config.preprocessing_function['oversample']( X, Y )
    pipeline_model = Pipeline( [( 'scaler', MinMaxScaler() ), ( 'model', model )] )
    rfc_random = GridSearchCV(estimator=pipeline_model, param_grid=random_grid, cv=3, verbose=2, n_jobs=-1)
    rfc_random.fit( X, Y )
    print( rfc_random.best_params_ )
if __name__ == '__main__':
    print('Starting Optimization for RandomForestClassifier')
    data_frame = pd.read_csv(config.TRAIN_CSV_PATH)
    data_frame = config.preprocessing_function['basic'](
        data_frame, isTrain=True)
    X = data_frame.values[ :, :-1 ]
    Y = data_frame[['risk_flag']].copy()
    X, Y = config.preprocessing_function['oversample']( X, Y )
    optimize( X, Y )
