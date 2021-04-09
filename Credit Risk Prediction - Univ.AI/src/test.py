import config
import numpy as np
import os
import config
import pandas as pd
import joblib
from datetime import datetime

def predict_separate( model_path, test_df ):
    print( "Loading Model from: ", model_path )
    name_of_model = os.path.basename( model_path ).split('_')[0]
    model = joblib.load( model_path )
    test_data = config.preprocessing_function[name_of_model]( test_df.values, model_path, isTrain = False )
    y_predProbs = model.predict_proba(test_data)  
    return y_predProbs


def return_submission_file_path():
    now = datetime.now()
    print( "Saving Submission File To:", os.path.join( config.SUBMISSION_PATH, now.strftime( "%H-%M-%S %d %b %Y" ) + '.csv' ) )
    return os.path.join( config.SUBMISSION_PATH, now.strftime( "%H-%M-%S %d %b %Y" ) + '.csv' )

def predict_ensemble():
    y_predProbs = np.zeros( 1, dtype = float )
    test_df = pd.read_csv( config.TEST_CSV_PATH )
    id_col = test_df[['id']].copy()
    test_df = config.preprocessing_function['basic'](test_df, isTrain = False )
    for path in os.listdir( config.SAVE_MODEL ):
        if 'preprocess' not in os.path.join( config.SAVE_MODEL, path ):
            y_predProbs = y_predProbs + predict_separate(
                os.path.join(config.SAVE_MODEL, path),
                test_df 
            )
    y_pred = np.argmax( y_predProbs, axis=-1 )
    results = pd.DataFrame( pd.concat( [id_col, pd.DataFrame( y_pred, columns = ['risk_flag']  )], axis = 1 ), columns=['id', 'risk_flag'] )
    results.to_csv( return_submission_file_path(), index = False )
    # evaluation.accuracy_of_model( y_true, y_pred )
    # evaluation.confusionMatrix_of_model( y_true, y_pred )
    # evaluation.classificationReport_of_model( y_true, y_pred )
if __name__ == '__main__':
    predict_ensemble()
