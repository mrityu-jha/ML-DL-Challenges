import os
import preprocess

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).split( 'src' )[0]
INPUT = os.path.join( ROOT_DIR, 'input' )
CSV_PATH = os.path.join( INPUT, 'csvs' )
TRAIN_CSV_PATH = os.path.join(CSV_PATH, 'train_modified.csv')
TEST_CSV_PATH = os.path.join(CSV_PATH, 'test_modified.csv')
SAVE_MODEL = os.path.join( ROOT_DIR, 'models' )
LOGS = os.path.join( ROOT_DIR, 'logs' )
PLOTS_PATH = os.path.join( ROOT_DIR, 'plots' )
SUBMISSION_PATH = os.path.join( ROOT_DIR, 'submissions' )

BATCH_SIZE = 64
MODELS = [ 'xgbc' ]
FOLD_OF_neuralNet = [ ( i + 1 ) for i in range( 0, len( MODELS ) ) if MODELS[i]  == 'neuralNet' ]
NUM_SPLITS = 5

CLASS_LABEL = { 0 : "Not Risky", 1 : "Risky" }
CLASS_WEIGHT = { 0 : 1., 1 : 5. }
NUM_CLASSES = len( CLASS_LABEL )

INPUT_SHAPE_FOR_neuralNet = ( 36 )

preprocessing_function = {
    'basic' : preprocess.basic,
    'oversample' : preprocess.oversample,
    'svc' : preprocess.svc,
    'rfc' : preprocess.rfc,
    'adaboost' : preprocess.adaboost,
    'xgbc' : preprocess.xgbc,
    'ctbc' : preprocess.ctbc,
    'neuralNet' : preprocess.neuralNet
}
