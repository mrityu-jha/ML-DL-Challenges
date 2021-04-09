import os
import preprocess

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).split( 'src' )[0]
INPUT = os.path.join( ROOT_DIR, 'input' )
CSV_PATH = os.path.join( INPUT, 'csvs' )
TRAIN_CSV_PATH = os.path.join(CSV_PATH, 'train.csv')
TEST_CSV_PATH = os.path.join(CSV_PATH, 'test.csv')
SAVE_MODEL = os.path.join( ROOT_DIR, 'models' )
LOGS = os.path.join( ROOT_DIR, 'logs' )
PLOTS_PATH = os.path.join( ROOT_DIR, 'plots' )
SUBMISSION_PATH = os.path.join( ROOT_DIR, 'submissions' )

BATCH_SIZE = 128
MODELS = [ 'adaboost', 'rfc' ]
NUM_SPLITS = 5

CLASS_LABEL = { 0 : "Not Risky", 1 : "Risky" }
NUM_CLASSES = len( CLASS_LABEL )

preprocessing_function = {
    'basic' : preprocess.basic,
    'svc' : preprocess.svc,
    'rfc' : preprocess.rfc,
    'adaboost' : preprocess.adaboost
}
