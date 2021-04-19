import numpy as np
import pandas as pd
from scipy.sparse import construct
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, matthews_corrcoef
from imblearn.metrics import classification_report_imbalanced

def accuracy_of_model( y_true, y_pred ):
    print( "The accuracy of the Model is: ", accuracy_score( y_true, y_pred ) )

def classificationReport_of_model( y_true, y_pred ):
    print( "-------------Classification Report using SciKit--------------" )
    print( classification_report( y_true, y_pred ) )
    print("-------------Classification Report using Imblearn--------------")
    print( classification_report_imbalanced( y_true, y_pred ) )

def roc_auc_score_of_model( y_true, y_predProbs ):
    print( 'The ROC Score of the Model is:', roc_auc_score( y_true, y_predProbs, multi_class = 'ovo' ) )


def matthews_corrcoef_score_of_model( y_true, y_pred ):
    print( 'The Matthew Corrcoef of Model is:', matthews_corrcoef( y_true, y_pred ) )

def score_of_model( model, X, Y, mode = '' ):
    score = model.score(X, Y)
    print( "The", mode, "Score of the Model is:", score )
    return score

def score_of_model_for_neuralNet( model, X, Y, mode = '' ):
    _, score = model.evaluate(X, Y)
    print( "The", mode, "Score of the Model is:", score )
    return score