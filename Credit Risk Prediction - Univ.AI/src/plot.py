from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import seaborn as sns
import config
import os 
from sklearn.metrics import plot_roc_curve, confusion_matrix
import numpy as np
import dython

def return_model_plot_path( num_fold, TYPE_OF_PLOT ):
    if( num_fold == 'NORMAL' ):
        return os.path.join( config.PLOTS_PATH, TYPE_OF_PLOT + '.jpg' )
    else:
        return os.path.join( config.PLOTS_PATH, config.MODELS[num_fold - 1] + '_' + str( num_fold ) + '_' + TYPE_OF_PLOT + '.jpg' )

def plot_loss( history, num_fold ):
    TYPE_OF_PLOT = 'LOSS'
    plt.figure( figsize = ( 8, 7 ) )
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss', fontsize = 14 )
    plt.ylabel('Loss', fontsize = 18 )
    plt.xlabel('Epoch', fontsize = 17 )
    plt.xticks( fontsize = 14 )
    plt.yticks( fontsize = 14 )
    plt.legend( ['Train Set', 'Val Set'], loc='best', fontsize = 14 )
    print( "Saving", TYPE_OF_PLOT, "to:", return_model_plot_path( num_fold, TYPE_OF_PLOT ) )
    plt.savefig( return_model_plot_path( num_fold, TYPE_OF_PLOT ) )
    plt.close() 



def plot_accuracy( history, num_fold):
    TYPE_OF_PLOT = 'ACCURACY'
    plt.figure( figsize = ( 8, 7 ) )
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy', fontsize = 14 )
    plt.ylabel('Accuracy', fontsize = 18 )
    plt.xlabel('Epoch', fontsize = 17 )
    plt.xticks( fontsize = 14 )
    plt.yticks(fontsize = 14 )
    plt.legend( ['Train Set', 'Val Set'], loc='best', fontsize = 14 )
    print( "Saving", TYPE_OF_PLOT, "to:", return_model_plot_path( num_fold, TYPE_OF_PLOT ) )
    plt.savefig( return_model_plot_path( num_fold, TYPE_OF_PLOT ) )
    plt.close()

def plot_confusionMatrix( y_true, y_pred, num_fold ):
    TYPE_OF_PLOT = 'CONFUSION-MATRIX'
    sns.heatmap( confusion_matrix(y_true, y_pred), annot=True, cmap='Blues', fmt='g')
    plt.savefig(return_model_plot_path( num_fold, TYPE_OF_PLOT))
    plt.close()

def plot_rocCurve( model, X, Y, num_fold ):
    TYPE_OF_PLOT = 'ROC-CURVE'
    print( "Saving", TYPE_OF_PLOT, "to:", return_model_plot_path( num_fold, TYPE_OF_PLOT ) )
    plot_roc_curve( model, X, Y )
    plt.savefig(return_model_plot_path(num_fold, TYPE_OF_PLOT))
    plt.close()

def plot_featureImportance( model, num_fold, col_names ):
    TYPE_OF_PLOT = 'FEATURE-IMPORTANCE'
    importance = model.feature_importances_
    idxs = np.argsort(importance)
    plt.title('Feature Importances')
    plt.figure( figsize = ( 25, 40 ))
    plt.barh(range(len(idxs)), importance[idxs], align='center')
    plt.yticks(range(len(idxs)), [col_names[i] for i in idxs], size = 5 )
    plt.xlabel('XGBOOST Feature Importance')
    print( "Saving", TYPE_OF_PLOT, "to:", return_model_plot_path( num_fold, TYPE_OF_PLOT ) )
    plt.savefig(return_model_plot_path(num_fold, TYPE_OF_PLOT))
    plt.close()

def plot_cramersV( data_frame, num_fold = 'NORMAL' ):
    TYPE_OF_PLOT = 'CRAMERS-V'
    fig, ax = plt.subplots( figsize = ( 10, 10 ))
    dython.nominal.associations( data_frame, ax = ax, plot = False )
    plt.close()
    print( "Saving", TYPE_OF_PLOT, "to:", return_model_plot_path( num_fold, TYPE_OF_PLOT ) )
    fig.savefig(return_model_plot_path(num_fold, TYPE_OF_PLOT))