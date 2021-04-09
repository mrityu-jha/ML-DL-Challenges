import matplotlib.pyplot as plt
import seaborn as sns
import config
import os 
from sklearn.metrics import plot_roc_curve

def return_model_plot_path( num_fold, TYPE_OF_PLOT ):
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



def plot_accuracy( history, num_fold, isBest = False ):
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

def plot_confusionMatrix( cf_matrix ):
    sns.heatmap( cf_matrix, annot = True, cmap = 'Blues', fmt = 'g' )
    plt.show()

def plot_rocCurve( model, X, Y, num_fold, isBest = False ):
    TYPE_OF_PLOT = 'ROC-CURVE'
    print( "Saving", TYPE_OF_PLOT, "to:", return_model_plot_path( num_fold, TYPE_OF_PLOT ) )
    plot_roc_curve( model, X, Y )
    plt.savefig(return_model_plot_path(num_fold, TYPE_OF_PLOT))
    plt.close()

