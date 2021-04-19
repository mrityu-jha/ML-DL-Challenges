import pandas as pd
import config
import plot

def cramers_V( data_frame ):
    plot.plot_cramersV( data_frame )

if __name__ == '__main__':
    data_frame = pd.read_csv(config.TRAIN_CSV_PATH)
    data_frame = config.preprocessing_function['basic'](data_frame, isTrain = True )
    cramers_V( data_frame )