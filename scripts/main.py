"""
Main algorithm.
"""

import numpy as np 
import pandas as pd 

from preprocess import distance_feature, time_feature, airports_feature, which_way_feature, clean_data
from visualisations import plot_dist_fare, imp_features
from models import simple_xgb
from inference import predict

np.warnings.filterwarnings('ignore')


# Set columns to most suitable type to optimize for memory usage
traintypes = {'fare_amount': 'float32',
              'pickup_datetime': 'str',
              'pickup_longitude': 'float32',
              'pickup_latitude': 'float32',
              'dropoff_longitude': 'float32',
              'dropoff_latitude': 'float32',
              'passenger_count': 'uint8'}


def main():

    # # read in the data
    # df_train = pd.read_pickle("../data/train_proccessed.pickle")
    #
    # print("data read in")
    # print(df_train.info())
    # print("ready to train")

    # # ************************* Not whole data set ****************************
    df_train = pd.read_csv("../data/train.csv", nrows = 100_000)


    # show the distribution of the data in order to analyse it
    # df_train.describe().T.to_csv('../images/desc_train.csv')

    # get the new features
    df_train = distance_feature(df_train)
    df_train = time_feature(df_train)
    df_train = airports_feature(df_train)
    df_train = which_way_feature(df_train)

    # clean the data
    df_train = clean_data(df_train)
    
    print(df_train.head())

    # plot the distribution of the distances
    plot_dist_fare(df_train)

    # plot the correlations between the features and the targets
    # correlations(df_train)

    # ********************* build a classifier on the training data ***********************
    #
    # define the features used in the model
    features = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
                'distance_polar', 'distance_lat', 'distance_long', 'passenger_count',
                'year', 'month', 'weekday', 'hour', 'min', 'which_way', 'which_air']

# ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
#             'distance_polar', 'distance_lat', 'distance_long', 'passenger_count',
#             'year', 'month', 'weekday', 'hour', 'min', 'which_way', 'which_air']

    # get the train and test score for the classifier
    # clf = train_test_lin_reg(df_train, features)
    # clf = train_test_dec_tree(df_train, features)
    # clf = train_test_rand_for(df_train, features)
    clf_xgb = simple_xgb(df_train, features)
    # clf_lgb = simple_lgb(df_train, features)
    # clf = MLP(df_train, features)

    # put the cls in a list
    # clfs = [clf_xgb, clf_lgb]

    # print the feature importance
    # imp_features(clf, df_train, features)

    # train a classifier on the training model
    # clf, score = cv_rand_for(df_clean, features)

    # get a value for the total score
    # tot_score = total_score(clf, df_train, features)
    #
    # print("total RMSE error: {}".format(tot_score))

    # *************************  improve score of classifiers *****************************8

    # see the row against the error plot
    # row_score('lgb', df_train, features)

    # get the best feature for the random forest classifier
    # best_paramaters(df_clean, features)

    # ************************** predict on the test data **************************

    # import the test data
    df_test = pd.read_csv("../data/test.csv")

    #  create the new features
    df_test = distance_feature(df_test)
    df_test = time_feature(df_test)
    df_test = airports_feature(df_test)
    df_test = which_way_feature(df_test)

    # get the list of prediction
    predict(clf_xgb, df_test, features)

    # get the list of predictions given the ensabled clfs
    # precict_ensamble(clfs, df_test, features)

    return 0


if __name__ == "__main__":
    main()
