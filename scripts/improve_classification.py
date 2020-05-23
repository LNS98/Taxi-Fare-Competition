"""
Function used to try and improve the accuracy of the classifier 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import SCORERS, mean_squared_error
from tqdm import tqdm

def row_score(clf_type, DataFrame, features):
    """
    See how the score for the train and test data changes as you increase the
    number of rows included in the data.
    """
    # copy the df in
    df = DataFrame.copy()

    # get a 2_000_000 row sample as a validation set

    df_validate = df.sample(2_000_000)
    # get the features and target for the whole data
    X_total = df_validate[features]
    y_total = df_validate["fare_amount"]

    # divide the  data in train and test
    X_train_total, X_test_total, y_train_total, y_test_total = train_test_split(X_total, y_total, test_size = 0.33)

    # define empty lsit which will contain all the scores for each round
    scores = []
    num_rows_list =[]

    # loop over each element in lsit
    for num_rows in range(100, 10_000_001, 1_000_000):

        # append the row list for the x axis later  on
        num_rows_list.append(num_rows)

        # sample the number of rows in the df as num_rows
        df_cur = df.sample(num_rows)

        print(df_cur.info())

        # get the features and target for the data
        X = df_cur[features]
        y = df_cur["fare_amount"]

        # divide the  data in train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

        if clf_type == 'random_forest':
            # initialise the classifier
            clf = RandomForestRegressor(n_jobs = -1, n_estimators = 100 , max_depth = 10)

        elif clf_type == 'linear_regression':
            # initialise the classifier
            clf = LinearRegression()

        elif clf_type == 'xgb':
            train_dmatrix = xgb.DMatrix(data = X_train, label = y_train)
            test_dmatrix = xgb.DMatrix(data = X_test, label = y_test)

            # initialise the classifier
            clf = xgb.XGBRegressor(objective ='reg:linear', num_boost_round = 20, max_depth = 10, n_estimators = 100, n_jobs = -1)

        elif clf_type == 'lgb':
            clf = lgb.LGBMRegressor(objective='regression', num_leaves = 35, n_estimators = 300)

        else:
            print("classifier type doesn't exist!")
            return None

        # train on the test data
        clf.fit(X_train, y_train)

        # predict on the train and test
        train_predict = clf.predict(X_train_total)
        test_predict = clf.predict(X_test_total)

        # get the RMSE score for both
        score_train = np.sqrt(mean_squared_error(y_train_total, train_predict))
        score_test = np.sqrt(mean_squared_error(y_test_total, test_predict))

        # append the score to the scores lsit
        scores.append([score_train, score_test])

        print(scores)

    # convert scores to a  numpy array
    scores = np.array(scores)


    # plot the scores against the num_rows for both training and testing
    plt.scatter(num_rows_list, scores[:, 0], s = 4, label = 'training error')
    plt.scatter(num_rows_list, scores[:, 1], s = 4, label = 'test error')
    plt.title('error variation as a fucntion of number of rows for {}'.format(clf_type))
    plt.xlabel('number of rows used')
    plt.ylabel('RMSE error')
    plt.legend()
    plt.show()

    return None

def best_paramaters(DataFrame, features):
    """
    find the best parameters using RandomizedSearchCV for the model.
    """
    # copy the df in
    df = DataFrame.copy()

    # define the clf
    clf = RandomForestRegressor()

    # second set of parameters
    params = {'max_depth' : [10, 20],
              'n_estimators': [10, 100, 150, 200]}

    # create the randominzed grid for the classifier to find.
    # GridSearchCV
    clf_random = RandomizedSearchCV(estimator = clf, param_distributions = params, cv = 2, verbose = 2, n_jobs = -1)


    # know fit the clf to the data

    # get the features and target for the data
    X = df[features]
    y = df["fare_amount"]

    # divide the  data in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

    # train on the test data
    clf_random.fit(X_train, y_train)

    # show the best parameters
    best_par = clf_random.best_params_

    print(best_par)

    # predict on the train and test
    train_predict = clf_random.predict(X_train)
    test_predict = clf_random.predict(X_test)

    # get the RMSE score for both
    score_train = np.sqrt(mean_squared_error(y_train, train_predict))
    score_test = np.sqrt(mean_squared_error(y_test, test_predict))

    print("train RMSE: {}\ntest RMSE: {}".format(score_train, score_test))

    # train the clf on the whole data
    # clf_random.fit(X, y)

    return None

