"""
Functions used to classify the data.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb




def cv_rand_for(DataFrame, features):
    """
    run a linear regression model by applying 10 fold cross validation on the X, y
    data.
    """
    # copy the dataframe
    df = DataFrame.copy()

    # fill or drop any na valus
    df.dropna(inplace = True)
    # df.fillna(-1, inplace = True)

    # get the target label
    target = ["fare_amount"]

    # get the X and y values
    X = np.array(df[features])
    y = np.array(df[target])

    # reshape to comply with sklearn specifications
    # X = X.reshape(-1, 1)

    # initialise your classifier
    clf = RandomForestRegressor()

    # fit the clf to the data
    clf.fit(X, y)

    # get the mean score from cross validation
    score = cross_val_score(clf, X, y, cv = 10, scoring = "neg_mean_absolute_error").mean()


    return clf, score

def total_score(clf, DataFrame, features):
    """
    Gets the total score on the training set
    """

    # copy df in
    df = DataFrame.copy()

    # fill or drop any na valus
    df.dropna(inplace = True)
    # df.fillna(-1, inplace = True)

    # get the x and y
    X = df[features]
    y = np.array(df.fare_amount).reshape(-1, 1)

    # get the predictions on thhe X
    predict = clf.predict(X)

    # get the score of the clf
    score = np.sqrt(mean_squared_error(predict, y))

    return score

def train_test_rand_for(DataFrame, features):
    """
    create a random forest classifier by dividing the data in the test and train
    to see if there is any overfitting
    """
    # copy df in
    df = DataFrame.copy()

    # get the features and target for the data
    X = df[features]
    y = df["fare_amount"]

    # divide the  data in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

    # initialise the classifier
    clf = RandomForestRegressor(n_jobs = 1, n_estimators = 150 , max_depth = 10)  #n_estimators = 100 , max_depth = 10

    # train on the test data
    clf.fit(X_train, y_train)

    # predict on the train and test
    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)

    # get the RMSE score for both
    score_train = np.sqrt(mean_squared_error(y_train, train_predict))
    score_test = np.sqrt(mean_squared_error(y_test, test_predict))

    print("train RMSE: {}\ntest RMSE: {}".format(score_train, score_test))

    # train the clf on the whole data
    clf.fit(X, y)

    return clf

def train_test_lin_reg(DataFrame, features):
    """
    create a Linear regression classifier by dividing the data in the test and train
    to see if there is any overfitting
    """
    # copy df in
    df = DataFrame.copy()

    # get the features and target for the data
    X = df[features]
    y = df["fare_amount"]

    # divide the  data in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

    # initialise the classifier
    clf = LinearRegression()

    # train on the test data
    clf.fit(X_train, y_train)

    # predict on the train and test
    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)

    # get the RMSE score for both
    score_train = np.sqrt(mean_squared_error(y_train, train_predict))
    score_test = np.sqrt(mean_squared_error(y_test, test_predict))

    print("train RMSE: {}\ntest RMSE: {}".format(score_train, score_test))

    # train the clf on the whole data
    clf.fit(X, y)

    return clf

def train_test_dec_tree(DataFrame, features):
    """
    create a Linear regression classifier by dividing the data in the test and train
    to see if there is any overfitting
    """
    # copy df in
    df = DataFrame.copy()

    # get the features and target for the data
    X = df[features]
    y = df["fare_amount"]

    # divide the data in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

    # initialise the classifier
    clf = DecisionTreeRegressor(max_depth = 10)

    # train on the test data
    clf.fit(X_train, y_train)

    # predict on the train and test
    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)

    # get the RMSE score for both
    score_train = np.sqrt(mean_squared_error(y_train, train_predict))
    score_test = np.sqrt(mean_squared_error(y_test, test_predict))

    print("train RMSE: {}\ntest RMSE: {}".format(score_train, score_test))

    # train the clf on the whole data
    clf.fit(X, y)

    return clf

def simple_xgb(DataFrame, features):
    """
    First simple go at using a XGB classifier
    """
    # copy df in
    df = DataFrame.copy()

    # get the features and target for the data
    X = df[features]
    y = df["fare_amount"]

    # divide the  data in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

    train_dmatrix = xgb.DMatrix(data = X_train, label = y_train)
    test_dmatrix = xgb.DMatrix(data = X_test, label = y_test)

    # initialise the classifier
    clf = xgb.XGBRegressor(objective ='reg:linear', early_stopping = 15,
                           num_boost_round = 1000, max_depth = 10, n_estimators = 100,
                           n_jobs = 1, verbose = True)

    # train on the test data
    clf.fit(X_train, y_train)

    # predict on the train and test
    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)

    # get the RMSE score for both
    score_train = np.sqrt(mean_squared_error(y_train, train_predict))
    score_test = np.sqrt(mean_squared_error(y_test, test_predict))

    print("train RMSE: {}\ntest RMSE: {}".format(score_train, score_test))

    # train the clf on the whole data
    clf.fit(X, y)

    return clf

def simple_lgb(DataFrame, features):
    """
    First simple go at using a XGB classifier
    """
    # copy df in
    df = DataFrame.copy()

    # get the features and target for the data
    X = df[features]
    y = df["fare_amount"]

    # divide the  data in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

    # initialise the classifier
    clf = lgb.LGBMRegressor(objective='regression', num_leaves = 35, n_estimators = 300)

    # fit the classifier
    clf.fit(X_train, y_train)

    # predict on the train and test
    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)

    # get the RMSE score for both
    score_train = np.sqrt(mean_squared_error(y_train, train_predict))
    score_test = np.sqrt(mean_squared_error(y_test, test_predict))

    print("train RMSE: {}\ntest RMSE: {}".format(score_train, score_test))

    # train the clf on the whole data
    clf.fit(X, y)

    return clf

def MLP(DataFrame, features):
    """
    First simple go at using a XGB classifier
    """
    # copy df in
    df = DataFrame.copy()

    # get the features and target for the data
    X = df[features]
    y = df["fare_amount"]

    # divide the  data in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

    # initialise the classifier
    clf = MLPRegressor()

    # fit the classifier
    clf.fit(X_train, y_train)

    # predict on the train and test
    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)

    # get the RMSE score for both
    score_train = np.sqrt(mean_squared_error(y_train, train_predict))
    score_test = np.sqrt(mean_squared_error(y_test, test_predict))

    print("train RMSE: {}\ntest RMSE: {}".format(score_train, score_test))

    # train the clf on the whole data
    clf.fit(X, y)

    return clf

