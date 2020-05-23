"""
Functions used to predict on new data
"""
 
import pandas as pd
import numpy as np

def predict(clf, DataFrame, features):
    """
    Predict on the given df with the clf and the features provided.
    """
    # copy df in
    df = DataFrame.copy()

    # as X values get the features in the df
    X = df[features]

    # use the clf to predict on the X
    predictions = clf.predict(X)

    predictions = predictions.ravel()

    # get the key from the df and the predictions and put them in a dictionary to then put in a df
    d = {"key": df.key.values, "fare_amount": predictions}

    # create a df with that data
    df_predicitons = pd.DataFrame(data = d)

    # write the predictions to a csv file, must specify the path such that it matches the classifier
    df_predicitons.to_csv("../results/XGB_all_feat_more_data.csv", index = False)

    return None

def precict_ensamble(clfs, DataFrame, features):
    """
    Predict on the given df with the clf and the features provided.
    """
    # copy df in
    df = DataFrame.copy()

    # as X values get the features in the df
    X = df[features]

    # get a prediciton list
    predictions_list = []

    # predict over every clf and average the result
    for clf in clfs:
        # use the clf to predict on the X
        predictions = clf.predict(X)
        predictions = predictions.ravel()
        predictions_list.append(predictions)

    predictions_list = np.array(predictions_list)

    print(predictions_list.shape)

    aver = np.mean(predictions_list, axis = 0)

    print(aver.shape)

    # get the key from the df and the predictions and put them in a dictionary to then put in a df
    d = {"key": df.key.values, "fare_amount": aver}

    # create a df with that data
    df_predicitons = pd.DataFrame(data = d)

    # write the predictions to a csv file
    df_predicitons.to_csv("../results/xgb_lgb_simple_ensembled.csv", index = False)

    return None

