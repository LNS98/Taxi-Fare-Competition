"""
Functions used ton visualise the data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

def plot_dist_fare(DataFrame):
    """
    Plot the distacne against the fare.
    """
    # copy df in
    df = DataFrame.copy()

    # # get the X and y
    # x = df["distance_lat"]
    # y = df["distance_long"]

    # plot a scatter graph
    sns.lmplot('distance_polar', 'fare_amount', size = 8,
           fit_reg = False, data = df.sample(10000))

    # plot a scatter graph
    # plt.scatter(x, y, s = 2)
    # plt.xlabel("distance_lat")
    # plt.ylabel("distance_long")

   # # create line with the coefficients
   #  X_fit = np.linspace(0.1, 100, 100)
   #  y_fit = m * X_fit + c

    # to get an idea plot the data against each other
    # plt.plot(X_fit, y_fit)
    plt.show()


    return None

def imp_features(clf, DataFrame, features):
    """
    Get the most important features in the model.
    """
    # get the most important features
    feature_importances = pd.DataFrame({'feature': features,'importance': clf.feature_importances_}).set_index('feature').sort_values('importance', ascending = True)

    feature_importances.plot(kind="barh", color = 'b')

    # plot a bar chart of feature feature importance
    # scalers = feature_importances['feature']
    # heights = feature_importances['importance']
    #
    # plt.bar(scalers, heights)
    # plt.xlabel('features')
    # plt.ylabel('importance')
    plt.show()

    return None

def correlations(DataFrame):
    """
    Plot the correlations between the features and the target.
    """
    # copy df in
    df = DataFrame.copy()

    # get the correlations
    corrs = df.corr()

    # get only the correlations for the target variable
    corrs_target = corrs['fare_amount'].sort_values(ascending = True)

    # corrs['fare_amount'].plot.bar(color = 'b') annot = True, vmin = -1, vmax = 1,

    # plot the total heat map
    # plt.figure(figsize = (12, 12))
    # sns.heatmap(corrs, annot = True, fmt = '.3f')
    # plt.show()

    # barplot of the correlations between fare amount and features
    corrs_target.plot(kind="barh", color = 'b')
    # sns.barplot(x = corrs_target.index.values, y = corrs_target.values)
    plt.show()


    return None

