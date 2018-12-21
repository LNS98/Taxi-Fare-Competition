"""
Linear simple model for predicting the price: Score of 5.48.
"""

import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import SCORERS, mean_squared_error




def main():

    # read in the data
    df_train = pd.read_csv("../data/train.csv", nrows = 200000)

    # get the new features
    df_new = distance_features(df_train)
    df_new = time_feature(df_new)

    # clean the data
    df_clean = clean_data(df_new)

    # ********************* build a classifier on the training data ***********************
    #
    # define the features used in the model
    features = ['distance_polar','passenger_count', 'year', 'month', 'weekday', 'hour', 'min']

    # train a classifier on the training model
    clf, score = lin_reg(features, df_clean)

    # get a value for the total score
    tot_score = total_score(clf, df_clean, features)

    print(score, tot_score)

    # ************************** predict on the test data **************************

    # import the test data
    df_test = pd.read_csv("../data/test.csv")

    #  create the new features
    df_test = distance_feature(df_test)
    df_test = time_feature(df_test)

    # get the list of predictions
    precict(clf, df_test, features)

    return 0

def read_in_data():
    """
    read in the data just to have a quick look
    """

    df_train = pd.read_csv("../data/train.csv", nrows = 2000000)
    df_test = pd.read_csv("../data/test.csv")

    print(df_train)

    print(list(df_train))
    return None

def clean_data(DataFrame):
    """
    general cleaning techiques to clean the data.
    """
    # copy the df in
    df = DataFrame.copy()

    # print the shape before to compare to after
    print(df.shape)

    # drop any rows where the fare is negative as this makes no sense
    df = df[df.fare_amount >= 2.5]

    # drop any na points
    df.dropna(inplace = True)

    # drop any data outside a NYC

    # first define the geo locations of NYC
    NYC = [-74.5, -72.8, 40.5, 41.8]

    # anyting outside drop
    df = df[(df.pickup_longitude > NYC[0]) & (df.pickup_longitude < NYC[1]) & \
         (df.pickup_latitude > NYC[2]) & (df.pickup_longitude < NYC[3])]

    df = df[(df.dropoff_longitude > NYC[0]) & (df.dropoff_longitude < NYC[1]) & \
            (df.dropoff_latitude > NYC[2]) & (df.dropoff_latitude < NYC[3])]


    # drop any points with distance over 100 miles and bigger than 0.1
    df = df[df.distance_polar < 100]
    # df = df[df.distance > 0.1]

    # drop any distance distance_vector
    # df = df[abs(df.distance_long) < 50]
    # df = df[abs(df.distance_lat) < 50]


    # print the shape at the end
    print(df.shape)

    return df


# ******************  funtions to get distance between points *******************

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees).
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def distance(lon1, lat1, lon2, lat2):
    """
    Other function to claculate the distance between two points as per the kernel.
    """

    p = np.pi / 180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p))/ 2

    distance = 0.6213712 * 12742 * np.arcsin(np.sqrt(a))

    return distance


# ********************** functoins to get more features ************************

def distance_features(DataFrame):
    """
    create distance features.
    """
    # copy df in
    df = DataFrame.copy()

    # create a distance feature
    df["distance_polar"] = [distance(a, b, c, d) for a, b, c, d in zip(df.pickup_longitude, df.pickup_latitude,df.dropoff_longitude, df.dropoff_latitude)]

    # create a more simple distance_vector in both directions
    df["distance_lat"] = (df.dropoff_latitude - df.pickup_latitude)
    df["distance_long"] = (df.dropoff_longitude - df.pickup_longitude)


    return df

def time_feature(DataFrame):
    """
    Create features out of the timestamp features.
    """
    # copy df in
    df = DataFrame.copy()

    # reformat the df to better access timestamp
    df_time = pd.to_datetime(df["pickup_datetime"].str.replace('UTC', ''), format = '%Y-%m-%d %H:%M:%S')

    # get the features of min, hour, weekday, moth and year from the data frame
    df["min"] = df_time.dt.minute
    df["hour"] = df_time.dt.hour

    df["weekday"] = df_time.dt.weekday
    df["month"] = df_time.dt.month
    df["year"] = df_time.dt.year


    return df

# ********************** functoins to classsify the data  ************************

def lin_reg(features, DataFrame):
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
    clf = LinearRegression()

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

# ******************** predict on the data *****************************

def precict(clf, DataFrame, features):
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

    # write the predictions to a csv file
    df_predicitons.to_csv("../submissions/simple_linear_reg.csv", index = False)

    return None



# ******************** visualise the data *****************************

def plot_dist_fare(DataFrame):
    """
    Plot the distacne against the fare.
    """
    # copy df in
    df = DataFrame.copy()

    # get the X and y
    x = df["distance_lat"]
    y = df["distance_long"]


    # plot a scatter graph
    plt.scatter(x, y, s = 2)
    plt.xlabel("distance_lat")
    plt.ylabel("distance_long")

   # # create line with the coefficients
   #  X_fit = np.linspace(0.1, 100, 100)
   #  y_fit = m * X_fit + c

    # to get an idea plot the data against each other
    # plt.plot(X_fit, y_fit)
    plt.show()


    return None



def help():

    # calculate the max distance in NYC
    # first define the geo locations of NYC
    NYC = [-74.5, -72.8, 40.5, 41.8]

    dist = distance(NYC[0], NYC[1], NYC[2], NYC[3])

    print(dist)

    print(sorted(SCORERS.keys()))

    return None



start = time.time()
main()
# help()
print("---------------------- time taken: {} ----------------".format(time.time() - start))
