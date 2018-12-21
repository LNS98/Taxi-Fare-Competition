"""
Explore the data
"""

import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import SCORERS, mean_squared_error
from tqdm import tqdm


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

    # read in the data
    df_train = pd.read_pickle("../data/train_proccessed.pickle")

    print("data read in")
    print(df_train.info())
    print("ready to train")

    # ********************* build a classifier on the training data ***********************
    #
    # define the features used in the model
    features = ['distance_polar','passenger_count', 'year', 'month', 'weekday', 'hour', 'min']

    # get the train and test score for the classifier
    clf = train_test_lin_reg(df_train, features)

    # print the feature importance
    # imp_features(clf, df_clean, features)

    # train a classifier on the training model
    # clf, score = cv_rand_for(df_clean, features)

    # get a value for the total score
    tot_score = total_score(clf, df_train, features)

    print("total RMSE error: {}".format(tot_score))

    # get the best feature for the random forest classifier
    # best_paramaters(df_clean, features)

    # ************************** predict on the test data **************************

    # # import the test data
    # df_test = pd.read_csv("../data/test.csv")
    #
    # #  create the new features
    # df_test = distance_feature(df_test)
    # df_test = time_feature(df_test)
    #
    # # get the list of predictions
    # precict(clf, df_test, features)

    return 0

def read_in_data():
    """
    read in the data just to have a quick look
    """

    df_train = pd.read_csv("../data/train.csv", nrows = 2000000)
    df_test = pd.read_csv("../data/test.csv")

    # features which are used for predicting
    features = ['distance_polar','passenger_count', 'year', 'month', 'weekday', 'hour', 'min', 'fare_amount']

    # list to hold the batch dataframe
    df_list = []

    for df_chunk in tqdm(pd.read_csv("../data/train.csv", usecols = list(traintypes.keys()), dtype = traintypes, chunksize = 5000000)):

        # Neat trick from https://www.kaggle.com/btyuhas/bayesian-optimization-with-xgboost
        # Using parse_dates would be much slower!

        # get the new features
        df_chunk = distance_feature(df_chunk)
        df_chunk = time_feature(df_chunk)

        # clean the data
        df_chunk = clean_data(df_chunk)

        # keep only the useful features for prediciting
        df_chunk = drop_unwanted(df_chunk, features)

        # Alternatively, append the chunk to list and merge all
        df_list.append(df_chunk)

    # Merge all dataframes into one dataframe
    train_df = pd.concat(df_list)

    # Delete the dataframe list to release memory
    del df_list

    # See what we have loaded
    train_df.info()

    # send it to a pickle file
    train_df.to_pickle("../data/train_proccessed.pickle")

    return None

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

def distance_feature(DataFrame):
    """
    create distance features.
    """
    # copy df in
    df = DataFrame.copy()

    # create a distance feature
    df["distance_polar"] = [distance(a, b, c, d) for a, b, c, d in zip(df.pickup_longitude, df.pickup_latitude,df.dropoff_longitude, df.dropoff_latitude)]

    # create a more simple distance_vector in both directions
    df["distance_lat"] = (df.dropoff_latitude - df.pickup_latitude).astype('float32')
    df["distance_long"] = (df.dropoff_longitude - df.pickup_longitude).astype('float32')

    return df

def time_feature(DataFrame):
    """
    Create features out of the timestamp features.
    """
    # copy df in
    df = DataFrame.copy()

    # reformat the df to better access timestamp
    df['pickup_datetime'] = df['pickup_datetime'].str.slice(0, 16)
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M')

    # get the features of min, hour, weekday, moth and year from the data frame
    df["min"] = df.pickup_datetime.dt.minute.astype('int8')
    df["hour"] = df.pickup_datetime.dt.hour.astype('int8')

    df["weekday"] = df.pickup_datetime.dt.weekday.astype('int8')
    df["month"] = df.pickup_datetime.dt.month.astype('int8')
    df["year"] = df.pickup_datetime.dt.year.astype('int32')

    return df

def drop_unwanted(DataFrame, features):
    """
    drop all the uneccessary features needed for predicting
    """
    # copy df in
    df = DataFrame.copy()

    # keep only the wanted ones
    df = df[features]

    return df

def clean_data(DataFrame):
    """
    general cleaning techiques to clean the data.
    """
    # copy the df in
    df = DataFrame.copy()

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


    return df

# ********************** functoins to classsify the data  ************************

def cv_lin_reg(DataFrame, features):
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

def train_test_lin_reg(DataFrame, features):
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
    df_predicitons.to_csv("../submissions/lin_reg_all_data.csv", index = False)

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

def imp_features(clf, DataFrame, features):
    """
    Get the most important features in the model.
    """
    # get the most important features
    feature_importances = pd.DataFrame({'feature': features,'importance': clf.feature_importances_}).sort_values('importance', ascending = False)

    print(feature_importances)

    # plot a bar chart of feature feature importance
    scalers = feature_importances['feature']
    heights = feature_importances['importance']

    plt.bar(scalers, heights)
    plt.xlabel('features')
    plt.ylabel('importance')
    plt.show()

    return None



# *********************** less important fucntions *****************************

def help():

    # calculate the max distance in NYC
    # first define the geo locations of NYC
    NYC = [-74.5, -72.8, 40.5, 41.8]

    dist = distance(NYC[0], NYC[1], NYC[2], NYC[3])

    print(dist)

    print(sorted(SCORERS.keys()))

    return None


# run the algorithm
start = time.time()
main()
# read_in_data()
print("---------------------- time taken: {} ----------------".format(time.time() - start))
