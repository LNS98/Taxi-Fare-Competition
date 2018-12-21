"""
Explore the data
"""

import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import radians, cos, sin, asin, sqrt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import SCORERS, mean_squared_error
from tqdm import tqdm
import xgboost as xgb
import lightgbm as lgb




np.warnings.filterwarnings('ignore')
sns.set_style('whitegrid')


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
    df_train = pd.read_csv("../data/train.csv", nrows = 2_000_000)


    # show the distribution of the data in order to analyse it
    # df_train.describe().T.to_csv('../images/desc_train.csv')

    # get the new features
    df_train = distance_feature(df_train)
    df_train = time_feature(df_train)
    df_train = airports_feature(df_train)
    df_train = which_way_feature(df_train)


    # clean the data
    df_train = clean_data(df_train)

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

def read_in_data():
    """
    read in the data just to have a quick look
    """

    df_train = pd.read_csv("../data/train.csv", nrows = 5_000_000)
    # df_test = pd.read_csv("../data/test.csv")

    print(df_train.info())

    # features which are used for predicting
    features = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
                'distance_polar', 'distance_lat', 'distance_long', 'passenger_count',
                'year', 'month', 'weekday', 'hour', 'min', 'which_way', 'which_air']

    # list to hold the batch dataframe
    df_list = []

    for df_chunk in tqdm(pd.read_csv("../data/train.csv", usecols = list(traintypes.keys()), dtype = traintypes, chunksize = 5000000)):

        # Neat trick from https://www.kaggle.com/btyuhas/bayesian-optimization-with-xgboost
        # Using parse_dates would be much slower!

        # # get the new features
        # df_chunk = distance_feature(df_chunk)
        # df_chunk = time_feature(df_chunk)
        # df_chunk = airports_feature(df_chunk)
        # df_chunk = which_way_feature(df_chunk)
        #
        # # clean the data
        # df_chunk = clean_data(df_chunk)
        #
        # # keep only the useful features for prediciting
        # df_chunk = drop_unwanted(df_chunk, features)
        #
        # Alternatively, append the chunk to list and merge all
        df_list.append(df_chunk)

    # Merge all dataframes into one dataframe
    train_df = pd.concat(df_list)

    # Delete the dataframe list to release memory
    del df_list

    # See what we have loaded
    train_df.info()

    # send it to a pickle file
    # train_df.to_pickle("../data/train_proccessed.pickle")

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


# ********************** functions to get more features ************************

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

def which_airport(lat1, long1, lat2, long2):
    """
    Deterimes whether either of the lat and long coordinates are  situated next
    to an airport in NYC.
    """

    # value deteriming the airport
    which_air = 0


    # define the coordinates of the three airports
    jfk = [40.6397, -73.7789]
    nwr = [40.6925, -74.1686]
    lga = [40.7773, -73.8726]

    # define an error tolereance parameter
    error_tol = 0.05

    # check if the pickup or drop off locations is one of the airports coordinates place a 0, 1, 2, 3
    if (abs(lat1 - jfk[0]) < error_tol and abs(long1 - jfk[1]) < error_tol) or \
       (abs(lat2 - jfk[0]) < error_tol and abs(long2 - jfk[1]) < error_tol):
        which_air = 1

    if (abs(lat1 - nwr[0]) < error_tol and abs(long1 - nwr[1]) < error_tol) or \
       (abs(lat2 - nwr[0]) < error_tol and abs(long2 - nwr[1]) < error_tol):
        which_air = 2

    if (abs(lat1 - lga[0]) < error_tol and abs(long1 - lga[1]) < error_tol) or \
       (abs(lat2 - lga[0]) < error_tol and abs(long2 - lga[1]) < error_tol):
        which_air = 3

    return which_air

def airports_feature(DataFrame):
    """
    Add a column containing if the journey is between an airport and therefore
    might contain a flat fare.
    """
    # copy df in
    df = DataFrame.copy()

    # define the coordinates of the three airports
    jfk = [40.6397, -73.7789]
    nwr = [40.6925, -74.1686]
    lga = [40.7773, -73.8726]

    # list contianing the values of which airport it is
    which_air = []

    # loop over all the values of latitude and long
    for [pickup_lat, pickup_long, dropoff_lat, dropoff_long] \
    in zip(df.pickup_latitude.values, df.pickup_longitude.values, df.dropoff_latitude.values, df.dropoff_longitude.values):
        #  get the airport for the coordinates
        airport = which_airport(pickup_lat, pickup_long, dropoff_lat, dropoff_long)

        # append the airport to the which_air list
        which_air.append(airport)

    # put the list in the df
    df['which_air'] = which_air

    # print(df[['fare_amount', 'which_air']].groupby('which_air').mean())

    return df

def which_way_feature(DataFrame):
    """
    new feature with direction N, S, W, E.
    """
    # copy df in
    df = DataFrame.copy()

    # empty list which will contain new feature
    ways_list = []

    # loop through the directions and check if the lat, long is N, S, W, E
    for lat, long in zip(df.distance_lat.values, df.distance_long.values):
        if lat > 0 and long > 0:
            ways_list.append(1)
        elif lat < 0 and long > 0:
            ways_list.append(2)
        elif lat < 0 and long < 0:
            ways_list.append(3)
        elif lat > 0 and long < 0:
            ways_list.append(4)
        else:
            ways_list.append(-1)


    # append the lsit to the  df
    df["which_way"] = ways_list

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

    print(df.shape)

    # drop any rows where the fare is negative as this makes no sense
    df = df[df.fare_amount >= 2.5]

    print(df.shape)

    # drop any rows where the num of people is more than 6
    df = df[df.passenger_count <= 6]

    print(df.shape)

    # drop any na points
    df.dropna(inplace = True)

    print(df.shape)

    # drop any data outside a NYC

    # first define the geo locations of NYC
    NYC = [-74.5, -72.8, 40.5, 41.8]

    # anyting outside drop
    df = df[(df.pickup_longitude > NYC[0]) & (df.pickup_longitude < NYC[1]) & \
         (df.pickup_latitude > NYC[2]) & (df.pickup_longitude < NYC[3])]

    df = df[(df.dropoff_longitude > NYC[0]) & (df.dropoff_longitude < NYC[1]) & \
            (df.dropoff_latitude > NYC[2]) & (df.dropoff_latitude < NYC[3])]

    print(df.shape)

    # drop any points with distance over 100 miles and bigger than 0.1
    df = df[df.distance_polar < 100]
    # df = df[df.distance > 0.1]

    # drop any distance distance_vector
    # df = df[abs(df.distance_long) < 50]
    # df = df[abs(df.distance_lat) < 50]


    return df

# ********************** functoins to classsify the data  ************************

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
    clf = RandomForestRegressor(n_jobs = -1, n_estimators = 150 , max_depth = 10)  #n_estimators = 100 , max_depth = 10

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
                           n_jobs = -1, verbose = True)

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


# ************************ improve the classifier *********************************

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

# ******************** predict on the data *****************************

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

    # write the predictions to a csv file
    df_predicitons.to_csv("../submissions/XGB_all_feat_more_data.csv", index = False)

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
    df_predicitons.to_csv("../submissions/xgb_lgb_simple_ensembled.csv", index = False)

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

def final_scores(file):
    """
    Plots a scatter of the final scores.
    """



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
