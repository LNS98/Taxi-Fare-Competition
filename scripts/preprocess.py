"""
File that includes some function used to pre-process the data.
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

from utils import distance 

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

# --------------- functions to add more features ---------------------

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

