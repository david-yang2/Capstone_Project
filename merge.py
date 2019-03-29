import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


trips = pd.read_csv("data/201812-fordgobike-tripdata.csv")
weather = pd.read_csv('data/weather.csv')


def num_malfunctions(df):
    '''
    INPUT: Dataframe with a "malfunction" column
    Sum the number of malfunctions up
    OUTPUT: Tuple with 
            first element as number of malfunctions and
            second element as number of non-malfunctions
    '''

    num_malfunctions = df.malfunction.sum()
    num_working = len(df.malfunction) - num_malfunctions

    return (num_malfunctions, num_working)

def frequent_malfunction(df):
    return df.bike_id[df.malfunction == True].value_counts()

def same_station(df):
    return df.bike_id[df.start_station_name == df.end_station_name].value_counts()

def model_city(df, city = 1):
    '''
    Breaks the dataset into 3 cities
    SF = 1
    OAK = 2
    SJ = 3

    INPUT: Dataframe
           Number for city to be modeled
    OUTPUT: returns a portion of the original dataframe
    ''' 
    if city == 1:
        return df[(df.start_station_latitude > 37.697799) & (df.start_station_longitude <-122.330676)]
    elif city == 2:
        return df[(df.start_station_latitude > 37.697799) & (df.start_station_longitude >-122.330676)]
    elif city == 3:
        return df[df.start_station_latitude < 37.697799]

def trips_weather_combined(df1, df2):
    '''
    INPUT df1 is the trips dataframe
          df2 is the weather dataframe
    OUTPUT 1 combined dataframe
    '''


def merge_dfs(df1, df2, column = 'date'):
    '''
    INPUT:
    df1 = trips df
    df2 = weather df

    OUTPUT: combined df

    '''
    #weather dates are type str
    #need to convert to datetime before we can merge with trips df
    #because dates column in trips df is datetime
    df2[column] = pd.to_datetime(df2[column])
    df2[column] = df2[column].dt.date
    combined_df = pd.merge(df1,df2, on='date', how='left')
    return combined_df
