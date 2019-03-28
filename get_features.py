import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def feature_addition(df):
    '''
    INPUT dataframe
    OUTPUT dataframe with extra features
    '''
    #create a new column that shows the day of week for each trip
    #need to transform panda series to datetime
    #before we can use .dt.dayofweek
    #monday = 0 and sunday = 6

    df.start_time = pd.to_datetime(df.start_time)

    df['day_of_week'] = df.start_time.dt.dayofweek

    df['date'] = df.start_time.dt.date

    #any trip with less than 90 seconds and where
    #start station IS the end station,
    #we make the assumption that its malfunctioned
    df['malfunction'] = (df.duration_sec < 90) & (df.start_station_name == df.end_station_name)
    df['age'] = 2019 - df.member_birth_year

    return df

def get_dummies(df):
    '''
    '''
    #start station dummies
    start_dummies = pd.get_dummies(df.start_station_name)
    #end station dummies
    end_dummies = pd.get_dummies(df.end_station_name)
    df = pd.concat([df,start_dummies], axis=1)
    df = pd.concat([df,end_dummies], axis=1)
    
    return df

def avg_duration_per_trip(df):
    unique_start_sations = df.start_station_id.unique()
    unique_end_stations = df.end_station_id.unique()
    for s in unique_start_sations[:50]:
        for e in unique_end_stations[:50]:
            df['avg_duration'][(df.start_station_id == s) & (df.end_station_id == e)] = df[(df.start_station_id == s) & (df.end_station_id == e)]['duration_sec'].mean()
        