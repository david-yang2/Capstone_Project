import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def load_data():
    data = pd.read_csv('data/master-fordgobike-tripdata')
    data.drop(["Unnamed: 0"],axis=1, inplace=True)
    weather = pd.read_csv('data/weather.csv')
    return data, weather

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
        return df[(df.end_station_latitude > 37.697799)     \
                & (df.end_station_longitude <-122.330676)   \
                & (df.start_station_latitude > 37.697799)   \
                & (df.start_station_longitude <-122.330676)]
    elif city == 2:
        return df[(df.end_station_latitude > 37.697799)     \
            & (df.end_station_longitude >-122.330676)       \
            & (df.start_station_longitude >-122.330676)     \
            & (df.start_station_longitude >-122.330676)]
    elif city == 3:
        return df[(df.end_station_latitude < 37.697799)     \
            & (df.end_station_latitude < 37.697799)]

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
    df['year'] = df.start_time.dt.year
    df['month'] = df.start_time.dt.month
    df['day'] = df.start_time.dt.day
    df['hour'] = df.start_time.dt.hour

    #any trip with less than 90 seconds and where
    #start station IS the end station,
    #we make the assumption that its malfunctioned
    df['malfunction'] = (df.duration_sec < 90) & (df.start_station_name == df.end_station_name)
    df['age'] = 2019 - df.member_birth_year

    return df

def subset_df(df, year, month, hist=3):
    #current month df
    cdf = df[(df.year == year) & (df.month == month)]
    #next month df
    ndf = df[(df.year == year) & (df.month == (month+1))]

    #create a new dataframe
    #which includes the current month's data
    #as well as data from previous months
    
    rollover = month-hist
    if rollover <0:
        lyear = year-1
        lmonth = 12+(rollover)+1
        dfu= df[(df.year == year)&(df.month <=month)]
        dfl= df[(df.year == lyear)&(df.month >=lmonth)]
        tsdf = dfu.append(dfl)
    else:
        tsdf = df[(df.year == year)&(df.month <=month) & (df.month>=month-hist)]

    #create a new column called days and give it an arbitrary number
    #we will adjust it later
    tsdf['days'] = 1


    #sort the months by ascending order
    months = np.sort(tsdf.month.unique())


    #create a multiplier based on months
    for idx, mon in enumerate(months):
        mult = idx+1
        #scale the days with the multiplier
        tsdf['days'][tsdf.month == month] = tsdf.day * mult
    return tsdf, cdf, ndf




def get_dummies(df):
    
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
        