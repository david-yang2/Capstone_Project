import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

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

