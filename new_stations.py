import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from get_features import *
from merge import*


def month_sep(df, year, month):
    cdf = df[(df.year == year) & (df.month == month)]
    ndf = df[(df.year == year) & (df.month == (month-1))]
    return cdf, ndf

def unique_stations(df):
    '''
    Given a dataframe, identify the unique start/end stations
    
    INPUT: DataFrame
    OUTPUT: 1 array of unique start station ids
  
    '''
    #names of each start station and the number of trips 
    lst_start_station_name = df.start_station_name.value_counts()
    #ids of each start station and the number of trips 
    lst_start_station_id = df.start_station_id.value_counts()
    num_unique_stations = lst_start_station_id.unique().size
    unique_start_sations = df.start_station_id.unique()
    unique_end_stations = df.end_station_id.unique()
    return unique_start_sations

def new_stn_coords(df1, df2):
    '''
    INPUT: 2 lists. 1 list of new station ids
                    1 list of old station ids
    '''

    new_stn = unique_stations(df2)
    curr_stn = unique_stations(df1)
    ps = set(new_stn) - set(curr_stn)
    lst_new = list(ps)
    return lst_new


def stn_coords(df):
    '''
    returns a dictionary with all station_id and coordinate combinations
    '''
    #getting the coordinates from the dataset
    coordinates = np.array(df[['start_station_longitude', 'start_station_latitude']])
    unique_coords = np.unique(coordinates, axis = 0)
    #create a dictionary with
    #station id as key
    #coordinates for the station id as values
    id_coord = {}
    for u in unique_coords:
        k = df.start_station_id[(df.start_station_longitude == u[0]) &(df.start_station_latitude == u[1])].iloc[0]
        id_coord[int(k)] = u
    return id_coord


def euclidean_distance(x, y):
    return np.sqrt(((x-y)**2).sum(axis=1))


def knn_proposed_stn(df1, df2, proposed_stn, num_neighbors = 3):
    

    coordinates = np.array(df1[['start_station_longitude', 'start_station_latitude']])
    unique_coords = np.unique(coordinates, axis = 0)
    #get the id and coords for current month
    id_coord_df1 = stn_coords(df1)
    id_coord_df2 = stn_coords(df2)
    knn_dict = {}
    for p in proposed_stn:
        neighbors = unique_coords[np.argsort(euclidean_distance(id_coord_df2.get(p), unique_coords))][1:num_neighbors+1]
#         k = df.start_station_id[(df.start_station_longitude == id_coord_df2.get(p)[0]) &(df.start_station_latitude == id_coord_df2.get(p)[1])].iloc[0]
        v = []
        for i in range(num_neighbors):
            knn_id = df1.start_station_id[(df1.start_station_longitude == neighbors[i][0]) &(df1.start_station_latitude == neighbors[i][1])].iloc[0]
            v.append(int(knn_id))
        knn_dict[int(p)] = v
    return knn_dict, id_coord_df1


