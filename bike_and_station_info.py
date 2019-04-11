import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from get_features import *



def unique_stations(df):
    '''
    Given a dataframe, identify the unique start/end stations
    
    INPUT: DataFrame
    OUTPUT: 1 array of unique start station ids
  
    '''
    #start station name/id and the number of trips for that station
    lst_start_station_name = df.start_station_name.value_counts()
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


def knn_proposed_stn(sub, df1, df2, proposed_stn, num_neighbors = 3):
    '''
    INPUT 3 dataframes, subsetted df, current month's df, and next month's df
            as well as the number of desired neighbors
    OUTPUT dict of knn, dict for id and coordinate combinations for each current and next month


    '''

    #all coordinates for each trip
    coordinates = np.array(df1[['start_station_longitude', 'start_station_latitude']])
    
    #unique coords in df1
    unique_coords = np.unique(coordinates, axis = 0)
    
    #current month
    cm = df1.month.unique()[0]
    
    #get the id and coords for current month
    id_coord_df1 = stn_coords(df1)
    id_coord_df2 = stn_coords(df2)
    
    knn_dict = {}
    
    #iterate through each proposed station
    for p in proposed_stn:
        
        #use euclidean_distance to find distances between each point
        dist = euclidean_distance(id_coord_df2.get(p), unique_coords)
        
        #sort the distances from closest to furthest
        potential_neighbors = unique_coords[np.argsort(dist)]
        
        neighbors = np.array([0,0])
        
        #in the list of potential neighbors, use neighbors with more than 30 days of trips
        for pot in potential_neighbors:
            
            #get the station id
            sid = sub.start_station_id[(sub.start_station_longitude==pot[0])\
                                       &(sub.start_station_latitude==pot[1])].unique()[0]
#             if len(cdf.days[cdf.start_station_id==sid].unique())>10:
            if len(sub.days[(sub.start_station_id==sid) & (sub.month == cm)].unique())>10:
                neighbors = np.vstack((neighbors, pot))
        neighbors = neighbors[1:num_neighbors+1]
        
        #list for storing neighboring station ids
        neighbor_ids = []
        for i in range(num_neighbors):
            knn_id = sub.start_station_id[(sub.start_station_longitude == neighbors[i][0]) &(sub.start_station_latitude == neighbors[i][1])].iloc[0]
            neighbor_ids.append(int(knn_id))
        knn_dict[int(p)] = neighbor_ids
    return knn_dict, id_coord_df1, id_coord_df2


def trips_per_day(df, station_id):
    '''
    number of trips per day, given a station id
    INPUT: df and station id
    OUTPUT: sorted array of trip counts by date
    '''
    tseries = df['days'][df.end_station_id == station_id].value_counts().reset_index()
    tseries = np.array(tseries)
    tseries = tseries[np.argsort(tseries[:,0])]
    return tseries


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
    '''
    Returns the bike id and number of times it "malfunctioned" in a given period
    '''
    return df.bike_id[df.malfunction == True].value_counts()

def same_station(df):
    return df.bike_id[df.start_station_name == df.end_station_name].value_counts()
