import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def days_for_ts(df, cdf), hist=3):
    cm = cdf.month.unique()[0]
    cy = cdf.year.unique()[0]

    #create a new dataframe
    #which includes the current month's data
    #as well as data from the 2 previous months
    tsdf = df[(df.year == cy)&(df.month <=cm) & (df.month>=cm-hist)]


    #create a new column called days and give it an arbitrary number
    #we will adjust it later
    tsdf['days'] = 1


    #sort the months by ascending order
    months = np.sort(tsdf.month.unique())


    #create a multiplier based on months
    # 3x for current month
    # 2x for previous month
    # 1x for months ago
    for idx, mon in enumerate(months):
        mult = idx+1
        #scale the days with the multiplier
        #range from 1-93
        tsdf['days'][tsdf.month == mon] = tsdf.day * mult
    return tsdf
def days_count(df, station_id):
    tsplt = df['days'][df.end_station_id == station_id].value_counts().reset_index()
    tsplt = np.array(tsplt)
    tsplt = tsplt[np.argsort(tsplt[:,0])]
    return tsplt

def plt_stn():
    row_sze = len(neighbors)
    col_sze = len(next(iter(neighbors.values())))
    rnge = row_sze*col_sze
    fig = plt.figure(figsize=(10,10))
    graph = 1
    for k, v in neighbors.items():
        num = 1
        for s_id in v:
            plt.subplot(rnge,1,graph)
            plt.subplots_adjust(top=10, bottom =5)
            plt.xlabel("days")
            plt.ylabel("trips per day")
            plt.title("This is station {}, and it is neighbor #{} for station {}".format(s_id, num, k))
            ts = days_count(qtr, s_id)
            plt.plot(ts[:,0],ts[:,1])
            num+=1
            graph+=1
