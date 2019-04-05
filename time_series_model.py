import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def days_for_ts(df, cdf, hist=3):
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

def ARIMA_pred(arr, p=3, d=1, q=0):
    tseries = pd.Series(arr[:,1])
    
    tscv = TimeSeriesSplit(n_splits=3)
    fig = plt.figure(figsize=(10,10))
    index = 1
    
    actual = []
    
    for train_index, test_index in tscv.split(tseries):
        train = tseries[train_index]
        test = tseries[test_index]

        train_matrix = train.as_matrix()
        a_model = ARIMA(train_matrix, order=(p, d, q)).fit()
    
        train_sze = len(train)
        predictions = trip_model.predict(train_sze, train_sze+len(test)-1, typ='levels')
        
        #calculate mean squared error
        mse = mean_squared_error(predictions, np.array(test))
        
        #combine to plot on same graph
        combined = np.append(trip_matrix, predictions)
        combined = pd.Series(combined)
        plt.subplot(3,1,index)
        plt.xlabel('Days')
        plt.ylabel('Trip Counts')
        plt.title('This has a mean squared error of {}'.format(mse))
        plt.plot(combined.index[:train_sze], combined[:train_sze])
        plt.plot(combined.index[train_sze:], combined[train_sze:])
#         plt.plot(test)
        print (np.array(test))
#         print(len(train_index))
#         print (test_index)
        print (predictions)
        
        index +=1
        


