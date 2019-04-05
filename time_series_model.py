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

def ARIMA_pred(arr, p=3, d=1, q=1):
    tseries = pd.Series(arr[:,1])
#     tseries = stationary_convert(tseries)
    tscv = TimeSeriesSplit(n_splits=3)
    fig = plt.figure(figsize=(10,10))
    index = 1
    
    actual = []
    mean_mse= np.array([])
    for train_index, test_index in tscv.split(tseries):
        train = tseries[train_index]
        test = tseries[test_index]

        train_vals = train.values
        trip_model = ARIMA(train_vals, order=(p, d, q)).fit()
    
        train_sze = len(train)
        predictions = trip_model.predict(train_sze, train_sze+len(test)-1, typ='levels')
        
        #calculate mean squared error
        mse = mean_squared_error(predictions, np.array(test))
        mean_mse = np.append(mean_mse, mse)
        
        #combine to plot on same graph
        combined = np.append(train_vals, predictions)
        combined = pd.Series(combined)
        plt.subplot(3,1,index)
        plt.xlabel('Days')
        plt.ylabel('Trip Counts')
        plt.title('This has a mean squared error of {}'.format(mse))
        plt.plot(combined.index[:train_sze], combined[:train_sze], label="train")
        plt.plot(combined.index[train_sze:], combined[train_sze:], label='predicted')
#         plt.plot(test)


        print (np.array(test))
#         print(len(train_index))
#         print (test_index)
        print (predictions)
        
        index +=1
    avg_mse = mean_mse.mean()
    return avg_mse
        


def best_ARIMA_param(arr):
    max_param = 3
    q = 0
    score = np.array([0,0,0,0])
    for p in range(1,3+1):
        for d in range(1,3-1):
            mse = ARIMA_pred(arr, p, d, q)
            score_data = np.array([p,d,q,mse])
            score = np.vstack((score, score_data))
    print("This is the array of scores")
    return score[1:]


def best_params(score):
    best_params = score[score[:,3] ==np.sort(score[:,3]).reshape(-1,1)[0]]
    return best_params


def forcast_nxt_30d(ts, b_params):
    tseries = pd.Series(ts[:,1])
    p = int(b_params[0][0])
    d = int(b_params[0][1])
    q = int(b_params[0][2])
    
    current_vals = tseries.values
    trip_model = ARIMA(current_vals, order=(p, d, q)).fit()

    train_sze = len(current_vals)
#     predictions = trip_model.predict(train_sze, train_sze+30, typ='levels')
    predictions = trip_model.predict(90, 120, typ='levels')

    #combine to plot on same graph
    combined = np.append(current_vals, predictions)
    combined = pd.Series(combined)
    plt.xlabel('Days')
    plt.ylabel('Trip Counts')
#     plt.title('This has a mean squared error of {}'.format(mse))
    plt.plot(combined.index[:train_sze], combined[:train_sze], label="train")
    plt.plot(combined.index[train_sze:], combined[train_sze:], label='predicted')
    
    next_month_avg_pred = np.round(predictions.mean(), decimals=2)
    return ("For the next month, this station is expected to receive {} trips per day".format(next_month_avg_pred))


def ready_set_go(qtr,cdf,neighbors):
    #stores the proposed location's stations id and its neighbors' overall average trip count per day
    trend = {}
    #get the orgin station and its neighbor
    for k, v in neighbors.items():
        
        avg_temp = []
        #access each neighbor for station k
        for s in v:
            
            print ("This is station {}, and the neighbor is {}".format(k,s))
            
            #time series data for station s
            ts = days_count(qtr,s)
            
            #scores for each ARIMA hyper parameter combination
            score = best_ARIMA_param(ts)
            
            #best hyper parameter for the model
            #based on smallest MSE
            b_params = best_params(score)
            
            #average prediction of trips per day for the next month
            avg_pred = forecast_nxt_30d(ts, b_params)
            avg_temp.append(s)
        neigh_avg = np.array(avg_temp).mean()
        trend[k] = neigh_avg
            
    return trend