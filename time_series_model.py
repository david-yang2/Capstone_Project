
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, pacf

from scipy import signal
from scipy import stats

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from math import sqrt


from bike_and_station_info import *
from time_series_model import *


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
            ts = trips_per_day(df, station_id)
            plt.plot(ts[:,0],ts[:,1])
            num+=1
            graph+=1

def ARIMA_pred(arr, p=1, d=1, q=1):
    
    tseries = pd.Series(arr[:,1])
#     tseries = stationary_convert(tseries)
    tscv = TimeSeriesSplit(n_splits=3)
    fig = plt.figure(figsize=(10,10))
    index = 1
    
    actual = []
    mean_rmse= np.array([])
    for train_index, test_index in tscv.split(tseries):
        train = tseries[train_index]
        test = tseries[test_index]

        train_vals = train.values
#         trip_model = ARIMA(train_vals, order=(p, d, q)).fit()
        try:
            trip_model = ARIMA(train_vals, order=(p, d, q)).fit()
        except:
            print("there was an exception with parameters {},{},{}".format(p,d,q))
        train_sze = len(train)
        predictions = trip_model.predict(train_sze, train_sze+len(test)-1, typ='levels')
        
        #calculate mean squared error
        mse = mean_squared_error(predictions, np.array(test))
        rmse = mse.mean()
        mean_rmse = np.append(mean_rmse, rmse)
        
        #combine train with predictions
        combined = np.append(train_vals, predictions)
        combined = pd.Series(combined)



    plt.xlabel('Days')
    plt.ylabel('Trip Counts')
    plt.title('This has a root mean squared error of {}'.format(rmse))
    #plot train
    plt.plot(combined.index[:train_sze], combined[:train_sze], 'g', label = "train")
    #plot prediction
    plt.plot(combined.index[train_sze:], combined[train_sze:], 'r', label =  "prediction")
    #plot actual
    plt.plot(tseries.index[train_sze:], tseries[train_sze:], 'b', label = "actual")
    plt.legend(loc='upper left')


    avg_rmse = mean_rmse.mean()
    return avg_rmse
        


def best_ARIMA_param(arr):
    max_param = 3
    score = np.array([0,0,0,0])
    exceptions = 0
    d=1
    q=0
    for p in range(1,max_param+1):
        # for d in range(1,max_param+1):
        #     for q in range(1,max_param+1):
        try:
            rmse = ARIMA_pred(arr, p, d, q)
            score_data = np.array([p,d,q,rmse])
            score = np.vstack((score, score_data))
        except: 
            # print("The hyper parameters {}, {}, and {} did not work.".format(p, d, q))
            exceptions +=1
    print ("There were a total of {} exceptions.".format(exceptions/max_param**3))
    return score[1:]


def best_params(score):
    best_params = score[score[:,3] ==np.sort(score[:,3]).reshape(-1,1)[0]]
    return best_params


def forecast_nxt_30d(ts, b_params, station_id, months=3):
    tseries = pd.Series(ts[:,1])
    p = int(b_params[0][0])
    d = int(b_params[0][1])
    q = int(b_params[0][2])
    
    current_vals = tseries.values
    model = ARIMA(current_vals, order=(p, d, q)).fit()

    train_sze = len(current_vals)
#     predictions = trip_model.predict(train_sze, train_sze+30, typ='levels')
    predictions = model.predict((30*months), (30*months)+30, typ='levels')

    #combine to plot on same graph
    combined = np.append(current_vals, predictions)
    combined = pd.Series(combined)
    plt.xlabel('Days')
    plt.ylabel('Trip Counts')
    plt.title("This is the forecasted trips per day for neighboring station {}.".format(station_id))
    # plt.plot(combined.index[:train_sze], combined[:train_sze])
    # plt.plot(combined.index[train_sze:], combined[train_sze:])
    plt.legend(loc='upper left')
    
    plt.plot(combined.index[:train_sze], combined[:train_sze], 'g', label = "train")
    plt.plot(combined.index[train_sze:], combined[train_sze:], 'r', label =  "prediction")
    plt.plot(tseries.index[train_sze:], tseries[train_sze:], 'b', label = "actual")
    next_month_avg_pred = predictions.mean()
    return next_month_avg_pred


def station_trends(ts, b_params):
    # #stores the proposed location's stations id and its neighbors' overall average trip count per day
    # trend = {}
    # #get the orgin station and its neighbor
    # for k, v in neighbors.items():
        
    #     avg_temp = []
    #     #access each neighbor for station k
    #     for s in v:
    try:
        print ("This is station {}, and the neighbor is {}".format(k,s))

        # #time series data for station s
        # ts = trips_per_day(sub,s)

        # #scores for each ARIMA hyper parameter combination
        # score = best_ARIMA_param(ts)

        # #best hyper parameter for the model
        # #based on smallest MSE
        # b_params = best_params(score)

        #average prediction of trips per day for the next month
        avg_pred = forecast_nxt_30d(ts, b_params)
        # avg_temp.append(avg_pred)
        return avg_pred
    except:
        print ("Combination did not work")
        # neigh_avg = np.array(avg_temp)
        # trend[k] = neigh_avg
    

def baseline(neighbors, sub):
    '''
    Use the average trips per day for the baseline for my model
    '''
    avg_count = {}
    for k, v in neighbors.items():
        avg_lst = []
        for s_id in v:
            avg = np.array(sub.days[sub.end_station_id == s_id].value_counts()).mean()
            avg_lst.append(avg)
        avg = np.array(avg_lst).mean()
        avg_count[k] = avg
    return avg_count

def validate(sub, neighbors, trend, ndf):
    
    agg = np.array([0,0,0])
    base = baseline(neighbors,sub)
    #validate the stations in the trend dictionary
    for k, v in trend.items():
        
        #mean of forecasted values
        neighbor_mean = v.mean()
        
        #using the following month's data to calculate actual trips per day
        count = ndf[ndf.end_station_id == k]["day"].value_counts()
        actual_trips_per_day = np.round(np.array(count).mean(), decimals = 3)

        store = np.array([base.get(k), neighbor_mean, actual_trips_per_day])
        agg = np.vstack((agg, store))
        print ("Validating for station {}".format(k))
        print ("The baseline estimate using the mean for neighboring station is {}".format(base.get(k)))
        print ("The average predicted trip count per day is {}.".format(neighbor_mean)) 
        print ("The actual trips per day for the following month is {}.".format(actual_trips_per_day))
        print ("----------------------------------------------------------------")
    return agg[1:].mean(axis = 0)


