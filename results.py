from get_features import *
from time_series_model import *
from bike_and_station_info import *
import math
import matplotlib.pyplot as plt
%matplotlib inline

def gather_data(city = 1):
    data, weather = load_data()
    df = feature_addition(data)
    city = model_city(df, city)
    return city


#months that didn't work
# 10-2017 no new stations
# 11-2017 no new stations
# 12-2017 cannot reshape array of size 0 into shape (0,newaxis)
# 01-2018 no new stations
# 12-2018 cannot reshape array of size 0 into shape (0,newaxis)
dates = np.array([[2017, 9],
      [2018, 2],
      [2018, 3],
      [2018, 4],
      [2018, 5],
      [2018, 6],
      [2018, 7],
      [2018, 8],
      [2018, 9],
      [2018, 10],
      [2018, 11]])

def run(dates):
    city = gather_data(city=1)
    results = np.array([0,0,0,0,0,0])
    for d in dates:
        sub, cdf, ndf = subset_df(city, d[0], d[1])
        ps = new_stn_coords(cdf, ndf)
        neighbors, id_coord1, id_coord2 = knn_proposed_stn(sub, cdf, ndf, ps)
        #go through each proposed station
        avg_pred_dict = {}
        for k, v in neighbors.items():
            pred_temp= []
            #neighboring stations
            for station_id in v:
                #get time series data
                ts = trips_per_day(sub, station_id)
                #find the rmse for each hyper parameter of time series data
                score = best_ARIMA_param(ts)
                #get the hyper parameter that produced the lowest rmse
                b_params = best_params(score)

                #average prediction for next month
                pred = forecast_nxt_30d(ts, b_params,station_id)

                #baseline for station 
                base = baseline(neighbors,sub)

                #add next month's prediction to list
                pred_temp.append(pred)

            #avg the predictions for knn
            #scale it downwards
            avg_temp = (np.array(pred_temp).mean())

            #add to dict
            avg_pred_dict[k] = avg_temp

        avg = validate(sub,neighbors,avg_pred_dict,ndf)
        time = np.array([d[0], d[1]])
        time = np.array([time,]*len(avg))
        data = np.hstack((time, avg))
        results = np.vstack((results, data))
    return results[1:]

results_arr = run(dates)

#create a new dataframe using the results matrix
results = pd.DataFrame(results_arr, columns = [['year', 'month','station_id','baseline_results', 'forecasted_results', 'actual_results']])


def plt_original_results(results):
    plt.figure(figsize=(20,15))
    plt.title("original results", size ="20")
    plt.ylabel("Trips Count", size = "20")
    plt.plot(results.index[:], np.array(results.forecasted_results), c = 'r',label = "forecast")
    plt.plot(results.index[:], np.array(results.actual_results), c = 'b', label = "atual" )
    plt.plot(results.index[:], np.array(results.baseline_results), c = 'g', label = "baseline" )
    plt.legend(loc='upper left',prop={'size': 20})

def scaler(results):
    
    #abs difference between baseline results and actual results
    baseline_diff =np.array(results['baseline_results']) - np.array(results['actual_results'])
    results['baseline_diff'] = abs(baseline_diff)
    
    #abs difference between forecasted results and actual results
    forecasted_diff =np.array(results['forecasted_results']) - np.array(results['actual_results'])
    results['forecasted_diff'] = abs(forecasted_diff)
    
    #store scaler and score
    scores_lst = []
    
    for s in range(1,100):
        
        scaler = s/100
        
        #scale forecast
        results['scaled_forecast'] = np.array(results['forecasted_results'])*(scaler)

        #abs difference between scaled forecast and actual
        scaled_forecasted_diff =np.array(results['scaled_forecast']) - np.array(results['actual_results'])
        results['scaled_diff'] = abs(scaled_forecasted_diff)

        #true if scaled forecast is less than baseline 
        results['scaled_score'] = np.array(results['scaled_diff']) < np.array(results['baseline_diff'])

        #get the overall accuracy
        accuracy = results.scaled_score.sum()/results.shape[0]
        
        #add results to list
        scores_lst.append([scaler, accuracy])
        
    scores_lst = np.array(scores_lst)

    sorted_lst = scores_lst[np.argsort(scores_lst[:,1])][::-1][0]

    best_scaler = sorted_lst[0]
    #plot scalers and scores
    plt.figure(figsize=(15,10))
    plt.ylabel('accuracy score')
    plt.xlabel('scalers')
    plt.title('Accuracy score for each scaler')
    plt.plot(scores_lst[:,0], scores_lst[:,1])
    plt.plot(sorted_lst[0], sorted_lst[1], marker="x", markersize = "20")
    
    
    #sort score from highest to lowest to get optimal scaler 
    
    
    #create scaled forecast
    results['scaled_forecast'] = np.array(results['forecasted_results'])*(best_scaler)
    scaled_forecasted_diff =np.array(results['scaled_forecast']) - np.array(results['actual_results'])
    results['scaled_diff'] = abs(scaled_forecasted_diff)
    results['scaled_score'] = np.array(results['scaled_diff']) < np.array(results['baseline_diff'])
    return scores_lst, results

scores, scaled_results = scaler(results)

def plt_scaled_results(results):
    plt.figure(figsize=(20,15))
    plt.title("Scaled Results", size = 20)
    plt.ylabel("Trip Count", size = 20)
    plt.plot(results.index[:], np.array(results.scaled_forecast), c = 'r',label = "forecast")
    plt.plot(results.index[:], np.array(results.actual_results), c = 'b', label = "actual" )
    plt.plot(results.index[:], np.array(results.baseline_results), c = 'g', label = "baseline" )
    plt.legend(loc='upper left',prop={'size': 20})

def get_score(scaled_results):
    return scaled_results.scaled_score.sum()/scaled_results.shape[0]