import matplotlib.pyplot as plt

#initial code

def plt_stn(neighbors):
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
            ts = trips_per_day(sub, s_id)
            plt.plot(ts[:,0],ts[:,1])
            num+=1
            graph+=1

def plt_neighbors(cdf, neighbors, coords):
    for k, v in neighbors.items(): 
        print("The origin station is {}.".format(k))
        n1 = int(neighbors.get(k)[0])
        n2 = int(neighbors.get(k)[1])
        n3 = int(neighbors.get(k)[2])
        print("The 3 closest stations are: {}, {}, {}".format(n1,n2,n3))
        plt.figure(figsize = (10,10))
        
        
        #lims for SF
        ll,rr = -122.50, -122.36
        bb = 37.73
#         #lims for OAK
#         ll,rr = -122.33, -122.19
#         bb = 37.76
        plt.xlim(ll,rr)
        plt.ylim(bb, bb+(rr-ll))
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Map of Ford GoBike stations in San Francisco")
        plt.scatter(cdf.end_station_longitude, cdf.end_station_latitude, s=1, c=('r'))
        plt.scatter(coords.get(k)[0], coords.get(k)[1], s=50, marker='x', c=('b'))

        for st in neighbors.get(k):
            plt.scatter(coords.get(st)[0],coords.get(st)[1], s= 20, c='g')
        plt.show()

def plt_original_results(results):
    plt.figure(figsize=(20,15))
    plt.plot(results.index[:], np.array(results.forecasted_results), c = 'r',label = "forecast")
    plt.plot(results.index[:], np.array(results.actual_results), c = 'b', label = "atual" )
    plt.plot(results.index[:], np.array(results.baseline_results), c = 'g', label = "baseline" )
    plt.legend(loc='upper left')

def plt_scaled_results(results):
    plt.figure(figsize=(20,15))
    plt.ylabel("Trip Count", size = 20)
    plt.plot(results.index[:], np.array(results.scaled_forecast), c = 'r',label = "forecast")
    plt.plot(results.index[:], np.array(results.actual_results), c = 'b', label = "actual" )
    plt.plot(results.index[:], np.array(results.baseline_results), c = 'g', label = "baseline" )
    plt.legend(loc='upper left',prop={'size': 20})