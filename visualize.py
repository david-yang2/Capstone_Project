import matplotlib.pyplot as plt


def euclidean_distance(x, y):
    return np.sqrt(((x-y)**2).sum(axis=1))



def knn_plot(df, s):
    
    coordinates = np.array(df[['end_station_longitude', 'end_station_latitude']])
    unique_coords = np.unique(coordinates, axis = 0)
    neigbors = unique_coords[np.argsort(euclidean_distance(unique_coords[t], unique_coords))][1:4]
    plt.figure(figsize = (10,10))
    plt.xlim(-122.50, -122.36)
    plt.scatter(sf.end_station_longitude, sf.end_station_latitude, s=1, c=('r'))
    plt.scatter(unique_coords[s][0], unique_coords[s][1], s=20, c=('b'))
    plt.scatter(neigbors[:,0],neigbors[:,1], s= 10, c='g')
    plt.show()