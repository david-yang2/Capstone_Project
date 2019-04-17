Forecast next month’s average daily trip count for new Ford GoBike Stations

Context:

San Francisco has an extensive public transportation network; however, traffic congestion in city has gotten increasingly worse. 
Bike share programs such as Ford GoBike offers users an alternative to traditional public transportation. 
Data for this project was collected from the company’s website. 
The goal of the project is to help the company with their expansion plans by using trip data to forecast how much traffic a new dock station is expected to receive.

Process
1)	Identify 3 closest stations to the newly proposed station by applying k-nearest neighbors.
a.	Used Euclidean Distance as the distance metric
2)	Fit each neighboring stations with their own time series to forecast daily trip counts for the next 30 days 
3)	Forecast next month’s average daily trip count for new Ford GoBike stations as the overall average forecasts for neighboring stations


Results: 

•	Baseline RMSE of 19.91

•	Forecast RMSE of 16.29

•	I believe the large RMSE in both 
baseline and forecast was due to
exogenous factors which I have not
incorporated yet.
(i.e. Emergence of new dock less
scooter / bike programs such as
Scoot, Jump, and Skip)

Next Steps:

•	Apply my model to the East Bay and San Jose

•	Incorporate data from new scooter/bike programs into my existing model

