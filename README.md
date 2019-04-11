Forecasting Average Daily Trips for New Location

Context:
San Francisco has an extensive public transportation network; however, traffic congestion in city
has gotten increasingly worse. Bike share programs such as FordGo Bike offer users to ...
Residents of San Francisco first noticed Ford GoBike stations in the summer of 2017. Although
many of the populated areas are saturated with stations, there are still many areas in the cities for
which the company can still expand to.

Process
1) Identify 3 closest stations to the newly proposed station by applying k-nearest neighbors.
  a. Used Euclidean Distance as the distance metric
2) Fit each neighboring stations with their own time series to forecast daily trip counts for the next 30 days
3) Forecast next month’s average daily trip count for new Ford GoBike stations as the overall average forecasts for neighboring stations

Results:

• Baseline RMSE of 19.91

• Forecast RMSE of 16.29

• I believe the large RMSE in both baseline and forecast was due to exogenous factors which I have not incorporated yet

Exogenous Factors

•	Emergence of new dock less scooter / bike programs such as Scoot, Jump, and Skip

Next Steps:

•	Apply my model to the East Bay and San Jose

•	Incorporate data from new scooter/bike programs into my existing model



