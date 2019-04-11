

San Francisco has an extensive public transportation network; however, traffic congestion in the Bay Area has gotten increasingly worse. That's where bike share programs such as Ford GoBikes come in. Ford GoBike has been working with local agencies to set up stations around their respective areas. So far, you can see Ford GoBike stations in San Francisco, East Bay, and San Jose. Residents of San Francisco first noticed  Ford GoBike stations in the summer of 2017. Although many of the populated areas are saturated with stations, there are still many areas in the cities for which the company can still expand to. 

My goal is to forecast next month's average daily trip count for a new Ford GoBike stations. 

To perform this model, I used Ford GoBike's system data.

To accomplish this, I used a 2 step process. First, if given a new station location, I would identify the 3 closest stations to the newly proposed station by applying k-nearest neighbors. The distance metric I used for knn was the Euclidean distance. 

For example, let the Blue X mark represent our new station. We will then compare the latitude and longitude between the Blue X with the latitude and longitude of all existing stations. Once distances have been calculated, the list of distances is then sorted in ascending order. The first 3 coordinates on our sorted list, will represent the neighboring stations. Those neighboring stations are represented by the green dots on this map. 

After neighboring stations have been identified, the daily trip counts for first neighboring station was fitted with a time series model, so we can generate daily trip counts for the next 30 days. This step is repeated for the two reamining neighbors. The forecast for next month's average daily trip count for a new Ford GoBike station is the average of all forecasts of the neighboring stations. 


The baseline result for this model was the average daily trip count for neighboring stations in the past quarter. Unfortunately, my results performed slightly worse than the baseline. The baseline RMSE was 26.7 and the RMSE for my forcast was 27.3.  This indicates that on average, the baseline result differed from the actual results by 26.7 trips and my forecasted results differed from the average by 27.3 trips. I believe there are a number of factors which caused the large RMSE for both the baseline and forecasted results. First one being, the introduction of new programs from companies like Scoot, Bird, and Jump. 





