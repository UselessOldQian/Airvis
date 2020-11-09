# Airvis
The propagation mode of air pollutants (such as pm2.5) is modeled and visualized through data mining according to wind speed, wind direction, air pollutant concentration and other indicators. The project aims to help experts in the field identify and locate pollution sources, as well as further decision support.

## Data
- [*wind_database*](https://github.com/UselessOldQian/Airvis/tree/main/wind_database): Part of wind speed data of weather station. The name of each wind table indicate the time(yyyymmddHH.txt)
* *WIN_D_Avg_2mi*: indicate the direction of the wind. It is 0 in due north direction and increases clockwise.
* *WIN_S_Avg_2mi*: Wind Speed(m/s)

- *pmtable*: [2019_city_aqi.csv](https://github.com/UselessOldQian/Airvis/tree/main/pmtable): The pm2.5 Table of 2019.

- [*trans_result*](https://github.com/UselessOldQian/Airvis/tree/main/trans_result): Intermediate data transmitted by PM2.5.

## How to run
Run the main.py, and the result of subgraph mining will be saved in [subgraph_result](https://github.com/UselessOldQian/Airvis/tree/main/subgraph_result), the result of pm2.5 propagation pattern and the upstream citys of the target city will be saved in [result](https://github.com/UselessOldQian/Airvis/tree/main/result2) folder(like the picture below).
![subgraph mining](https://github.com/UselessOldQian/Airvis/blob/main/Pic/subgraph_mining_results.png)
![pm2.5 propagation pattern](https://github.com/UselessOldQian/Airvis/blob/main/Pic/All.png)
![the upstream citys of the target city](https://github.com/UselessOldQian/Airvis/blob/main/Pic/Near.png)
