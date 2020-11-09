import pandas as pd
from datetime import datetime
import math


def concat_wind_pm_station(table_wind, table_station, table_pm):
    '''

    :param table_wind:
    :param table_station:
    :param table_pm:
    :return:
    '''
    table_wind = pd.read_csv(table_wind, encoding='utf-8')
    table_station = pd.read_csv(table_station, encoding='utf-8')
    table_pm = pd.read_csv(table_pm, encoding='utf-8')
    table_wind = get_wind(table_wind)
    table_station = get_station(table_station)
    table_pm = get_pm(table_pm)
    table_station[['x', 'y']] = table_station.apply(lambda x: millerToXY(x['Lon'], x['Lat']),
                                                    axis=1, result_type="expand")
    citylist = set(table_station['CityName'])
    citydict = dict(zip(citylist, range(len(citylist))))
    table_station['cityind'] = table_station['CityName'].map(citydict)
    table_station['cityind'] = table_station['cityind'].astype('int')

    table = pd.merge(table_station, table_wind,
                     left_on=['Aws_Id'], right_on=['Station_Id_C'],how='right')
    table.drop(['Station_Id_C'],axis=1,inplace=True)

    table['Datetime'] = pd.to_datetime(table['Datetime'], format='%Y-%m-%d %H:%M:%S')
    table['time_point'] = table['Datetime'].apply(lambda x: datetime.strftime(x, '%Y-%m-%d'))
    table['time_point_m'] = table['Datetime'].apply(lambda x: datetime.strftime(x, '%Y-%m'))
    table['time_point_y'] = table['Datetime'].apply(lambda x: datetime.strftime(x, '%Y'))

    table = table[table['CityName'].notnull()]
    table = pd.merge(table, table_pm, left_on=['CityName', 'time_point'],
                     right_on=['area', 'time_point'],how='left')
    table = table[table['time_point_m'] == '2018-05']
    table.to_csv('output1.csv',encoding='utf-8')
    return table


def get_wind(table_wind):
    return table_wind.loc[:, ['Station_Id_C', 'Datetime', 'WIN_S_Max', 'WIN_D_S_Max']]


def get_station(table_station):
    return table_station.loc[:, ['CityName', 'Aws_Id', 'Lon', 'Lat']]


def get_pm(table_pm):
    table_pm['time_point'] = pd.to_datetime(table_pm['time_point'], format='%Y-%m-%d %H:%M:%S')
    table_pm['time_point'] = table_pm['time_point'].apply(lambda x: datetime.strftime(x, '%Y-%m-%d'))
    return table_pm.loc[:, ['area', 'pm2_5', 'time_point']]


def millerToXY(lon, lat):
    """
    :param lon: Longitude
    :param lat: Latitude
    :return:
    """
    L = 6381372 * math.pi * 2  # earth circumference
    W = L  # The plane is expanded and the perimeter is treated as the X axis
    H = L / 2  # The Y axis is about half the circumference
    mill = 2.3  # A constant in Miller's projection, ranging from plus or minus 2.3
    x = lon * math.pi / 180  # Converts longitude from degrees to radians
    y = lat * math.pi / 180
    # Converts latitude from degrees to radians
    y = 1.25 * math.log(math.tan(0.25 * math.pi + 0.4 * y))  # Here is the transformation of Miller projection

    # Here, the radian is converted into the actual distance, and the unit of conversion is km
    x = (W / 2) + (W / (2 * math.pi)) * x
    y = (H / 2) - (H / (2 * mill)) * y
    return int(round(x)), int(round(y))