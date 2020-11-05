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
    :param lon: 经度
    :param lat: 维度
    :return:
    """
    L = 6381372 * math.pi * 2  # 地球周长
    W = L  # 平面展开，将周长视为X轴
    H = L / 2  # Y轴约等于周长一般
    mill = 2.3  # 米勒投影中的一个常数，范围大约在正负2.3之间
    x = lon * math.pi / 180  # 将经度从度数转换为弧度
    y = lat * math.pi / 180
    # 将纬度从度数转换为弧度
    y = 1.25 * math.log(math.tan(0.25 * math.pi + 0.4 * y))  # 这里是米勒投影的转换

    # 这里将弧度转为实际距离 ，转换结果的单位是公里
    x = (W / 2) + (W / (2 * math.pi)) * x
    y = (H / 2) - (H / (2 * mill)) * y
    return int(round(x)), int(round(y))