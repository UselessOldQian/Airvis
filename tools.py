import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
import math


class Tools:

    @staticmethod
    def gen_dates(b_date, days):
        day = timedelta(days=1)
        for i in range(days):
            yield b_date + day * i

    @staticmethod
    def gen_hours(b_hour, days, seconds):
        hours = seconds // 3600
        hour = timedelta(hours=1)
        day = timedelta(days=1)
        for i in range(days):
            for j in range(24):
                yield b_hour + day * i + hour * j
        for i in range(hours):
            yield b_hour + day * days + hour * i

    @staticmethod
    def get_date_list(start=None, end=None):
        """
        get the date list
        :param start: start date str,‘2019010100’
        :param end: end date
        :return:
        """
        if start is None:
            start = datetime.strptime("2019010100", "%Y%m%d%H")
        else:
            start = datetime.strptime(str(start), "%Y%m%d%H")
        if end is None:
            end = datetime.now()
        else:
            end = datetime.strptime(str(end), "%Y%m%d%H")
        data = []
        for d in Tools.gen_dates(start, (end - start).days):
            data.append(d)
        return data

    @staticmethod
    def get_hour_list(start=None, end=None, threshold=0):
        """
        get date list
        :param start: start date str,‘2019010100’
        :param end: end date
        :return:
        """
        if start is None:
            start = datetime.strptime("2019010100", "%Y%m%d%H")
        else:
            start = datetime.strptime(str(start), "%Y%m%d%H")
        if end is None:
            end = datetime.now()
        else:
            end = datetime.strptime(str(end), "%Y%m%d%H")
        data = []
        for d in Tools.gen_hours(start, (end - start).days, (end - start).seconds):
            data.append(d)
        if data == []:
            return []
        base = data[-1]
        hour = timedelta(hours=1)
        for i in range(threshold):
            data.append(base + hour * (i+1))
        return data

    @staticmethod
    def getdistance(point, city_point):
        '''
        Calculate the distance
        :param point:
        :param city_point:
        :return:
        '''
        return np.sqrt(np.sum((np.array(point) - np.array(city_point[0:2])) ** 2))

    @staticmethod
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

    @staticmethod
    def get_needed_points(table_origin, needed_points):
        '''
        get the table without nan value
        :param table_origin:
        :param needed_points:
        :return:
        '''
        table = pd.merge(needed_points, table_origin,
                         left_on=['SiteId'], right_on=['Station_Id_d'], how='left')
        table.drop(['Station_Id_d', 'Lat_y', 'Lon_y'], axis=1, inplace=True)
        table = table[table.notnull()]
        return table

    @staticmethod
    def get_wind_x_y(wind_direct, wind_speed):
        '''
        translate the wind direction into the unit victor * speed
        :param wind_direct:direction of wind
        :param wind_speed:speed of wind
        :return:
        '''
        direct_y = math.cos(wind_direct / 180 * math.pi) * wind_speed * 3600
        direct_x = math.sin(wind_direct / 180 * math.pi) * wind_speed * 3600
        return direct_x, direct_y

    @staticmethod
    def cal_mean_time(time_array):
        '''
        Calculate the mean time
        :param time_array:
        :return:
        '''
        time_array.sort()
        t = time_array[0] - time_array[0]
        for time in time_array:
            t = t + time - time_array[0]
        return t / len(time_array) + time_array[0]