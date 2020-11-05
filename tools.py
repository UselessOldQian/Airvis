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
        获取日期列表
        :param start: 开始日期 str,‘2019010100’
        :param end: 结束日期
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
        获取日期列表
        :param start: 开始日期 str,‘2019010100’
        :param end: 结束日期
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
        计算距离
        :param point:
        :param city_point:
        :return:
        '''
        return np.sqrt(np.sum((np.array(point) - np.array(city_point[0:2])) ** 2))

    @staticmethod
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

    @staticmethod
    def get_needed_points(table_origin, needed_points):
        table = pd.merge(needed_points, table_origin,
                         left_on=['SiteId'], right_on=['Station_Id_d'], how='left')
        table.drop(['Station_Id_d', 'Lat_y', 'Lon_y'], axis=1, inplace=True)
        table = table[table.notnull()]
        return table

    @staticmethod
    def get_wind_x_y(wind_direct, wind_speed):
        '''
        将风向从角度转为 单位向量*风速
        :param wind_direct:角度
        :param wind_speed:风速
        :return:
        '''
        direct_y = math.cos(wind_direct / 180 * math.pi) * wind_speed * 3600
        direct_x = math.sin(wind_direct / 180 * math.pi) * wind_speed * 3600
        return direct_x, direct_y

    @staticmethod
    def cal_mean_time(time_array):
        '''
        计算平均时间
        :param time_array:
        :return:
        '''
        time_array.sort()
        t = time_array[0] - time_array[0]
        for time in time_array:
            t = t + time - time_array[0]
        return t / len(time_array) + time_array[0]