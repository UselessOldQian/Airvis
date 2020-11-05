import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
import math
from numpy.random import random
import heapq
from tools import Tools
from hourly_graph import HourlyGraph
from collections import defaultdict
import warnings
import time
import KDTree

warnings.filterwarnings("ignore")


class BuildGraphHourly:
    def __init__(self, start="2019120100", end="2019120410",
                 is_build_graph=True, save_path='edge.npy',
                 distance_range=30000,
                 receive_distance_range=30000,
                 random_point_num=10,
                 time_threshold=12):
        self.start = start
        self.end = end
        # if cal_station is None:
        #     self.table_station = pd.read_csv('table2.csv', encoding='utf-8')
        #     self.table_station = self.table_station.loc[:, ['CityName', 'Aws_Id', 'Lon', 'Lat']]
        #     self.table_station[['x', 'y']] = self.table_station.apply(
        #         lambda x: Tools.millerToXY(x['Lon'], x['Lat']), axis=1, result_type="expand")
        #     self.table_station.to_csv('station.csv',sep=',',index=False,encoding='utf-8')
        # else:
        self.table_station = pd.read_csv('station.csv',encoding='utf-8')
        self.pm_data = pd.read_csv('2019aqihour.csv', encoding='gbk')
        self.pm_data['time_point'] = pd.to_datetime(self.pm_data['time_point'],
                                                    format='%Y-%m-%d %H:%M:%S')
        self.get_pm_data_in_timespan(start, end)
        self.pm_data = self.pm_data.loc[:, ['area', 'pm2_5', 'latitude', 'longitude', 'time_point']]
        self.pm_data = self.pm_data[self.pm_data['pm2_5'].notnull()]

        # self.needed_points = pd.read_csv('wind_points.csv', encoding='utf-8')
        # self.needed_points[['x', 'y']] = self.needed_points.apply(
        #     lambda x: Tools.millerToXY(x['Lon'], x['Lat']), axis=1, result_type="expand")
        # self.needed_points = self.needed_points.loc[:, ['SiteId', 'Lon', 'Lat', 'x', 'y']]
        # self.needed_points.to_csv('needed_points.csv',sep=',',index=False,encoding='utf-8')
        self.needed_points = pd.read_csv('needed_points.csv',encoding='utf-8')
        self.pm_data = pd.merge(self.table_station, self.pm_data,
                                left_on=['CityName'], right_on=['area'], how='right')
        self.hour_list = Tools.get_hour_list(start, end)
        self.hour_dict = {i: self.hour_list[i] for i in range(len(self.hour_list))}
        self.pointset = sorted(set(self.pm_data['area'].values))
        self.city2ind_dict = {city: i for i, city in enumerate(self.pointset)}
        self.ind2city_dict = {i: city for i, city in enumerate(self.pointset)}
        self.win_tables = None
        self.isvisited_metrix = {}
        self.edge_Pij_dict = {}
        self.graphdict = {}
        self.adjecent_dict_reverse = {}
        self.adjecent_dict = {}
        self.name_xy_dict = {cityname: (
            self.table_station[self.table_station['CityName'] == cityname]['Lon'].values[0],
            self.table_station[self.table_station['CityName'] == cityname]['Lat'].values[0]
        ) for cityname in self.pointset}
        self.max_hour = -1
        if is_build_graph:
            self.build_graph(save_path=save_path,
                             distance_range=distance_range,
                             receive_distance_range=receive_distance_range,
                             random_point_num=random_point_num,
                             time_threshold=time_threshold)

    def get_pm_data_in_timespan(self, start, end):
        start = datetime.strptime(str(start), "%Y%m%d%H")
        end = datetime.strptime(str(end), "%Y%m%d%H")
        self.pm_data = self.pm_data[(self.pm_data['time_point'] >= start) &
                                    (self.pm_data['time_point'] <= end)]

    @staticmethod
    def search_windata(time_start, time_end, threshold=0):
        hour_list = Tools.get_hour_list(start=time_start, end=time_end, threshold=threshold)
        all_data = []
        for d in hour_list:
            file_name = 'database/' + d.strftime("%Y%m%d%H") + '.txt'
            data_of_single_day = pd.read_table(file_name, header=0,
                                               usecols=['Station_Id_d', 'Lat', 'Lon',
                                                        'WIN_D_Avg_2mi', 'WIN_S_Avg_2mi'],
                                               encoding='gbk', sep='\s+')
            all_data.append(data_of_single_day[data_of_single_day['WIN_D_Avg_2mi'] <= 360])
        return all_data

    @staticmethod
    def get_needed_points(win_table, win_needed_points):
        win_table = pd.merge(win_needed_points, win_table,
                             left_on=['SiteId'], right_on=['Station_Id_d'], how='left')
        win_table.drop(['Station_Id_d', 'Lat_y', 'Lon_y'], axis=1, inplace=True)
        win_table = win_table[win_table.notnull()]
        return win_table

    def get_edge(self, point_name, start_time_id,
                 windatas, root,
                 random_point_num=10,
                 time_threshold=24,
                 distance_range=30000,
                 receive_distance_range=30000):
        '''

        :param point_name:
        :param start_time_id:
        :param random_point_num:
        :param time_threshold:
        :param distance_range:
        :return:
        '''
        point = (self.table_station[self.table_station['CityName'] == point_name]['x'].values[0],
                 self.table_station[self.table_station['CityName'] == point_name]['y'].values[0])
        # plt.scatter(table['x'],table['y'],c='b')
        # plt.scatter(point[0],point[1],c='gold')
        if len(windatas) < start_time_id + time_threshold:
            return None, None

        t = random(size=random_point_num) * 2 * np.pi - np.pi
        x = np.cos(t)
        y = np.sin(t)
        length = random(size=random_point_num) * distance_range
        random_point_x = point[0] - length * x
        random_point_y = point[1] - length * y
        random_point_list = [(x, y) for x, y in zip(random_point_x, random_point_y)]
        PassingPoint = []
        PassingTime = []
        for point in random_point_list:
            # color=['g', 'r', 'c', 'm', 'y', 'k', 'w','gold','goldenrod']
            # c = np.random.choice(color)
            for i in range(time_threshold):
                win_table = windatas[start_time_id + i]
                sub_win_table = BuildGraphHourly.get_needed_points(win_table, self.needed_points)

                point = BuildGraphHourly.movement(point, sub_win_table, root)
                dis = []
                for ind in range(len(self.table_station)):
                    dis.append(Tools.getdistance(point,
                                                 (self.table_station.iloc[ind]['x'],
                                                  self.table_station.iloc[ind]['y'])))
                dis = np.array(dis)
                dis_smaller_than_thre = (dis < receive_distance_range)
                if sum(dis_smaller_than_thre) > 0:
                    # plt.scatter(point[0],point[1],c=c)
                    passing_point = self.table_station.loc[dis_smaller_than_thre].loc[:, ['CityName']].values.flatten()

                    if point_name in passing_point and len(passing_point) == 1:
                        continue
                    elif point_name in passing_point:
                        passing_point = list(passing_point)
                        passing_point.remove(point_name)
                    PassingPoint.extend(passing_point)
                    node_time = i+1
                    PassingTime.extend([node_time] * len(passing_point))
                    break
        if len(PassingPoint) == 0:
            return None, None
        PassingTime = np.array(PassingTime)
        proportionDict = {}
        time_dict = {}
        for ind, key in enumerate(PassingPoint):
            proportionDict[key] = proportionDict.get(key, 0) + 1
            time_dict.setdefault(key, []).append(ind)
        # print('time_dict:',time_dict)
        for key in proportionDict.keys():
            proportionDict[key] = proportionDict[key] / random_point_num
            mean_time = np.mean(PassingTime[np.array(time_dict[key])])
            time_dict[key] = int(round(mean_time))
        return proportionDict, time_dict

    @staticmethod
    def movement(point, win_table, root):
        '''
        计算单个点pm2.5的移动
        :param point:
        :param table:
        :return:
        '''

        res = KDTree.findNN(root, point, k=5)
        small_val_neg_sum = 0
        movement_x, movement_y = 0, 0
        for k in res.keys():
            index = win_table['SiteId'] == res[k].id
            if sum(index) != 0:
                win_d = win_table[index]['WIN_D_Avg_2mi'].values[0]
                win_s = win_table[index]['WIN_S_Avg_2mi'].values[0]
                direct_x, direct_y = Tools.get_wind_x_y(wind_direct=win_d,wind_speed=win_s)
                movement_x += 1 / k * direct_x
                movement_y += 1 / k * direct_y
                small_val_neg_sum += 1/k

        # dis = []
        # for i in range(len(win_table)):
        #     city = win_table.iloc[i]
        #     city_point = (city['x'], city['y'])
        #     dis.append(Tools.getdistance(point, city_point))
        # small_value_inds = list(map(dis.index, heapq.nsmallest(5, dis)))  # 求最小的五个索引    nsmallest与nlargest相反，求最小
        # small_values = heapq.nsmallest(5, dis)
        # small_val_neg_sum = sum([1 / v for v in small_values])
        # movement_x, movement_y = 0, 0
        # for ind, val in zip(small_value_inds, small_values):
        #     direct_x, direct_y = Tools.get_wind_x_y(win_table.iloc[ind]['WIN_D_Avg_2mi'],
        #                                             win_table.iloc[ind]['WIN_S_Avg_2mi'])
        #     movement_x += 1 / val * direct_x
        #     movement_y += 1 / val * direct_y
        if small_val_neg_sum != 0:
            movement_x = movement_x / small_val_neg_sum
            movement_y = movement_y / small_val_neg_sum
        point = (point[0] + movement_x, point[1] + movement_y)
        return point

    def build_graph(self, random_point_num=10, time_threshold=12,
                    distance_range=30000,
                    receive_distance_range=30000,
                    save_path='test.npy'):
        '''
        根据表构建单日的传播图
        :param random_point_num: 随机生成的点
        :param time_threshold: pm2.5消散时间，单位小时
        :return: self.edge_Pij_dict[time] 字典，索引为日期，值为该日期传播图中的所有边以及其可能性
        '''
        windatas = BuildGraphHourly.search_windata(self.start, self.end, threshold=time_threshold)
        data_list = []
        for row in range(len(self.needed_points)):
            data_list.append(list(self.needed_points.iloc[row][['x', 'y','SiteId']]))
        root = KDTree.createKDTree(data_list)
        # propagation_array = np.zeros([len(self.hour_list), len(self.pointset)])
        for time_id in self.hour_dict.keys():
            subtable = self.pm_data[self.pm_data['time_point'] ==
                                    self.hour_dict[time_id].strftime("%Y-%m-%d %H:%M:%S")]
            pointset = sorted(set(self.pm_data['area'].values))
            num_of_point = len(pointset)
            edge_Pij = []
            start_time = time.time()
            for point_ind, point_name in enumerate(pointset):
                # print('Point_id:',point_id,'Processing....')
                percentage = int((point_ind + 1) / num_of_point * 100)
                if percentage >= 100:
                    end_time = time.time()
                    process = "\r[%3s%%]: |%-100s|time consumption:%s\n" % (percentage,
                                                                            '|' * percentage,
                                                                            round(end_time - start_time))
                else:
                    process = "\r[%3s%%]: |%-100s|" % (percentage, '|' * percentage)
                print(process + 'time: {}'.format(time_id), end='', flush=True)

                C = 0 if len(subtable[subtable['area'] == point_name]) == 0 else \
                subtable[subtable['area'] == point_name].iloc[0]['pm2_5']
                if C < 30:
                    continue

                connected_points, passing_time = self.get_edge(
                    point_name, time_id, windatas, root,
                    random_point_num=random_point_num,
                    time_threshold=time_threshold,
                    distance_range=distance_range,
                receive_distance_range=receive_distance_range)
                if connected_points is None:
                    continue

                for connected_point in connected_points.keys():
                    TC = C * connected_points[connected_point]
                    if TC < 30:
                        continue
                    if passing_time[connected_point] > time_threshold:
                        print('ERROR:{}'.format(passing_time[connected_point]))
                    edge_Pij.append([point_name, connected_point,
                                     connected_points[connected_point],  # P
                                     time_id,  # start_time
                                     time_id + passing_time[connected_point],  # end_time
                                     TC])  # quantity of pm
            self.edge_Pij_dict[time_id] = edge_Pij
            np.save(save_path, self.edge_Pij_dict)
        print("\n" + "处理完成")

    def getGraphList(self, reload=None):
        if reload is not None:
            self.edge_Pij_dict = np.load(reload, allow_pickle=True).item()
        self.get_max_hour()
        self.get_is_visited()
        self.get_adjecent_dict()
        gid = 0
        for tid in range(self.max_hour, 0, -1):
            for cityname in self.adjecent_dict_reverse[tid].keys():  # proplist:['南京市', '马鞍山市', 1.0, 0, 1, 41.0]
                pid = self.city2ind_dict[cityname]
                g_true_id = gid
                pid = int(pid)
                # 初始化graph
                if self.isvisited_metrix[tid][pid] == -1:
                    g = HourlyGraph(g_true_id)
                    g.add_vertex(cityname, tid)
                    self.isvisited_metrix[tid][pid] = g_true_id
                else:
                    # print("original:visited:{},gid_now:{}".format(self.isvisited_metrix[tid][pid],gid))
                    g = self.graphdict[self.isvisited_metrix[tid][pid]]
                    g_true_id = g.gid
                for frm_info_list in self.adjecent_dict_reverse[tid][cityname]:
                    frm_p = self.city2ind_dict[frm_info_list[0]]
                    if self.isvisited_metrix[frm_info_list[2]][frm_p] == -1:
                        g.add_vertex(frm_info_list[0], frm_info_list[2])
                        g.add_edge(frm_info_list[0], cityname, frm_info_list[2], frm_info_list[3])
                        self.isvisited_metrix[frm_info_list[2]][frm_p] = g_true_id
                    else:
                        # print("merge:visited:{} , now:{}".format(self.isvisited_metrix[tid-1][frm_p],gid))
                        # print("MERGE Now gid:{}".format(gid))
                        g.add_edge(frm_info_list[0], cityname, frm_info_list[2], frm_info_list[3])
                        g_in_dict = self.graphdict[self.isvisited_metrix[frm_info_list[2]][frm_p]]
                        for t in g.vertices.keys():
                            for p_name in g.vertices[t]:
                                p = self.city2ind_dict[p_name]
                                self.isvisited_metrix[t][p] = g_in_dict.gid
                        g = g_in_dict.merge(g)
                        g_true_id = g.gid

                if gid == g.gid and g.get_vertex_num() > 1:
                    self.graphdict[gid] = g
                    gid += 1

    def get_max_hour(self):
        self.max_hour = -1
        for _, list in self.edge_Pij_dict.items():
            for v in list:
                if self.max_hour < v[-2]:
                    self.max_hour = v[-2]

    def get_is_visited(self):
        for tid in range(self.max_hour + 1):
            self.isvisited_metrix[tid] = [-1] * len(self.pointset)

    def get_adjecent_dict(self):
        '''
        根据图的边list构建dict，便于搜索
        :return:
        '''
        for i in range(self.max_hour + 1):
            self.adjecent_dict_reverse[i] = defaultdict(list)
        for time in self.edge_Pij_dict.keys():
            edge_Pij = self.edge_Pij_dict[time]
            self.adjecent_dict[time] = defaultdict(list)
            for e in edge_Pij:
                self.adjecent_dict[time][e[0]].append([e[1], e[2], e[3], e[4], e[5]])
                self.adjecent_dict_reverse[e[-2]][e[1]].append([e[0], e[2], e[3], e[4], e[5]])

    def dict_concat(self,dict_list):
        dict1 = np.load(dict_list[0], allow_pickle=True).item()
        dict_con = dict1.copy()
        for i in range(1,len(dict_list)):
            dict2 = np.load(dict_list[i], allow_pickle=True).item()
            maxnum = max(dict_con.keys())+1
            for k in dict2.keys():
                for v in dict2[k]:
                    v[3] += maxnum
                    v[4] += maxnum
                dict_con[k + maxnum] = dict2[k]
        self.edge_Pij_dict = dict_con

    def plot(self, min_vertex_count=0, contain_name = None):
        for key, graph in self.graphdict.items():
            if graph.get_vertex_num() < min_vertex_count:
                continue
            if contain_name is None:
                graph.plot()
            if contain_name is not None and graph.is_contain(contain_name):
                graph.plot()

    def output_graph(self, min_vertex_count=0):
        for key, graph in self.graphdict.items():
            if graph.get_vertex_num() < min_vertex_count:
                continue
            graph.plot()


if __name__ == '__main__':
    bgh = BuildGraphHourly(start="2019120100", end="2019120113",
                           is_build_graph=True,time_threshold=8,
                           distance_range=50000,
                           receive_distance_range=10000,
                           random_point_num=20)
    #bgh.dict_concat(['edge_Pij_dict.npy','Dec0103_06.npy'])
    bgh.getGraphList(reload='edge.npy')  # reload='edge_Pij_dict.npy')
    bgh.plot(min_vertex_count=3, contain_name= '上海市')