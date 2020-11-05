import pandas as pd
from concat_table import concat_wind_pm_station
import math
import numpy as np
import heapq
from datetime import datetime
from numpy.random import random
from pyecharts.charts import *
from pyecharts import options as opts
import json


class BuildGraph:

    def __init__(self, time_point: list, table_wind=None, table_station=None, table_pm=None, reload=False):
        #self.edge_TCij_dict = {}
        #self.edge_Tij_dict = {}
        self.edge_Pij_dict = {}
        if reload:
            assert (table_wind is not None) and (table_station is not None) and (table_pm is not None)
            self.table = concat_wind_pm_station(table_wind, table_station, table_pm)
        else:
            self.table = pd.read_csv('output1.csv', encoding='utf-8')

        self.table = self.table[(self.table['time_point'] >= time_point[0])
                                & (self.table['time_point'] <= time_point[1])]
        self.table = self.table[self.table['pm2_5'].notnull()]
        self.timelist = sorted(set(self.table['time_point']))
        self.pointset = sorted(set(self.table['cityind'].values))
        #self.timenum = len(self.timelist)
        #self.timeinds = range(self.timenum)


    def __getitem__(self, time_point: list):
        return self.table[(self.table['time_point'] >= time_point[0]) & (self.table['time_point'] <= time_point[1])]

    @staticmethod
    def get_edge(point_id, table, time, random_point_num=10, time_threshold=24):
        '''
        计算单点的单日传播图
        :param point_id:
        :param table:
        :param time:
        :param random_point_num:
        :param time_threshold:
        :return:
        '''
        point = (table[table['cityind'] == point_id]['x'].values[0], table[table['cityind'] == point_id]['y'].values[0])
        # plt.scatter(table['x'],table['y'],c='b')
        # plt.scatter(point[0],point[1],c='gold')
        t = random(size=random_point_num) * 2 * np.pi - np.pi
        x = np.cos(t)
        y = np.sin(t)
        length = random(size=random_point_num) * 30000
        random_point_x = point[0] - length * x
        random_point_y = point[1] - length * y
        random_point_list = [(x, y) for x, y in zip(random_point_x, random_point_y)]
        PassingPoint = []
        PassingTime = []
        for point in random_point_list:
            # color=['g', 'r', 'c', 'm', 'y', 'k', 'w','gold','goldenrod']
            # c = np.random.choice(color)
            for i in range(time_threshold):
                point = BuildGraph.movement(point, table)
                dis = []
                for ind in range(len(table)):
                    dis.append(BuildGraph.getdistance(point, (table.iloc[ind]['x'], table.iloc[ind]['y'])))
                dis = np.array(dis)
                if sum(dis < 30000) > 0:
                    # plt.scatter(point[0],point[1],c=c)
                    passing_point = table.loc[dis < 30000].loc[:, ['cityind']].values.flatten()

                    if point_id in passing_point and len(passing_point) == 1:
                        continue
                    elif point_id in passing_point:
                        passing_point = list(passing_point)
                        passing_point.remove(point_id)
                    PassingPoint.extend(passing_point)
                    node_time = [datetime.strptime(time + ' ' + str(i), '%Y-%m-%d %H')]
                    PassingTime.extend(node_time * len(passing_point))
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
            proportionDict[key] = proportionDict[key] / len(PassingPoint)
            mean_time = BuildGraph.cal_mean_time(PassingTime[np.array(time_dict[key])])
            time_dict[key] = mean_time
        return proportionDict, time_dict

    @staticmethod
    def getdistance(point, city_point):
        '''
        计算距离
        :param point:
        :param city_point:
        :return:
        '''
        return np.sqrt(np.sum((np.array(point) - np.array(city_point)) ** 2))

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
    def movement(point, table):
        '''
        计算单个点pm2.5的移动
        :param point:
        :param table:
        :return:
        '''
        dis = []
        for i in range(len(table)):
            city = table.iloc[i]
            city_point = (city['x'], city['y'])
            dis.append(BuildGraph.getdistance(point, city_point))
        small_value_inds = list(map(dis.index, heapq.nsmallest(5, dis)))  # 求最小的五个索引    nsmallest与nlargest相反，求最小
        small_values = heapq.nsmallest(5, dis)
        if 0 not in small_values:
            small_val_neg_sum = sum([1 / v for v in small_values])
            movement_x, movement_y = 0, 0
            for ind, val in zip(small_value_inds, small_values):
                direct_x, direct_y = BuildGraph.get_wind_x_y(table.iloc[ind]['WIN_D_S_Max'],
                                                             table.iloc[ind]['WIN_S_Max'])
                movement_x += 1 / val * direct_x
                movement_y += 1 / val * direct_y
            movement_x = movement_x / small_val_neg_sum
            movement_y = movement_y / small_val_neg_sum
        else:
            movement_x, movement_y = BuildGraph.get_wind_x_y(table.iloc[small_value_inds[0]]['WIN_D_S_Max'],
                                                             table.iloc[small_value_inds[0]]['WIN_S_Max'])
        point = (point[0] + movement_x, point[1] + movement_y)
        return point

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

    def build_graph(self, random_point_num=10, time_threshold=12):
        '''
        根据表构建单日的传播图
        :param random_point_num: 随机生成的点
        :param time_threshold: pm2.5消散时间，单位小时
        :return: self.edge_Pij_dict[time] 字典，索引为日期，值为该日期传播图中的所有边以及其可能性
        '''
        time_set = sorted(set(self.table['time_point'].values))

        for time in time_set:
            subtable = self.table[self.table['time_point'] == time]
            pointset = sorted(set(self.table['cityind'].values))
            num_of_point = len(pointset)
            edge_Pij = []
            for point_ind, point_id in enumerate(pointset):
                # print('Point_id:',point_id,'Processing....')
                percentage = int((point_ind + 1) / num_of_point * 100)
                if percentage >= 100:
                    process = "\r[%3s%%]: |%-100s|\n" % (percentage, '|' * percentage)
                else:
                    process = "\r[%3s%%]: |%-100s|" % (percentage, '|' * percentage)
                print(process+'time: {}'.format(time), end='', flush=True)

                connected_points, passing_time = BuildGraph.get_edge(point_id, subtable, time,
                                                                     random_point_num=random_point_num,
                                                                     time_threshold=time_threshold)
                if connected_points is None:
                    continue
                C = subtable[subtable['cityind'] == point_id].iloc[0]['pm2_5']
                for connected_point in connected_points.keys():
                    TC = C * connected_points[connected_point]
                    if TC < 30:
                        continue
                    edge_Pij.append([point_id, connected_point,
                                     connected_points[connected_point],
                                     passing_time[connected_point],TC])
            self.edge_Pij_dict[time] = edge_Pij
        print("\n" + "处理完成")

    def save_edge(self):
        np.save('edge.npy', self.edge_Pij_dict)

    def load_edge(self, path):
        self.edge_Pij_dict = np.load(path,allow_pickle=True).item()

    def getGraphNodes(self, time):
        '''
        表中的城市数据转为字典，画图用
        :param time:
        :return:
        '''
        table = self.table[self.table['time_point'] == time]
        pointset = sorted(set(table['cityind'].values))
        min_v = min(table['pm2_5'].values)
        max_v = max(table['pm2_5'].values)
        graph_nodes = []
        for point_id in pointset:
            city = table[table['cityind'] == point_id].iloc[0]

            node = {'id': str(point_id), 'name': city['CityName'],
                    'symbolSize': BuildGraph.getSize(min_v, max_v, city['pm2_5']),
                    'value': city['pm2_5'], 'x': city['Lon'], 'y': city['Lat'],
                    'label': {"normal": {"show": True}},
                    'category': 0}
            graph_nodes.append(node)
        return graph_nodes

    @staticmethod
    def getSize(min_v, max_v, value):
        '''
        画图时站点图形尺寸
        :param min_v: 最大v值
        :param max_v: 最小v值
        :param value: 该站点v值
        :return:
        '''
        return (value - min_v) / (max_v - min_v) * 30 + 1

    def getGraphEdge(self, time):
        '''
        将edge从list转为字典
        :param edges:
        :return:
        '''
        graph_edges = []
        for i,e in enumerate(self.edge_Pij_dict[time]):
            edge = {'id': i, 'source': e[0], 'target': e[1], 'value': round(e[-1], 2)}
            graph_edges.append(edge)
        return graph_edges

    def paint_graph(self):
        for time in self.timelist:
            nodes = self.getGraphNodes(time)
            links = self.getGraphEdge(time)
            categories = [{"name": "City"}]
            j = {'categories': [{"name": "City"}], 'nodes': nodes, 'links': links}

            class NpEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    else:
                        return super(NpEncoder, self).default(obj)

            jsObj = json.dumps(j, cls=NpEncoder)
            save_path = 'jsonFile_' + time + '.json'
            fileObject = open(save_path, 'w')
            fileObject.write(jsObj)
            fileObject.close()
            with open(save_path, "r", encoding="utf-8") as f:
                j = json.load(f)
                nodes = j["nodes"]
                links = j["links"]
                categories = j["categories"]
            c = (
                Graph(init_opts=opts.InitOpts(width="1000px", height="600px"))
                    .add(
                    "",
                    nodes=nodes,
                    links=links,
                    categories=categories,
                    layout=None,  # "circular",
                    is_rotate_label=True,
                    linestyle_opts=opts.LineStyleOpts(color="source", curve=0.3),
                    label_opts=opts.LabelOpts(position="right"),
                    edge_symbol=[None, 'arrow'],
                )
                    .set_global_opts(
                    title_opts=opts.TitleOpts(title="PM2.5 Propagation Graph"),
                    #              legend_opts= opts.LegendOpts(
                    #                  orient="vertical", pos_left="2%", pos_top="20%"
                    #              ),
                )
            )
            page = Page()  # 实例化page类
            page.add(c)  # TODO 向page中添加图表
            page.render('PM2_5_'+time+'.html')


if __name__ == '__main__':
    bg = BuildGraph(('2018-05-01', '2018-05-30'), reload=False)
    bg.build_graph()
    # bg.save_edge()
    bg.load_edge('edge.npy')
    bg.paint_graph()

    # import sys
    #
    # sys.path.append("../gSpan-master")
    # from gspan_mining.config import parser
    # from gspan_mining.main import main
    #
    # args_str = '-s 5 -d False -l 3 -p True -w True mygraph.data'
    # FLAGS, _ = parser.parse_known_args(args=args_str.split())
    # gs = main(FLAGS)
