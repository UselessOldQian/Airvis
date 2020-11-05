import numpy as np
import pandas as pd
from tools import Tools
from collections import defaultdict
from datetime import datetime
import copy
from pandas.core.frame import DataFrame
import warnings
import queue
from pyecharts import GeoLines, Style
warnings.filterwarnings("ignore")

class Statistics:
    def __init__(self,edge_Pij_dict,
                 start_edge_Pij_dict,
                 end_edge_Pij_dict,
                 start,
                 end,
                 station='station.csv',
                 pollution_table='pmtable/2019_city_aqi.csv',
                 days=None,
                 cls=None):
        hour_lst = []
        if days is not None:
            days = pd.read_csv(days, encoding='utf-8')
            days.iloc[:,0] = pd.to_datetime(days.iloc[:,0], format='%Y-%m-%d')
            if cls is not None:
                days = days[days.iloc[:,1]==cls]
            day_lst = []
            for _, row in days.iterrows():
                day_lst.append(datetime.strftime(row[0],'%Y%m%d00'))
            if day_lst != []:
                start = day_lst[0][:4]+"010100"
            for item in day_lst:
                hour = len(Tools.get_hour_list(start=start, end=item, threshold=0))
                add_lst=[]
                for i in range(0,24):
                    add_lst.append(hour+i)
                hour_lst.extend(add_lst)

        temp_dic = np.load(edge_Pij_dict, allow_pickle=True).item()
        self.edge_Pij_dict = {}
        if hour_lst != []:
            for h in hour_lst:
                self.edge_Pij_dict[h] = temp_dic[h]
        else:
            self.edge_Pij_dict = temp_dic
        self.station = pd.read_csv(station, encoding='utf-8')
        self.pollution_table = pd.read_csv(pollution_table, encoding='utf-8')
        self.pollution_table['time_point'] = pd.to_datetime(self.pollution_table['time_point'],
                                                            format='%Y-%m-%d %H:%M:%S')
        self.hour_list = Tools.get_hour_list(start=start_edge_Pij_dict,
                                             end=end_edge_Pij_dict, threshold=50)
        self.hour_dict = {v.strftime('%Y%m%d%H'):i for i,v in enumerate(self.hour_list)}
        self.time_min = self.hour_dict[start]
        self.time_max = self.hour_dict[end]
        dict_list_value = defaultdict(list)
        dict_list_prob = defaultdict(list)
        dict_list_all = defaultdict(list)
        for k,v in self.edge_Pij_dict.items():
            if self.time_min <= k <= self.time_max:
                for edge in v:
                    dict_list_value[edge[0]+' '+edge[1]].append(edge[-1])
                    dict_list_prob[edge[0]+' '+edge[1]].append(edge[2])
                    dict_list_all[edge[0]+' '+edge[1]].append(edge)
        self.dict_trans_value = defaultdict(dict)
        self.dict_trans_prob = defaultdict(dict)
        self.dict_trans_all = defaultdict(dict)
        for k,v in dict_list_value.items():
            frm,to = k.split(' ')
            self.dict_trans_value[frm][to] = v
            self.dict_trans_prob[frm][to] = dict_list_prob[k]
            self.dict_trans_all[frm][to] = dict_list_all[k]

        self.dict_trans_vol = defaultdict(dict)
        self.dict_trans_count = defaultdict(dict)
        self.dict_trans_reverse = defaultdict(dict)
        self.dict_trans_count_reverse = defaultdict(dict)
        self.dict_trans_isvisit = defaultdict(dict)
        self.dict_node_isvisited = {}
        for k_frm, d in self.dict_trans_value.items():
            for k_to, value in d.items():
                self.dict_trans_vol[k_frm][k_to] = round(sum(value))
                self.dict_trans_reverse[k_to][k_frm] = round(sum(value))
                self.dict_trans_count[k_frm][k_to] = len(value)
                self.dict_trans_count_reverse[k_to][k_frm] = len(value)
                self.dict_trans_isvisit[k_to][k_frm] = 0
                self.dict_node_isvisited[k_to] = 0
                self.dict_node_isvisited[k_frm] = 0

    def init_dict_trans_isvisit(self):
        self.dict_trans_isvisit = defaultdict(dict)
        for k_frm, d in self.dict_trans_value.items():
            for k_to, value in d.items():
                self.dict_trans_isvisit[k_to][k_frm] = 0

    def init_dict_node_isvisited(self):
        for k in self.dict_node_isvisited.keys():
            self.dict_node_isvisited[k] = 0

    def get_start_end_pm_columns(self,output_name):
        edge_list = []
        for time_id, edges in self.edge_Pij_dict.items():
            if self.time_min <= time_id <= self.time_max:
                edge_list.extend(edges)

        df = pd.DataFrame(edge_list, columns=['Source',
                                              'Target',
                                              'Probability',
                                              'Start_time',
                                              'End_time',
                                              'Trans_Volume'])
        df['End_time'] = df['End_time'].apply(lambda x: self.hour_list[x])#.strftime('%Y-%m-%d %H:%M:%S'))
        df['Start_time'] = df['Start_time'].apply(lambda x: self.hour_list[x])#.strftime('%Y-%m-%d %H:%M:%S'))
        df = pd.merge(df, self.pollution_table.loc[:, ['area', 'pm2_5', 'time_point']],
                      how='left', left_on=['Source', 'Start_time'], right_on=['area', 'time_point'])
        df.drop(columns=['time_point', 'area'], inplace=True)
        df.rename(columns={'pm2_5': 'start_pm'}, inplace=True)
        df = pd.merge(df, self.pollution_table.loc[:, ['area', 'pm2_5', 'time_point']],
                      how='left', left_on=['Target', 'End_time'], right_on=['area', 'time_point'])
        df.drop(columns=['time_point', 'area'], inplace=True)
        df.rename(columns={'pm2_5': 'end_pm'}, inplace=True)
        df.to_csv(output_name+'.csv')
        return df

    def search_frm(self, to_name,
                   res=set(), minimum=1000):
        if self.dict_trans_reverse[to_name]:
            for frm, value in self.dict_trans_reverse[to_name].items():
                if self.dict_trans_isvisit[to_name][frm] == 0 and \
                        self.dict_trans_reverse[to_name][frm] >= minimum:
                    res.add((frm,to_name,value))
                    self.dict_trans_isvisit[to_name][frm] = 1
                    res = self.search_frm(frm,res=res,minimum=minimum)
        return res

    def getupstream(self, name, outputname,minimum=0):
        self.upstream = self.search_frm(name, minimum=minimum)
        self.prop_df = DataFrame(list(self.upstream), columns=['Source', 'Target', 'Volume'])
        bins = [np.percentile(self.prop_df['Volume'], i*100/3) for i in range(0, 4)]
        if len(set(bins))<3:
            self.prop_df['class'] = '中度'
        else:
            self.prop_df['class'] = pd.cut(self.prop_df['Volume'], bins=[0]+bins[1:], precision=2,
                                      labels=['轻度','中度','重度'])
        self.plot_upstream(outputname)
        self.prop_df.to_csv(outputname+'传播总量.csv')
        return self.prop_df

    def plot_upstream(self,outputname):
        self.init_dict_trans_isvisit()
        style = Style(
            title_color="#fff",
            title_pos="center",
            width=1510,
            height=840,
            background_color="#404a59" # #404a59"
        )

        style_geo = style.add(
            maptype="china",
            is_label_show=False,
            line_curve=0.1,
            line_opacity=0.6,
            legend_text_color="#eee",
            legend_pos="right",
            geo_effect_symbol="arrow",
            symbol_size=5,
            geo_effect_symbolsize=5,
            label_color=['#a6c84c', '#46bee9','red'],
            label_pos="right",
            label_formatter="{b}",
            label_text_color="#eee",
        )

        geolines = GeoLines(outputname, **style.init_style)
        geolines.add(name = "轻度",data = self.prop_df[self.prop_df['class']=='轻度'][['Source','Target']].values.tolist(),
                     is_legend_show=False,**style_geo)
        geolines.add(name = "中度",data = self.prop_df[self.prop_df['class']=='中度'][['Source','Target']].values.tolist(),
                     is_legend_show=True,**style_geo)
        geolines.add(name = "重度",data = self.prop_df[self.prop_df['class']=='重度'][['Source','Target']].values.tolist(),
                     is_legend_show=True,**style_geo)
        geolines.render(outputname+'.html')

    def k_nearest_neibor(self,name,max_depth=3):
        res = defaultdict(list)
        res = self.dfs(name, 0, max_depth,res)
        self.init_dict_trans_isvisit()
        return res

    def dfs(self,to_name,depth,max_depth,res):
        if self.dict_trans_reverse[to_name]:
            for frm in self.dict_trans_reverse[to_name].keys():
                if self.dict_trans_isvisit[to_name][frm] == 0 and \
                        depth < max_depth:
                    res[depth].append([frm,to_name])
                    self.dict_trans_isvisit[to_name][frm] = 1
                    res = self.dfs(frm,depth+1,max_depth,res)
        return res

    def dfs_node(self,to_name,depth,max_depth,res):
        self.dict_node_isvisited[to_name] = 1
        res[depth].append(to_name)
        if self.dict_trans_reverse[to_name]:
            for frm in self.dict_trans_reverse[to_name].keys():
                if self.dict_node_isvisited[frm] == 0:
                    res = self.dfs_node(frm, depth + 1, max_depth, res)
        return res

    def bfs(self,to_name,max_depth,minimum=300):
        q = queue.Queue()
        self.dict_node_isvisited[to_name] = 1
        res = defaultdict(list)
        q.put([to_name,0])
        while not q.empty():
            to_list = q.get()
            if to_list[1] >= max_depth:
                continue
            if self.dict_trans_reverse[to_list[0]]:
                for frm in self.dict_trans_reverse[to_list[0]].keys():
                    if self.dict_node_isvisited[frm] == 0 and self.dict_trans_vol[frm][to_list[0]]>minimum:
                        self.dict_node_isvisited[frm] = 1
                        res[to_list[1]+1].append((frm, to_list[0]))
                        q.put([frm,to_list[1]+1])
        self.init_dict_node_isvisited()
        return res

    def plot_bfs(self, outputname, source_city='上海市', k=5, minimum=300):
        import json
        dict_nn = self.bfs(source_city, k, minimum=minimum)
        jsonstr = json.dumps(dict_nn) 
        filename = open(outputname+'.json', 'w')  # dict转josn
        filename.write(jsonstr)
        style = Style(
            title_color="#fff",
            title_pos="center",
            width=1510,
            height=840,
            background_color="#404a59"  # #404a59"
        )

        style_geo = style.add(
            maptype="china",
            is_label_show=False,
            line_curve=0.1,
            line_opacity=0.6,
            legend_text_color="#eee",
            legend_pos="right",
            geo_effect_symbol="arrow",
            symbol_size=5,
            geo_effect_symbolsize=5,
            label_color=['#a6c84c', '#46bee9', 'red', '#ffa022','green'],#'#a6c84c'
            label_pos="right",
            label_formatter="{b}",
            label_text_color="#eee",
            # geo_cities_coords=airports_geo
        )

        geolines = GeoLines(outputname, **style.init_style)
        for i in range(1,k+1):
            geolines.add(name=str(i)+"阶", data=dict_nn[i],
                         is_legend_show=True, **style_geo)
        geolines.render(outputname+'近邻城市.html')
        return dict_nn


if __name__ == '__main__':
    sta = Statistics('trans_result/2019.npy',"2019010100", "2019123123",
                     "2019010100", "2019123123")#,days='test.csv',cls=1)
    dataframe_list = sta.get_start_end_pm_columns('result2/2019')
    res = sta.getupstream('上海市','result2/2019')
    sta.plot_bfs('result2/2019',source_city='上海市',minimum=0)
