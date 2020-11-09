from Air_statistics import Statistics
from build_graph_hourly import BuildGraphHourly


bgh = BuildGraphHourly(start="2019120100", end="2019120110",
                           is_build_graph=False,time_threshold=8,
                           distance_range=50000,
                           receive_distance_range=10000,
                           random_point_num=20)
#bgh.dict_concat(['edge_Pij_dict.npy','Dec0103_06.npy'])
bgh.getGraphList(reload='trans_result/2019.npy')  # reload='edge_Pij_dict.npy')
bgh.plot(min_vertex_count=10, contain_name= '上海市')
sta = Statistics('trans_result/2019.npy',"2019010100", "2019120110",
                     "2019010100", "2019120110")#,days='test.csv',cls=1)
dataframe_list = sta.get_start_end_pm_columns('result2/2019')
res = sta.getupstream('上海市','result2/2019')
sta.plot_bfs('result2/2019',source_city='上海市',minimum=0)