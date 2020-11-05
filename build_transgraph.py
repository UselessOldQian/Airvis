# -*- coding:utf-8 -*-
from collections import defaultdict
import itertools
import numpy as np
import matplotlib.pyplot as plt
from pyecharts.charts import *
from pyecharts import options as opts
import json


class Graph(object):
    """Graph class."""

    def __init__(self, gid, timeinds):
        """Initialize Graph instance.

        Args:
            gid: id of this graph.
            is_undirected: whether this graph is directed or not.
            eid_auto_increment: whether to increment edge ids automatically.
        """
        self.gid = gid
        self.vertices = dict()
        self.edge = dict()
        for tid in timeinds:
            self.vertices[tid] = []
            self.edge[tid] = []
        self.counter = itertools.count()
        self.vid_dict = {}
        self.tid_dict = {}
        self.xy_dict = {}

    def get_vidandtid_dict(self,vid_dict,tid_dict,xy_dict):
        self.vid_dict = vid_dict
        self.tid_dict = tid_dict
        self.xy_dict = xy_dict

    def add_vertex(self, vid, ti):
        """Add a vertex to the graph."""
        if vid in self.vertices[ti]:
            return self
        self.vertices[ti].append(vid)
        return self

    def add_edge(self, frm, to, ti):
        """Add an edge to the graph."""
        if (frm in self.vertices[ti - 1] and
                to in self.vertices[ti] and
                (frm, to) in self.edge[ti]):
            return self
        self.edge[ti].append((frm, to))
        return self

    def get_vertex_num(self):
        ret_len = 0
        for var in self.vertices.keys():
            ret_len += len(self.vertices[var])
        return ret_len

    def set_gid(self, gid):
        self.gid = gid

    def merge(self, other):
        for tid in other.vertices.keys():
            for vid in other.vertices[tid]:
                self.add_vertex(vid, tid)
            for e in other.edge[tid]:
                self.add_edge(e[0], e[1], tid)
        return self

    def getGraph(self):
        print('gid:', self.gid)
        print('edge:', self.edge)
        print('vertices:',self.vertices)
        print('-------------------')

    def plot(self):
        """Visualize the graph."""
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
        except Exception as e:
            print('Can not plot graph: {}'.format(e))
            return
        gnx = nx.DiGraph()
        vlbs = {}
        for tid, vid_list in self.vertices.items():
            for vid in vid_list:
                vlbs[vid] = tid
        for tid, vid in self.vertices.items():
            gnx.add_nodes_from(vid, label=tid)
        for tid, edges in self.edge.items():
            gnx.add_edges_from(edges, label=tid)
        plt.figure(self.gid)
        pos = self.xy_dict
        vid_Edict = {0:'taizhou',1:'shaoxing',2:'liuan',3:'lishui',4:'hangzhou',
                     5:'wenzhou',6:'huainan',7:'taizhou',8:'wuhu',9:'ningbo',
                     10:'jinghua',11:'banghu',12:'suqian',13:'xuzhou',14:'jiaxing',
                     15:'tongling',16:'changzhou',17:'haozhou',18:'lianyungang',
                     19:'hefei',20:'shanghai',21:'suzhou',22:'suzhou',23:'huyang',
                     24:'huangshan',25:'zhoushan',26:'nantong',27:'yangzhou',
                     28:'yancheng',29:'huaian',30:'xuancheng',31:'quzhou',32:'chizhou',
                     33:'wuxi',34:'maanshan',35:'nanjing',36:'zhengjiang',37:'huzhou',
                     38:'tuzhou',39:'huaibei',40:'anqing'}
        nx.draw(gnx, pos, arrows=True, with_labels=True, labels = vid_Edict)
        for i in self.vid_dict.keys():
            print(str(i)+':'+self.vid_dict[i])
        plt.show()

    def getGraphNodes(self):
        '''
        表中的城市数据转为字典，画图用
        :return:
        '''
        graph_nodes = []
        for t in self.vertices.keys():
            for vertex in self.vertices[t]:
                node = {'id': str(vertex), 'name': vertex,
                        #'x': city['Lon'], 'y': city['Lat'],
                        'label': {"normal": {"show": True}},
                        'category': 0}
                graph_nodes.append(node)
        return graph_nodes

    def getGraphEdge(self):
        '''
        将edge从list转为字典
        :param edges:
        :return:
        '''
        graph_edges = []
        for t in self.edge.keys():
            for e in self.edge[t]:
                edge = {'id': 0, 'source': e[0], 'target': e[1]}
                graph_edges.append(edge)
        return graph_edges

    def plot_in_pyechart(self):
        # nodes = self.getGraphNodes()
        # links = self.getGraphEdge()
        # categories = [{"name": "City"}]
        # j = {'categories': [{"name": "City"}], 'nodes': nodes, 'links': links}
        #
        # class NpEncoder(json.JSONEncoder):
        #     def default(self, obj):
        #         if isinstance(obj, np.integer):
        #             return int(obj)
        #         elif isinstance(obj, np.floating):
        #             return float(obj)
        #         elif isinstance(obj, np.ndarray):
        #             return obj.tolist()
        #         else:
        #             return super(NpEncoder, self).default(obj)
        #
        # jsObj = json.dumps(j, cls=NpEncoder)
        # save_path = 'graphs/gid_' + str(self.gid) + '.json'
        # fileObject = open(save_path, 'w')
        # fileObject.write(jsObj)
        # fileObject.close()
        with open('graphs/gid_2.json', "r", encoding="utf-8") as f:
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
        page.render('graphs/gid' + str(self.gid) + '.html')


class TransportGraph(object):
    def __init__(self, table, edges_dict):
        self.adjecent_dict = {}
        self.adjecent_dict_reverse = {}
        self.isvisited_metrix = {}
        self.graphdict = {}
        self.edges_dict = {}
        assert type(edges_dict) in [dict, str]
        if type(edges_dict) is str:
            self.edges_dict = np.load(edges_dict, allow_pickle=True).item()
        else:
            self.edges_dict = edges_dict
        self.table = table
        self.get_adjecent_dict()

        self.timelist = sorted(set(self.table['time_point']))
        self.pointset = sorted(set(self.table['cityind'].values))
        self.timenum = len(self.timelist)
        self.timeinds = range(self.timenum)
        self.getGraphList()
        self.get_is_visited()
        self.tid_dict = {tid:time for tid,time in zip(self.timeinds,self.timelist)}
        #
        self.vid_dict = {vid:self.table[self.table['cityind']==vid]['CityName'].values[0] for vid in self.pointset}
        self.xy_dict = {vid:(self.table[self.table['cityind']==vid]['Lon'].values[0],
                             self.table[self.table['cityind']==vid]['Lat'].values[0]) for vid in self.pointset}

    def get_is_visited(self):
        for tid in self.timeinds:
            self.isvisited_metrix[tid] = [-1] * len(self.pointset)

    def dfs(self, point_id, graphind, time_ind):
        def _dfs(frm,ti):
            time = self.timelist[ti]
            #self.isvisited_metrix[time] =

    def getGraphList(self):
        self.get_is_visited()
        gid = 0
        for tid in self.timeinds[:0:-1]:
            for pid in self.pointset:
                g_true_id = gid
                pid = int(pid)
                time = self.timelist[tid]
                #初始化graph
                if self.isvisited_metrix[tid][pid] == -1:
                    g = Graph(g_true_id, timeinds=self.timeinds)
                    g.add_vertex(pid,tid)
                    self.isvisited_metrix[tid][pid] = g_true_id
                else:
                    #print("original:visited:{},gid_now:{}".format(self.isvisited_metrix[tid][pid],gid))
                    g = self.graphdict[self.isvisited_metrix[tid][pid]]
                    g_true_id = g.gid
                for frm_p in self.adjecent_dict_reverse[time][pid]:
                    frm_p = int(frm_p)
                    if self.isvisited_metrix[tid-1][frm_p] == -1:
                        g.add_vertex(frm_p, tid-1)
                        g.add_edge(frm_p, pid, tid)
                        self.isvisited_metrix[tid-1][frm_p] = g_true_id
                    else:
                        #print("merge:visited:{} , now:{}".format(self.isvisited_metrix[tid-1][frm_p],gid))
                        #print("MERGE Now gid:{}".format(gid))
                        g.add_edge(frm_p, pid, tid)
                        g_in_dict = self.graphdict[self.isvisited_metrix[tid-1][frm_p]]
                        for t in g.vertices.keys():
                            for p in g.vertices[t]:
                                self.isvisited_metrix[t][p] = g_in_dict.gid
                        g = g_in_dict.merge(g)
                        g_true_id = g.gid

                if gid == g.gid and g.get_vertex_num() > 1:
                    self.graphdict[gid] = g
                    gid += 1

    def get_adjecent_dict(self):
        '''
        根据图的边list构建dict，便于搜索
        :return:
        '''
        for time in self.edges_dict.keys():
            edge_Pij = self.edges_dict[time]
            self.adjecent_dict[time] = defaultdict(list)
            self.adjecent_dict_reverse[time] = defaultdict(list)
            for e in edge_Pij:
                self.adjecent_dict[time][e[0]].append(e[1])
                self.adjecent_dict_reverse[time][e[1]].append(e[0])

    def display(self):
        """Display the graph as text."""
        display_str = ''
        i = 0
        for key, g in self.graphdict.items():
            #print('t # {}\n'.format(i))
            display_str += 't # {}\n'.format(i)
            for tid,vertex_list in g.vertices.items():
                for vid in vertex_list:
                    display_str += 'v {} {}\n'.format(vid, tid)
                for e in g.edge[tid]:
                    #print('e {} {} {}'.format(e[0],e[1],1))
                    display_str += 'e {} {} {}\n'.format(e[0], e[1], tid)
            i += 1
        #print('t # -1')
        display_str += 't # -1'

        strs = display_str.split('\n')
        for s in strs:
            open('mygraph.data', mode='a',encoding='utf-8').writelines([s,'\n'])
        print(display_str)
        return display_str

    def plot(self,min_vertex_count=0):
        for key, graph in self.graphdict.items():
            if graph.get_vertex_num() <= min_vertex_count:
                continue
            graph.get_vidandtid_dict(self.vid_dict, self.tid_dict,self.xy_dict)
            graph.plot()

    def plot_in_pyechart(self,min_vertex_count=0):
        for key, graph in self.graphdict.items():
            if graph.get_vertex_num() <= min_vertex_count:
                continue
            graph.get_vidandtid_dict(self.vid_dict,self.tid_dict)
            graph.plot_in_pyechart()


if __name__ == '__main__':
    from build_graph import BuildGraph
    bg = BuildGraph(('2018-05-01', '2018-05-30'), reload=False)
    tg = TransportGraph(bg.table, 'edge.npy')
    # tg.display()
    tg.plot(min_vertex_count=4)
    # for k in tg.graphdict.keys():
    #     tg.graphdict[k].getGraph()
