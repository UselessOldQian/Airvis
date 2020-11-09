# -*- coding:utf-8 -*-
from collections import defaultdict
import itertools
import numpy as np
import matplotlib.pyplot as plt
import json


class HourlyGraph(object):
    """Graph class."""

    def __init__(self, gid):
        """Initialize Graph instance.

        Args:
            gid: id of this graph.
            is_undirected: whether this graph is directed or not.
            eid_auto_increment: whether to increment edge ids automatically.
        """
        self.gid = gid
        self.vertices = dict()
        self.edge = defaultdict(list)
        self.vertices = defaultdict(list)
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

    def add_edge(self, frm, to, ts, te):
        '''

        :param frm:
        :param to:
        :param ts: start time
        :param te: end time
        :return:
        '''
        if (frm in self.vertices[ts] and
                to in self.vertices[te] and
                (frm, to) in self.edge[(ts,te)]):
            return self
        self.edge[(ts,te)].append((frm, to))
        return self

    def is_contain(self, name):
        for t in self.vertices.keys():
            for v in self.vertices[t]:
                if v == name:
                    return True
        return False

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
        for estart_end in other.edge.keys():
            for edge_list in other.edge[estart_end]:
                self.add_edge(edge_list[0], edge_list[1], estart_end[0], estart_end[1])
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
        nx.draw(gnx, arrows=True, with_labels=True)
        for i in self.vid_dict.keys():
            print(str(i)+':'+self.vid_dict[i])
        #plt.show()
        plt.savefig('subgraph_result/'+ str(self.gid) +'.png')


