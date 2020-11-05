import numpy
from functools import wraps
from time import time
from tools import Tools
import heapq
import numpy as np
from random import random
import math
import copy


def get_time(roundnumber=100):
    def get_time_deco(func):
        @wraps(func)
        def decorator(*args, **kwargs):
            start = time()
            for i in range(roundnumber):
                res = func(*args, **kwargs)
            end = time()
            print('%s运行时间为:%.6f' % (func.__name__, (end - start)))
            return res
        return decorator
    return get_time_deco


class KD_node:
    def __init__(self, point=None, id=None, split=None, LL=None, RR=None):
        """
        point:数据点
        split:划分域
        LL, RR:节点的左儿子跟右儿子
        """
        self.point = point
        self.id = id
        self.split = split
        self.left = LL
        self.right = RR
        self.isvisited = 0

    def __str__(self):
        return 'point:{} split:{} left:{} right:{}'.format(self.point,
                                                           self.split,
                                                           self.left.point,
                                                           self.right.point)

    def initialize(self):
        self.isvisited = 0
        if self.left:
            self.left.initialize()
        if self.right:
            self.right.initialize()


def createKDTree(data_list):
    """
    root:当前树的根节点
    data_list:数据点的集合(无序)
    return:构造的KDTree的树根
    """
    LEN = len(data_list)
    if LEN == 0:
        return
        # 数据点的维度
    dimension = len(data_list[0])-1
    # 方差
    max_var = 0
    # 最后选择的划分域
    split = 0
    for i in range(dimension):
        ll = []
        for t in data_list:
            ll.append(t[i])
        var = computeVariance(ll)
        if var > max_var:
            max_var = var
            split = i
            # 根据划分域的数据对数据点进行排序
    data_list.sort(key=lambda x: x[split])
    # 选择下标为len / 2的点作为分割点
    point = data_list[LEN // 2]
    root = KD_node(point=point[:-1],id=point[-1], split=split)
    root.left = createKDTree(data_list[0:(LEN // 2)])
    root.right = createKDTree(data_list[(LEN // 2 + 1):LEN])
    return root


def computeVariance(arrayList):
    """
    arrayList:存放的数据点
    return:返回数据点的方差
    """
    for ele in arrayList:
        ele = float(ele)
    LEN = len(arrayList)
    array = numpy.array(arrayList)
    sum1 = array.sum()
    array2 = array * array
    sum2 = array2.sum()
    mean = sum1 / LEN
    # D[X] = E[x^2] - (E[x])^2
    variance = sum2 / LEN - mean ** 2
    return variance


#@get_time(roundnumber=100)
def findNN(root, query, k=1):
    """
    root:KDTree的树根
    query:查询点
    return:返回距离data最近的点NN，同时返回最短距离min_dist
    """
    # 初始化为root的节点
    nodeList = []
    candidates = {}
    temp_root = copy.copy(root)
    ##二分查找建立路径
    nodeList, candidates = kdTreeForwardSearch(temp_root, query, nodeList, candidates, k)

    ##回溯查找
    while nodeList:
        # 使用list模拟栈，后进先出
        back_point = nodeList.pop()
        ss = back_point.split
        ##判断是否需要进入父亲节点的子空间进行搜索
        if abs(query[ss] - back_point.point[ss]) <= max(candidates.keys()):
            if query[ss] <= back_point.point[ss]:
                temp_root = back_point.right
                if temp_root:
                    nodeList, candidates = kdTreeForwardSearch(temp_root, query,
                                                               nodeList, candidates, k)
            else:
                temp_root = back_point.left
                if temp_root:
                    nodeList, candidates = kdTreeForwardSearch(temp_root, query,
                                                               nodeList, candidates, k)

    # res = []
    root.initialize()
    # for i in candidates:
    #     res.append(candidates[i].point)
    return candidates


def kdTreeForwardSearch(root, query, nodeList, candidates, k=1):
    temp_root = root
    while temp_root:
        if temp_root.isvisited == 1:
            break
        nodeList.append(temp_root)
        dd = computeDist(query, temp_root.point)
        if len(candidates) < k:
            candidates[dd] = temp_root
        elif max(candidates.keys()) > dd:
            del (candidates[max(candidates.keys())])
            candidates[dd] = temp_root
        temp_root.isvisited = 1

        # 当前节点的划分域
        ss = temp_root.split
        if query[ss] <= temp_root.point[ss]:
            temp_root = temp_root.left
        else:
            temp_root = temp_root.right
    return nodeList, candidates


def computeDist(pt1, pt2):
    """
    计算两个数据点的距离
    return:pt1和pt2之间的距离
    """
    sum = 0.0
    for i in range(len(pt1)):
        sum = sum + (pt1[i] - pt2[i]) * (pt1[i] - pt2[i])
    return math.sqrt(sum)


def preorder(root):
    """
    KDTree的前序遍历
    """
    print(root.point)
    if root.left:
        preorder(root.left)
    if root.right:
        preorder(root.right)


def KNN(list, query):
    min_dist = np.inf
    NN = list[0]
    for pt in list:
        dist = computeDist(query, pt)
        if dist < min_dist:
            NN = pt
            min_dist = dist
    return NN, min_dist


@get_time(roundnumber = 100)
def iterator(x, data_list):
    dis = []
    candidates = []
    for i in range(len(data_list)-1):
        dis.append(Tools.getdistance(x, data_list[i]))
    small_value_inds = list(map(dis.index, heapq.nsmallest(5, dis)))
    for i in small_value_inds:
        candidates.append(data_list[i])
    return candidates


if __name__ == '__main__':
    import pandas as pd

    station = pd.read_csv('needed_points.csv', encoding='utf-8')
    data_list = []
    for row in range(len(station)):
        data_list.append(list(station.iloc[row][['x', 'y','SiteId']]))
    root = createKDTree(data_list)
    res = findNN(root,(100000, 100000), k=5)
    res2 = iterator((100000, 100000), data_list)
    for k in res.keys():
        print(res[k].point)
    #print(sorted(res2))
    #print(station[station['SiteId']==res[-1][-1]])