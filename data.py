import json, os
import math, copy, time
import numpy as np
from collections import defaultdict
import pandas as pd
from .utils import *

import math
from tqdm import tqdm

import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import dill
from functools import partial
import multiprocessing as mp
import torch.nn.functional as F

class Graph():
    def __init__(self):
        super(Graph, self).__init__()
        '''
            node_forward and bacward are only used when building the data. 
            Afterwards will be transformed into node_feature by DataFrame
            
            node_forward: name -> node_id
            node_bacward: node_id -> feature_dict
            node_feature: a DataFrame containing all features
        '''
        self.node_forward = defaultdict(lambda: {})
        self.node_bacward = defaultdict(lambda: [])
        self.node_feature = defaultdict(lambda: [])

        '''
            edge_list: index the adjacancy matrix (time) by 
            <target_type, source_type, relation_type, target_id, source_id>
        '''
        self.edge_list = defaultdict( #target_type
                            lambda: defaultdict(  #source_type
                                lambda: defaultdict(  #relation_type
                                    lambda: defaultdict(  #target_id
                                        lambda: defaultdict( #source_id(
                                            lambda: float # weight
                                        )))))
        self.times = {}
        self.nodes = []
        self.edge_types = []
        self.edges = []
        self.node_feature = []
        self.fea_id = []
        self.train_pairs = []
        self.val_pairs = []
        self.test_pairs = []
        self.max_edgetype_nums = 0
    def add_node(self, node):
        if node['id'] not in self.nodes:
            self.nodes.append(node['id'])
        return self.nodes.index(node['id'])
    def add_edge(self, source_node, target_node, weight, relation_type = None, directed = False):
        edge = [self.add_node(source_node), self.add_node(target_node)]
        '''
            Add bi-directional edges with different relation type
        '''
        self.edge_list[target_node['type']][source_node['type']][relation_type][edge[1]][edge[0]] = weight
        
        if directed:
            self.edge_list[source_node['type']][target_node['type']]['rev_' + relation_type][edge[0]][edge[1]] = weight
        else:
            self.edge_list[source_node['type']][target_node['type']][relation_type][edge[0]][edge[1]] = weight
            self.edges.append([edge[0],edge[1]])
        
    def update_node(self, node):
        nbl = self.node_bacward[node['type']]
        ser = self.add_node(node)
        for k in node:
            if k not in nbl[ser]:
                nbl[ser][k] = node[k]

    def get_meta_graph(self):
        metas = []
        for target_type in self.edge_list:
            for source_type in self.edge_list[target_type]:
                for r_type in self.edge_list[target_type][source_type]:
                    metas += [(target_type, source_type, r_type)]
        return metas
    
    def get_types(self):
        return list(self.node_feature.keys())

    
def edge_type_data(type_graph):
    edge_index = []
    edge_weight = []
    for i,j,w in type_graph.edges.data():
        edge_index += [[list(type_graph.nodes()).index(i),list(type_graph.nodes()).index(j)]]
        edge_index += [[list(type_graph.nodes()).index(j),list(type_graph.nodes()).index(i)]]
        edge_weight +=[w['weight']]
        edge_weight +=[w['weight']]
    edge_index = torch.LongTensor(edge_index).t()
    edge_weight = F.normalize(torch.FloatTensor(edge_weight),dim=0)
    edge_type_num = len(type_graph.nodes())
    
    return edge_type_num, edge_index, edge_weight    

    
class RenameUnpickler(dill.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "GPT_GNN.data" or module == 'data':
            renamed_module = "pyHGT.data"
        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()
