from .conv import *

class Classifier(nn.Module):
    def __init__(self, n_hid, n_out):
        super(Classifier, self).__init__()
        self.n_hid    = n_hid
        self.n_out    = n_out
        self.linear   = nn.Linear(n_hid,  n_out)
    def forward(self, x):
        tx = self.linear(x)
        return torch.sigmoid(tx.squeeze())
    def __repr__(self):
        return '{}(n_hid={}, n_out={})'.format(
            self.__class__.__name__, self.n_hid, self.n_out)

class Matcher(nn.Module):
    '''
        Matching between a pair of nodes to conduct link prediction.
        Use multi-head attention as matching model.
    '''
    def __init__(self, n_hid):
        super(Matcher, self).__init__()
        self.left_linear    = nn.Linear(n_hid,  n_hid)
        self.right_linear   = nn.Linear(n_hid,  n_hid)
        self.sqrt_hd  = math.sqrt(n_hid)
        self.cache      = None
    def forward(self, x, y, infer = False, pair = False):
        ty = self.right_linear(y)
        if infer:
            '''
                During testing, we will consider millions or even billions of nodes as candidates (x).
                It's not possible to calculate them again for different query (y)
                Since the model is fixed, we propose to cache them, and dirrectly use the results.
            '''
            if self.cache != None:
                tx = self.cache
            else:
                tx = self.left_linear(x)
                self.cache = tx
        else:
            tx = self.left_linear(x)
        if pair:
            res = (tx * ty).sum(dim=-1)
        else:
            res = torch.matmul(tx, ty.transpose(0,1))
        return res / self.sqrt_hd
    def __repr__(self):
        return '{}(n_hid={})'.format(
            self.__class__.__name__, self.n_hid)
    

class TypeGAT(nn.Module):
    def __init__(self, in_dim, n_hid, edge_type_num, n_heads_type, n_layers_type, 
        node_num, n_heads_node, n_layers_node, num_types, num_relations,
        n_heads_nedge, n_layers_nedge, edge_type_array,edge_type_enum, pretrained,
        dropout = 0.2, prev_norm = True, last_norm = True):
        super(TypeGAT, self).__init__()
        self.in_dim = in_dim
        self.n_hid     = n_hid
        
        self.n_heads_type = n_heads_type
        self.edge_type_num = edge_type_num
        self.edge_type_array = edge_type_array
        self.edge_type_enum = edge_type_enum
        self.edge_type_features = torch.nn.Embedding(edge_type_num+1, n_hid, padding_idx=edge_type_num)
        self.gct = nn.ModuleList()
        for l in range(n_layers_type):
            self.gct.append(GATConv2(self.n_hid, self.n_hid, self.n_heads_type))

        self.gcs = nn.ModuleList()
        self.node_num = node_num
        self.num_types_node = num_types
        self.node_features = torch.nn.Embedding(node_num, in_dim)
        self.node_features.weight.data.copy_(pretrained)
        #self.node_features.weight.requires_grad = False
        del pretrained
        self.adapt_ws  = nn.ModuleList()
        self.drop      = nn.Dropout(dropout)
        print('n',node_num)
        
        for t in range(num_types):
            self.adapt_ws.append(nn.Linear(in_dim, n_hid))
        for l in range(n_layers_node - 1):
            self.gcs.append(GeneralConv('htn', n_hid, n_hid, num_types, num_relations, n_heads_node, dropout, use_norm = prev_norm))
        self.gcs.append(GeneralConv('htn', n_hid, n_hid, num_types, num_relations, n_heads_node, dropout, use_norm = last_norm))
        self.num_types_nedge = num_relations
        
        self.gcse = nn.ModuleList()
        
        num_relations_nedge = num_types
        
        
        for l in range(n_layers_nedge - 1):
            self.gcse.append(GeneralConv('hte', n_hid, n_hid, self.num_types_nedge, num_relations_nedge, n_heads_nedge, dropout, use_norm = prev_norm))
        self.gcse.append(GeneralConv('hte', n_hid, n_hid, self.num_types_nedge, num_relations_nedge, n_heads_nedge, dropout, use_norm = last_norm))
               


    def forward(self, edge_type_feature, edge_type_index, edge_type_weight, 
        node_id, node_type, edge_index, edge_type_id,
        ednode_type, ededge_index, ededge_type, edge_node1, edge_node2):
        #edge_type_feature是切分的边类型，二维tensor
        #edge_type_array是合成好的边类型，三维tensor
        #edge_type_id是合成好的边类型id,即就是在c-g-d网络中的边类型
        
        type_feature = self.edge_type_features(edge_type_feature)
        for gc in self.gct:
            type_feature = gc(type_feature, edge_type_index, edge_type_weight)
        
         
        pad = torch.zeros(1, self.n_hid).to(edge_type_feature.device)       
        type_features_update = torch.cat((type_feature,pad),dim = 0)
        
        weight = self.edge_type_enum.reshape((self.edge_type_enum.shape[0],1)).to(edge_type_feature.device)
        relation_type_fea = F.gelu(type_features_update[self.edge_type_array.to(edge_type_feature.device)].sum(dim=1) * weight)    
        del type_feature,type_features_update
        
        
        #*******train node features*******
        node_feature = self.node_features(node_id)
        res = torch.zeros(node_feature.size(0), self.n_hid).to(edge_type_feature.device)
        
        for t_id in range(self.num_types_node):
            idx = (node_type == int(t_id))
            if idx.sum() == 0:
                continue
            res[idx] = torch.tanh(self.adapt_ws[t_id](node_feature[idx]))
        
        meta_xs = self.drop(res)
        del res
        
        
        for gc in self.gcs:
            meta_xs = gc(meta_xs, node_type, edge_index, edge_type_id, relation_type_fea)
        
        #**********************************

        edge_node1_feature = meta_xs[edge_node1]
        edge_node2_feature = meta_xs[edge_node2]
        
        #*******train nedge features*******
        
        etype_feature = relation_type_fea[ednode_type]
        nedge_node_feature = edge_node1_feature * etype_feature * edge_node2_feature
        meta_xse = self.drop(nedge_node_feature)
        
        
        del nedge_node_feature,etype_feature,edge_node1_feature,edge_node2_feature
        
        #for gc in self.gcse:
        #    meta_xse = gc(meta_xse, ednode_type, ededge_index, ededge_type, relation_type_fea)
        
        #**********************************   

        return meta_xse


