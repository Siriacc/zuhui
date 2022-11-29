import sys
from pyHGT.data import *
from pyHGT.model import *
from warnings import filterwarnings
filterwarnings("ignore")

import argparse
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='Training GNN on Paper-Field (L2) classification task')

'''
    Dataset arguments
'''
parser.add_argument('--data_dir', type=str, default='ttd',
                    help='The address of preprocessed graph.')
parser.add_argument('--model_dir', type=str, default='./model_save',
                    help='The address for storing the models and optimization results.')
parser.add_argument('--task_name', type=str, default='ttd_2',
                    help='The name of the stored models and optimization results.')
parser.add_argument('--cuda', type=int, default=0,
                    help='Avaiable GPU ID')
parser.add_argument('--domain', type=str, default='_CS',
                    help='CS, Medicion or All: _CS or _Med or (empty)')         
'''
   Model arguments 
'''
parser.add_argument('--conv_name', type=str, default='htn',
                    choices=['htn', 'gcn', 'gat', 'rgcn'],
                    help='The name of GNN filter.')
parser.add_argument('--n_hid', type=int, default=256,
                    help='Number of hidden dimension')
parser.add_argument('--n_heads', type=int, default=8,
                    help='Number of attention head')
parser.add_argument('--n_layers', type=int, default=1,
                    help='Number of GNN layers')
parser.add_argument('--dropout', type=int, default=0.2,
                    help='Dropout ratio')
parser.add_argument('--sample_depth', type=int, default=6,
                    help='How many numbers to sample the graph')
parser.add_argument('--sample_width', type=int, default=128,
                    help='How many nodes to be sampled per layer per type')

'''
    Optimization arguments
'''
parser.add_argument('--optimizer', type=str, default='adamw',
                    choices=['adamw', 'adam', 'sgd', 'adagrad'],
                    help='optimizer to use.')
parser.add_argument('--data_percentage', type=int, default=1.0,
                    help='Percentage of training and validation data to use')
parser.add_argument('--n_epoch', type=int, default=100,
                    help='Number of epoch to run')
parser.add_argument('--n_pool', type=int, default=4,
                    help='Number of process to sample subgraph')    
parser.add_argument('--n_batch', type=int, default=32,
                    help='Number of batch (sampled graphs) for each epoch') 
parser.add_argument('--repeat', type=int, default=1,
                    help='How many time to train over a singe batch (reuse data)') 
parser.add_argument('--batch_size', type=int, default=32,
                    help='Number of output nodes for training')    
parser.add_argument('--clip', type=int, default=0.25,
                    help='Gradient Norm Clipping') 


args = parser.parse_args()

if args.cuda != -1:
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")

def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

mkdirs(os.path.join(args.model_dir))

graph = renamed_load(open(os.path.join(args.data_dir + '/graph_ttd_256_4.pk'), 'rb'))


edge_type_graph = renamed_load(open(os.path.join(args.data_dir + '/graph_edgetype_ttd.pk'), 'rb'))
edge_type_num, edge_type_index, edge_type_weight = edge_type_data(edge_type_graph)
edge_type_node = torch.tensor(range(edge_type_num),dtype = torch.long)

#edge types 
re_type_nodes = list(edge_type_graph.nodes())
edge_type_input = []
edge_type_enum = []
for types in graph.edge_types:
    n = 0
    temp = [len(re_type_nodes) for j in range(graph.max_edgetype_nums)]
    for i,re_type in enumerate(types.split('|')):
        temp[i] = re_type_nodes.index(re_type)
        n = n + 1
    edge_type_input +=[temp]
    edge_type_enum.append(1/n)
edge_type_input = torch.tensor(edge_type_input)
edge_type_enum = torch.FloatTensor(edge_type_enum)



criterion = nn.BCELoss()

def sample_subnodes(inp):
    '''
        regard chemicals,genes, and diseases as nodes, the relation as edge, sample subgraphs.
        input: graph: original graph
               inp: chindex, cgtype, geindex, diindex
               re_type_nodes: relation edge list, for relationtype index
        output:
               node_index , node_type
               edge_index , edge_type
    '''
    

    def sample_node(node, stype, ttype, node_index, node_type, edge_index, edge_type):
        '''
        global node_index
        global node_type
        global edge_index
        global edge_type
        '''
        if stype==0 and ttype==1:
            te = graph.edge_list['chemical']['gene']
        elif stype==1 and ttype==0:
            te = graph.edge_list['gene']['chemical']
        elif stype==1 and ttype==2:
            te = graph.edge_list['gene']['disease']
        elif stype==2 and ttype==1:
            te = graph.edge_list['disease']['gene']
        for relation_type in te:
            if node in te[relation_type]:
                for tid in te[relation_type][node]:
                    if node not in node_index:
                        node_index +=[node]
                        node_type +=[stype]
                    if tid not in node_index:
                        node_index +=[tid]
                        node_type +=[ttype]
                    if [node, tid] not in edge_index:
                        edge_index +=[[node, tid]]
                        edge_type +=[graph.edge_types.index(relation_type)]
                    if [tid, node] not in edge_index:
                        edge_index +=[[tid, node]]
                        edge_type +=[graph.edge_types.index(relation_type)]
                        
        return node_index, node_type, edge_index, edge_type
        
    #0: chemical  
    #1: gene
    #2: disease
    node_index   = []
    node_type    = []
    edge_index   = []
    edge_type    = []
    for i in range(len(inp)):
        chindex, cgtype, geindex, gdtype, diindex = inp[i]
        node_index, node_type, edge_index, edge_type = sample_node(chindex, 0, 1, node_index, node_type, edge_index, edge_type)
        node_index, node_type, edge_index, edge_type = sample_node(geindex, 1, 0, node_index, node_type, edge_index, edge_type)
        node_index, node_type, edge_index, edge_type = sample_node(geindex, 1, 2, node_index, node_type, edge_index, edge_type)
        node_index, node_type, edge_index, edge_type = sample_node(diindex, 2, 1, node_index, node_type, edge_index, edge_type)

    edge_index_new = []
    for i in range(len(edge_index)):
        edge_index_new.append([node_index.index(edge_index[i][0]),node_index.index(edge_index[i][1])])
    
    del edge_index

    node_index   = torch.LongTensor(np.array(node_index))
    node_type    = torch.LongTensor(np.array(node_type))
    edge_index_new   = torch.LongTensor(np.array(edge_index_new)).t()
    edge_type    = torch.LongTensor(np.array(edge_type))

    
    return node_index, node_type, edge_index_new, edge_type

def sample_subedges(inp):
    '''
        regard relation edges as nodes, the common node(chemical, gene, disease) as edge, sample subgraphs.
        input: graph: original graph
               inp: chindex, cgtype, geindex, diindex
               re_type_nodes: relation edge list, for relationtype index
        output:
               node_index , node_type
               edge_index , edge_type
               edge_node1 , edge_node2: is used to trace back the nodes in original graph. A node is composed of edge_node1 and edge_node2
    '''

    node_index   = []
    node_type    = []
    edge_index   = []
    edge_type    = []

    edge_node1   = []
    edge_node2   = []

    def add_node_edge(ni,nj,ni1,ni2,nj1,nj2,retype,relation_type,edtype,\
        node_index,node_type,edge_index,edge_type,edge_node1,edge_node2):

        if ni not in node_index:
            node_index +=[ni]
            edge_node1 +=[ni1]
            edge_node2 +=[ni2]
            node_type +=[graph.edge_types.index(retype)]

        if nj not in node_index:
            node_index +=[nj]
            edge_node1 +=[nj1]
            edge_node2 +=[nj2]
            node_type +=[graph.edge_types.index(relation_type)]
            
                        
        if [ni, nj] not in edge_index:
            edge_index +=[[ni, nj]]
            edge_type +=[edtype]
        if [nj, ni] not in edge_index:
            edge_index +=[[nj, ni]]
            edge_type +=[edtype]
        return node_index,node_type,edge_index,edge_type,edge_node1,edge_node2

    def sample_edge(node1, node2, retype,flag,node_index,node_type,edge_index,edge_type,edge_node1,edge_node2):
        #node2:gene
        #node1:chemical or disease
        te = graph.edge_list

        if flag==1:
            ntype = 'disease' #type of node1
        else: 
            ntype = 'chemical'
        
        for relation_type in te['gene']['chemical']:
            if node2 in te['gene']['chemical'][relation_type]:
                for tid in te['gene']['chemical'][relation_type][node2]:
                    if tid==node1:
                        continue
                    if ntype=='chemical':
                        ni = graph.edges.index([node1,node2])
                        nj = graph.edges.index([tid,node2])
                        add_node_edge(ni,nj,node1,node2,tid,node2,retype,relation_type,1,\
                            node_index,node_type,edge_index,edge_type,edge_node1,edge_node2)
                    elif ntype=='disease':
                        ni = graph.edges.index([node2,node1])                   
                        nj = graph.edges.index([tid,node2])
                        add_node_edge(ni,nj,node2,node1,tid,node2,retype,relation_type,1,\
                            node_index,node_type,edge_index,edge_type,edge_node1,edge_node2)
                    
                    
        for relation_type in te['gene']['disease']:
            if node2 in te['gene']['disease'][relation_type]:
                for tid in te['gene']['disease'][relation_type][node2]:
                    if tid==node1:
                        continue
                    if ntype=='chemical':
                        ni = graph.edges.index([node1,node2])
                        nj = graph.edges.index([node2,tid])
                        add_node_edge(ni,nj,node1,node2,node2,tid,retype,relation_type,1,\
                            node_index,node_type,edge_index,edge_type,edge_node1,edge_node2)
                    elif ntype=='disease':
                        ni = graph.edges.index([node2,node1])                   
                        nj = graph.edges.index([node2,tid])
                        add_node_edge(ni,nj,node2,node1,node2,tid,retype,relation_type,1,\
                            node_index,node_type,edge_index,edge_type,edge_node1,edge_node2)
        

        for relation_type in te[ntype]['gene']:
            if node1 in te[ntype]['gene'][relation_type]:
                for tid in te[ntype]['gene'][relation_type][node1]:
                    if tid==node2:
                        continue
                    if ntype == 'chemical':
                        ni = graph.edges.index([node1,node2])
                        nj = graph.edges.index([node1,tid])
                        add_node_edge(ni,nj,node1,node2,node1,tid,retype,relation_type,0,\
                            node_index,node_type,edge_index,edge_type,edge_node1,edge_node2)
                    elif ntype == 'disease':
                        ni = graph.edges.index([node2,node1])
                        nj = graph.edges.index([tid,node1])
                        add_node_edge(ni,nj,node2,node1,tid,node1,retype,relation_type,2,\
                            node_index,node_type,edge_index,edge_type,edge_node1,edge_node2) 
        return node_index,node_type,edge_index,edge_type,edge_node1,edge_node2  
                    
        
    
    for i in range(len(inp)):
        chindex, cgtype, geindex, gdtype, diindex = inp[i]
        
        node_index,node_type,edge_index,edge_type,edge_node1,edge_node2 = sample_edge(chindex, geindex, cgtype,0,\
            node_index,node_type,edge_index,edge_type,edge_node1,edge_node2)
         
        node_index,node_type,edge_index,edge_type,edge_node1,edge_node2 = sample_edge(diindex, geindex, gdtype,1,\
            node_index,node_type,edge_index,edge_type,edge_node1,edge_node2)

    edge_index_new = []
    for i in range(len(edge_index)):
        edge_index_new.append([node_index.index(edge_index[i][0]),node_index.index(edge_index[i][1])])
    del edge_index

    

    node_index   = torch.LongTensor(np.array(node_index))
    node_type    = torch.LongTensor(np.array(node_type))
    edge_index_new   = torch.LongTensor(np.array(edge_index_new)).t()
    edge_type    = torch.LongTensor(np.array(edge_type))
    

    
    return node_index, node_type, edge_index_new, edge_type, edge_node1, edge_node2


def node_classification_sample(target_ids):
    
    target_info = []
    x_cg = []#c_g edge index
    x_gd = []#g_d edge index
    ylabel = np.zeros([args.batch_size, 2])
    for i, target_id in enumerate(target_ids):
        ch_id, cgtype, ge_id, gdtype, di_id, label_id = target_id
        
        chindex = graph.nodes.index(ch_id)
        geindex = graph.nodes.index(ge_id)
        diindex = graph.nodes.index(di_id)

        target_info += [[chindex, cgtype, geindex, gdtype, diindex]]
        x_cg +=[graph.edges.index([chindex,geindex])]
        x_gd +=[graph.edges.index([geindex,diindex])]
        ylabel[i][int(label_id)] = 1
    ylabel = torch.FloatTensor(np.array(ylabel))

    node_index, node_type, edge_index, edge_type = sample_subnodes(target_info)

    ednode_index, ednode_type, ededge_index, ededge_type, edge_node1, edge_node2 = sample_subedges(target_info)
    x_cgid = []
    x_gdid = []
    for i in range(len(x_cg)):
        x_cgid.append(ednode_index.numpy().tolist().index(x_cg[i]))
        x_gdid.append(ednode_index.numpy().tolist().index(x_gd[i]))
    x_cgid = torch.LongTensor(np.array(x_cgid))
    x_gdid = torch.LongTensor(np.array(x_gdid))

    edge_node1_new = []
    edge_node2_new = []
    for i in range(len(edge_node1)):
        edge_node1_new.append(node_index.numpy().tolist().index(edge_node1[i]))
        edge_node2_new.append(node_index.numpy().tolist().index(edge_node2[i]))


    del target_info,x_cg,x_gd,edge_node1, edge_node2, ednode_index

    edge_node1_new   = torch.LongTensor(np.array(edge_node1_new))
    edge_node2_new   = torch.LongTensor(np.array(edge_node2_new))

    return node_index, node_type, edge_index, edge_type, ednode_type, ededge_index, ededge_type, edge_node1_new, edge_node2_new, x_cgid, x_gdid, ylabel

def main(): 
    def prepare_data(pool):
        
        jobs = []
        pairs = np.array(train_pairs)
        
        for batch_id in np.arange(args.n_batch):           
            target_ids = pairs[np.random.choice(pairs.shape[0],args.batch_size, replace = False),:]
            p = pool.apply_async(node_classification_sample, args=(target_ids,))
            jobs.append(p)
        
        
        return jobs

    pretrained_emb = []
    for i in range(len(graph.nodes)):
        pretrained_emb.append(graph.node_feature[graph.fea_id.index(i)])

    pretrained_emb = torch.FloatTensor(np.array(pretrained_emb))
    del graph.node_feature

    train_pairs = graph.train_pairs
    val_pairs = graph.val_pairs
    test_pairs = graph.test_pairs


    gnn = TypeGAT(in_dim=256, n_hid = args.n_hid, edge_type_num=edge_type_num, n_heads_type=args.n_heads, n_layers_type= args.n_layers,
        node_num=len(graph.nodes), n_heads_node = args.n_heads, n_layers_node = args.n_layers, num_types=3, num_relations=len(graph.edge_types),
        n_heads_nedge = args.n_heads, n_layers_nedge = args.n_layers, edge_type_array=edge_type_input, edge_type_enum=edge_type_enum, pretrained=pretrained_emb).to(device)
    #type_rep = type_model.forward(edge_type,edge_type_index, edge_type_weight)
    classifier = Classifier(args.n_hid, 2).to(device)
    del pretrained_emb

    model = nn.Sequential(gnn, classifier)



    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters())
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters())
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)
    elif args.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters())

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000, eta_min=1e-6)

    stats = []
    res = []
    best_val   = 0
    train_step = 1500
    
    pool = mp.Pool(args.n_pool)
    st = time.time()
    jobs = prepare_data(pool)
    
    for epoch in np.arange(args.n_epoch) + 1:
        
        train_data = [job.get() for job in jobs]
        pool.close()
        pool.join()
       
        pool = mp.Pool(args.n_pool)
        jobs = prepare_data(pool)
        et = time.time()
        print('Data Preparation: %.1fs' % (et - st))

        model.train()
        train_losses = []
        torch.cuda.empty_cache()
        for _ in range(args.repeat):
            for node_index, node_type, edge_index, edge_type, ednode_type,ededge_index, ededge_type, edge_node1, edge_node2, x_cgid, x_gdid, ylabel in train_data:
                rep = gnn.forward(edge_type_node.to(device), edge_type_index.to(device), edge_type_weight.to(device), 
                    node_index.to(device), node_type.to(device), edge_index.to(device), edge_type.to(device), 
                    ednode_type.to(device), ededge_index.to(device), ededge_type.to(device), edge_node1.to(device), edge_node2.to(device))
                
                res  = classifier.forward(torch.mul(rep[x_cgid],rep[x_gdid]))
                
                loss = criterion(res, ylabel.to(device))
                
                optimizer.zero_grad() 
                torch.cuda.empty_cache()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()

                train_losses += [loss.cpu().detach().tolist()]
                train_step += 1
                scheduler.step(train_step)
                del res, loss, rep
        
        model.eval()
        with torch.no_grad():
            val_res = []
            val_y = []
            val_losses = []
            for batch_id in np.arange(int(len(val_pairs)/args.batch_size)):
                if (batch_id+1) * args.batch_size>=len(val_pairs):
                    target_ids =val_pairs[batch_id * args.batch_size:]
                else:
                    target_ids = val_pairs[batch_id * args.batch_size:(batch_id+1) * args.batch_size]
                node_index, node_type, edge_index, edge_type, ednode_type,ededge_index, ededge_type, edge_node1, edge_node2, x_cgid, x_gdid, ylabel = \
                        node_classification_sample(target_ids)
                rep = gnn.forward(edge_type_node.to(device), edge_type_index.to(device), edge_type_weight.to(device), 
                    node_index.to(device), node_type.to(device), edge_index.to(device), edge_type.to(device), 
                    ednode_type.to(device), ededge_index.to(device), ededge_type.to(device), edge_node1.to(device), edge_node2.to(device))
                
                res  = classifier.forward(torch.mul(rep[x_cgid],rep[x_gdid]))
                for y,pre in zip(ylabel,res):
                    val_res.append(pre.cpu().numpy().tolist())
                    val_y.append(y.numpy().tolist())
            
                loss = criterion(res, ylabel.to(device))
                val_losses += [loss.cpu().detach().tolist()]
            
            val_res = torch.from_numpy(np.array(val_res))
            val_y = torch.from_numpy(np.array(val_y))
            valid_roc = roc_auc_score(torch.argmax(val_y,dim=1), torch.argmax(val_res,dim=1))
            
            if valid_roc > best_val:
                best_val = valid_roc
                torch.save(model, os.path.join(args.model_dir, args.task_name + '_' + args.conv_name + '.pth'))
                print('UPDATE!!!')
            
            st = time.time()
            print(("Epoch: %d (%.1fs)  LR: %.5f Train Loss: %.2f  Valid Loss: %.2f  Valid ROC: %.4f") % \
                  (epoch, (st-et), optimizer.param_groups[0]['lr'], np.average(train_losses), \
                        np.average(val_losses), valid_roc))
            stats += [[np.average(train_losses), np.average(val_losses)]]
            del res, loss,rep
        del train_data

    
    with torch.no_grad():
        test_res = []
        test_y = []
        for batch_id in np.arange(int(len(test_pairs)/args.batch_size)):
            if (batch_id+1) * args.batch_size>=len(test_pairs):
                target_ids =test_pairs[batch_id * args.batch_size:]
            else:
                target_ids = test_pairs[batch_id * args.batch_size:(batch_id+1) * args.batch_size]
            node_index, node_type, edge_index, edge_type, ednode_type,ededge_index, ededge_type, edge_node1, edge_node2, x_cgid, x_gdid, ylabel = \
                        node_classification_sample(target_ids)
            rep = gnn.forward(edge_type_node.to(device), edge_type_index.to(device), edge_type_weight.to(device), 
                node_index.to(device), node_type.to(device), edge_index.to(device), edge_type.to(device), 
                ednode_type.to(device), ededge_index.to(device), ededge_type.to(device), edge_node1.to(device), edge_node2.to(device))
            
            res  = classifier.forward(torch.mul(rep[x_cgid],rep[x_gdid]))
            
            for y,pre in zip(ylabel,res):
                test_res.append(pre.cpu().numpy().tolist())
                
                test_y.append(y.numpy().tolist())
            del res,rep
        test_res = torch.from_numpy(np.array(test_res))
        test_y = torch.from_numpy(np.array(test_y))
        
        test_roc = roc_auc_score(torch.argmax(test_y,dim=1), torch.argmax(test_res,dim=1))
        print('Last Test ROC: %.4f' % np.average(test_roc))
        
        test_f1 = f1_score(torch.argmax(test_y,dim=1), torch.argmax(test_res,dim=1))
        print('Last Test F1: %.4f' % np.average(test_f1))
        
        test_precision = precision_score(torch.argmax(test_y,dim=1), torch.argmax(test_res,dim=1))
        print('Last Test Precision: %.4f' % np.average(test_precision))
        
        test_recall = recall_score(torch.argmax(test_y,dim=1), torch.argmax(test_res,dim=1))
        print('Last Test Recall: %.4f' % np.average(test_recall))
    
    best_model = torch.load(os.path.join(args.model_dir, args.task_name + '_' + args.conv_name + '.pth'))
    best_model.eval()
    gnn, classifier = best_model
    with torch.no_grad():
        test_res = []
        test_y = []
        for batch_id in np.arange(int(len(test_pairs)/args.batch_size)):
            if (batch_id+1) * args.batch_size>=len(test_pairs):
                target_ids =test_pairs[batch_id * args.batch_size:]
            else:
                target_ids = test_pairs[batch_id * args.batch_size:(batch_id+1) * args.batch_size]
            node_index, node_type, edge_index, edge_type, ednode_type,ededge_index, ededge_type, edge_node1, edge_node2, x_cgid, x_gdid, ylabel = \
                        node_classification_sample(target_ids)
            rep = gnn.forward(edge_type_node.to(device), edge_type_index.to(device), edge_type_weight.to(device), 
                node_index.to(device), node_type.to(device), edge_index.to(device), edge_type.to(device), 
                ednode_type.to(device), ededge_index.to(device), ededge_type.to(device), edge_node1.to(device), edge_node2.to(device))
            
            res  = classifier.forward(torch.mul(rep[x_cgid],rep[x_gdid]))
            for y,pre in zip(ylabel,res):
                test_res.append(pre.cpu().numpy().tolist())
                test_y.append(y.numpy().tolist())
            del res,rep
            
        test_res = torch.from_numpy(np.array(test_res))
        test_y = torch.FloatTensor(np.array(test_y))
        
        test_roc = roc_auc_score(torch.argmax(test_y,dim=1), torch.argmax(test_res,dim=1))
        print('Best Test ROC: %.4f' % np.average(test_roc))
        
        test_f1 = f1_score(torch.argmax(test_y,dim=1), torch.argmax(test_res,dim=1))
        print('Best Test F1: %.4f' % np.average(test_f1))
        
        test_precision = precision_score(torch.argmax(test_y,dim=1), torch.argmax(test_res,dim=1))
        print('Best Test Precision: %.4f' % np.average(test_precision))
        
        test_recall = recall_score(torch.argmax(test_y,dim=1), torch.argmax(test_res,dim=1))
        print('Best Test Recall: %.4f' % np.average(test_recall))
        '''
        f = open("results/output.txt", "a")
        f.write('Best Test ROC: %.4f\n' % np.average(test_roc))
        f.write('Best Test F1: %.4f\n' % np.average(test_f1))
        f.write('Best Test Precision: %.4f\n' % np.average(test_precision))
        f.close()
        '''

if __name__ == '__main__':
    main()