import os
import sys
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
import scanpy as sc
import anndata as ad
from .TOSICA_model import scTrans_model as create_model
from TOSICA.train import compute_edge_weights
from TOSICA.train import knn_module
from torch_geometric.loader import NeighborSampler
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

import torch.nn as nn

#model_weight_path = "./weights20220429/model-5.pth" 
#mask_path = os.getcwd()+'/mask.npy'

def todense(adata):
    import scipy
    if isinstance(adata.X, scipy.sparse.csr_matrix) or isinstance(adata.X, scipy.sparse.csc_matrix):
        return adata.X.todense()
    else:
        return adata.X

def get_weight(att_mat,pathway):
    att_mat = torch.stack(att_mat).squeeze(1)
    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)
    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]
    
    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    v = pd.DataFrame(v[0,1:].detach().numpy()).T
    #print(v.size())
    v.columns = pathway
    return v


def mergeCGraph(adata, edge_index):
    graph_comm = adata.uns['communication_graph']['edge_index']

    if isinstance(graph_comm, (list, np.ndarray)):
        graph_comm = torch.tensor(graph_comm, dtype=torch.long)

    if graph_comm.shape[0] != 2:
        raise ValueError("Invalid shape for edge_index while predicting")

    edge_index = torch.cat([edge_index, graph_comm], dim=1)

    return edge_index


def buildPreGraph(adata, batch, label_name='Celltype'):
    neighbor = 6
    print('neighbor:', neighbor)
    # batch_size = 128
    # valid_batch = [4, 13, 18]
    # 获取adata.obs中的'x'和'y'值
    x = adata.obs['x'].values
    y = adata.obs['y'].values
    # 将'x'和'y'值组合成坐标矩阵
    coordinates = torch.tensor(list(zip(x, y)))
    mask = adata.obs['batch'].isin(batch)
    batch_pre = adata.obs.loc[mask, 'batch']

    batch_pre = torch.from_numpy(np.array(batch_pre).astype(np.int64))
    # 使用knn_module函数计算edge_index
    edge_index = knn_module(coordinates, batch_pre, neighbor)
    # 计算edge_weight
    # edge_weights = compute_edge_weights(coordinates, edge_index)

    node_idx = torch.arange(coordinates.shape[0])

    batch_size = len(node_idx)
    # batch_size = 16

    # 合并通信图
    print('edge_index shape:', edge_index.shape)
    edge_index = mergeCGraph(adata, edge_index)
    print('edge_index after merge shape:', edge_index.shape)

    sampler = NeighborSampler(
        edge_index, node_idx=node_idx,
        sizes=[neighbor],  # 指定采样的邻居规模
        batch_size=batch_size,   # 一次性将整个图的节点都采样
        shuffle=False,
        num_workers=4,
    )

    exp_gcn = pd.DataFrame(todense(adata), index=np.array(adata.obs_names).tolist(),
                           columns=np.array(adata.var_names).tolist())
    genes = exp_gcn.columns.values
    exp_gcn = np.array(exp_gcn)
    exp_gcn = exp_gcn.astype(np.float32)
    exp_gcn = torch.from_numpy(np.array(exp_gcn).astype(np.float32))

    label = adata.obs[label_name].astype('str')
    label_encoder = LabelEncoder()
    label = label_encoder.fit_transform(label)
    inverse = label_encoder.inverse_transform(range(0, np.max(label) + 1))
    label = label.astype(np.float32)
    label = torch.from_numpy(np.array(label).astype(np.int64))
    return sampler, exp_gcn, label, edge_index, inverse, genes, coordinates


def prediect(adata,model_weight_path,project,mask_path,laten=False,save_att = 'X_att', save_lantent = 'X_lat',n_step=10000,cutoff=0.1,n_unannotated = 1,batch_size = 50,embed_dim=48,depth=2,num_heads=4, use_gcn=False, data_batch=[]):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    print("depth:",depth)
    print("num_heads:",num_heads)
    print("embed_dim:",embed_dim)
    print("use_gcn:",use_gcn)
    
    num_genes = adata.shape[1]
    #mask_path = os.getcwd()+project+'/mask.npy'
    mask = np.load(mask_path)
    project_path = os.getcwd()+'/%s'%project
    pathway = pd.read_csv(project_path+'/pathway.csv', index_col=0)
    dictionary = pd.read_table(project_path+'/label_dictionary.csv', sep=',',header=0,index_col=0)
    n_c = len(dictionary)
    label_name = dictionary.columns[0]
    dictionary.loc[(dictionary.shape[0])] = 'Unknown'
    dic = {}
    for i in range(len(dictionary)):
        dic[i] = dictionary[label_name][i]

    # pre_batch = [19]
    # pre_batch_dlpfc = ['151673', '151674', '151675', '151676']
    # pre_batch_dlpfc = ['151673']
    sampler, exp_gcn_pre, label_pre, edge_index_pre, inverse, genes, coordinates_pre = buildPreGraph(adata, data_batch, label_name)


    model = create_model(num_classes=n_c, num_genes=num_genes,mask = mask, has_logits=False,depth=depth,num_heads=num_heads, embed_dim=embed_dim, use_gcn=use_gcn).to(device)
    # model = create_model(num_classes=n_c, num_genes=num_genes,mask = mask, has_logits=False,depth=depth,num_heads=num_heads, embed_dim=embed_dim, use_gcn=use_gcn)

    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)
    # model = model.to(device)

    # load model weights
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    parm={}
    for name,parameters in model.named_parameters():
        #print(name,':',parameters.size())
        parm[name]=parameters.detach().cpu().numpy()
    gene2token = parm['feature_embed.fe.weight']
    gene2token = gene2token.reshape((int(gene2token.shape[0]/embed_dim),embed_dim,adata.shape[1]))
    gene2token = abs(gene2token)
    gene2token = np.max(gene2token,axis=1)
    gene2token = pd.DataFrame(gene2token)
    gene2token.columns=adata.var_names
    gene2token.index = pathway['0']
    gene2token.to_csv(project_path+'/gene2token_weights.csv')
    latent = torch.empty([0,embed_dim]).cpu()
    att = torch.empty([0,(len(pathway))]).cpu()
    predict_class = np.empty(shape=0)
    pre_class = np.empty(shape=0)      
    latent = torch.squeeze(latent).cpu().numpy()
    l_p = np.c_[latent, predict_class,pre_class]
    att = np.c_[att, predict_class,pre_class]
    adata_list = []
    # batch_size = 128

    with torch.no_grad():
        # predict class
        data_loader = tqdm(sampler)
        for step, data in enumerate(data_loader):
            #print(step)
            _, n_id, adjs = data
            # Sampled batch
            exp = exp_gcn_pre[n_id].to(device)
            edge_index, _, _ = adjs.to(device)
            # Forward pass
            coordinates = coordinates_pre[n_id].to(device)
            edge_weights = compute_edge_weights(coordinates, edge_index)
            # _, pred, _ = model(exp.to(device))
            lat, pre, _ = model(exp.to(device), edge_index, edge_weights, coordinates)
            # lat, pre, weights = model(exp.to(device))
            # lat, pre = model(exp.to(device))
            pre = torch.squeeze(pre).cpu()
            pre = F.softmax(pre,1)
            # 将pre保存到pre.csv文件中
            # pre_to_save = pd.DataFrame(pre.numpy())
            # pre_to_save = pd.concat([pre_to_save, pd.DataFrame(coordinates.cpu().numpy())], axis=1)
            # pre_to_save.to_csv('pre.csv')

            predict_class = np.empty(shape=0)
            pre_class = np.empty(shape=0) 
            for i in range(len(pre)):
                if torch.max(pre, dim=1)[0][i] >= cutoff: 
                    predict_class = np.r_[predict_class,torch.max(pre, dim=1)[1][i].numpy()]
                else:
                    predict_class = np.r_[predict_class,n_c]
                pre_class = np.r_[pre_class,torch.max(pre, dim=1)[0][i]]     
            l_p = torch.squeeze(lat).cpu().numpy()
            # att = torch.squeeze(weights).cpu().numpy()
            meta = np.c_[predict_class,pre_class]
            meta = pd.DataFrame(meta)
            meta.columns = ['Prediction','Probability']
            meta.index = meta.index.astype('str')
            if laten:
                l_p = l_p.astype('float32')
                new = sc.AnnData(l_p, obs=meta)
            else:
                # att = att[:,0:(len(pathway)-n_unannotated)]
                # att = att.astype('float32')
                varinfo = pd.DataFrame(pathway.iloc[0:len(pathway)-n_unannotated,0].values,index=pathway.iloc[0:len(pathway)-n_unannotated,0],columns=['pathway_index'])
                temp_X = np.ones((meta.shape[0], varinfo.shape[0]))
                new = sc.AnnData(temp_X, obs=meta, var = varinfo)
            adata_list.append(new)
    new = ad.concat(adata_list)
    new.obs.index = adata.obs.index
    new.obs['Prediction'] = new.obs['Prediction'].map(dic)
    new.obs[adata.obs.columns] = adata.obs[adata.obs.columns].values
    return(new)
