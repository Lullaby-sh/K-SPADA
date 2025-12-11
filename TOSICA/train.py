import random
import numpy as np

from torch.utils.data import Dataset

import sys
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict

import os
import math
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import time
import platform

from .TOSICA_model import scTrans_model as create_model
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import knn_graph
from imblearn.over_sampling import SMOTE

import torch.nn as nn


def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

def todense(adata):
    import scipy
    if isinstance(adata.X, scipy.sparse.csr_matrix) or isinstance(adata.X, scipy.sparse.csc_matrix):
        return adata.X.todense()
    else:
        return adata.X


def knn_module(x, batch, k):
    # batch 的同一个值表示同一个批次，必须是连续的，否则会edge_index与node_idx不匹配
    edge_index = []
    # 遍历每个批次
    if len(batch) == 1:
        return knn_graph(x, k, loop=False)

    sum = 0
    for b in torch.unique(batch):
        # 获取当前批次的所有点
        batch_indices = (batch == b)
        batch_points = x[batch_indices]
        # 如果当前批次内的点少于 k 个，跳过
        if torch.sum(batch_indices) < k:
            continue
        # 计算 k 近邻图
        batch_edge_index = knn_graph(batch_points, k, loop=False)
        # 将批次内的点索引映射回原始索引
        # batch_edge_index += torch.sum(~batch_indices)
        batch_edge_index += sum
        sum += batch_points.shape[0]
        # 添加到列表中
        edge_index.append(batch_edge_index)
    # 合并所有的边
    if len(edge_index) > 0:
        edge_index = torch.cat(edge_index, dim=1)
    else:
        # 如果没有边，返回一个空张量
        edge_index = torch.empty((2, 0), dtype=torch.long)
    # print(edge_index[0].max())
    # print(edge_index[0].min())
    return edge_index


class MyDataSet(Dataset):
    """ 
    Preproces input matrix and labels.

    """
    def __init__(self, exp, label, edge_index, edge_weight, edges=5):
        self.exp = exp
        self.label = label
        self.len = len(label)
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.edges = edges

    def compute_graph(self, node_index):
        # 找到包含指定节点的边的索引
        edge_indices = (self.edge_index[0] == node_index) | (self.edge_index[1] == node_index)
        # 获取包含指定节点的边
        sub_edge_index = self.edge_index[:, edge_indices]
        # 获取这些边对应的权重
        sub_edge_weight = self.edge_weight[edge_indices]

        # 如果边的数量少于self.edges，复制现有的边
        while sub_edge_index.shape[1] < self.edges:
            sub_edge_index = torch.cat((sub_edge_index, sub_edge_index), dim=1)
            sub_edge_weight = torch.cat((sub_edge_weight, sub_edge_weight))

        # 如果边的数量多于self.edges，只取前self.edges个
        if sub_edge_index.shape[1] > self.edges:
            sub_edge_index = sub_edge_index[:, :self.edges]
            sub_edge_weight = sub_edge_weight[:self.edges]

        return sub_edge_index, sub_edge_weight

    def __getitem__(self,index):
        # sub_edge_index, sub_edge_weight = self.compute_graph(index)
        return self.exp[index], self.label[index], self.edge_index, self.edge_weight

    def __len__(self):
        return self.len

def balance_populations(data):
    ct_names = np.unique(data[:,-1])
    ct_counts = pd.value_counts(data[:,-1])
    max_val = min(ct_counts.max(),np.int32(2000000/len(ct_counts)))
    balanced_data = np.empty(shape=(1,data.shape[1]),dtype=np.float32)
    # 每种不同类型的细胞取同样的数量，为数量最多的那个类型的数量
    for ct in ct_names:
        tmp = data[data[:,-1] == ct]
        idx = np.random.choice(range(len(tmp)), max_val)
        tmp_X = tmp[idx]
        balanced_data = np.r_[balanced_data,tmp_X]
    return np.delete(balanced_data,0,axis=0)
  
def splitDataSet(adata, edge_index, edge_weights, label_name='Celltype', tr_ratio= 0.7):
    """ 
    Split data set into training set and test set.

    """
    label_encoder = LabelEncoder()
    el_data = pd.DataFrame(todense(adata),index=np.array(adata.obs_names).tolist(), columns=np.array(adata.var_names).tolist())
    el_data[label_name] = adata.obs[label_name].astype('str')
    #el_data = pd.read_table(data_path,sep=",",header=0,index_col=0)
    genes = el_data.columns.values[:-1]
    el_data = np.array(el_data)
    # el_data = np.delete(el_data,-1,axis=1)
    el_data[:,-1] = label_encoder.fit_transform(el_data[:,-1])
    inverse = label_encoder.inverse_transform(range(0,np.max(el_data[:,-1])+1))
    el_data = el_data.astype(np.float32)
    # 不平衡数据集
    # el_data = balance_populations(data = el_data)
    n_genes = len(el_data[1])-1
    train_size = int(len(el_data) * tr_ratio)
    train_dataset, valid_dataset = torch.utils.data.random_split(el_data, [train_size,len(el_data)-train_size])
    exp_train = torch.from_numpy(np.array(train_dataset)[:,:n_genes].astype(np.float32))
    label_train = torch.from_numpy(np.array(train_dataset)[:,-1].astype(np.int64))
    exp_valid = torch.from_numpy(np.array(valid_dataset)[:,:n_genes].astype(np.float32))
    label_valid = torch.from_numpy(np.array(valid_dataset)[:,-1].astype(np.int64))

    # Split edge_index and edge_weight
    train_indices = torch.tensor(train_dataset.indices)
    valid_indices = torch.tensor(valid_dataset.indices)

    # 找到 edge_index 中起点或者终点等于 train_indices 中任意一个元素的边
    mask = torch.any(torch.eq(edge_index[0].view(-1, 1), train_indices), dim=1) | torch.any(
        torch.eq(edge_index[1].view(-1, 1), train_indices), dim=1)

    # 使用掩码来筛选边
    edge_index_train = edge_index[:, mask]
    edge_weights_train = edge_weights[mask]

    mask_valid = torch.any(torch.eq(edge_index[0].view(-1, 1), valid_indices), dim=1) | torch.any(
        torch.eq(edge_index[1].view(-1, 1), valid_indices), dim=1)
    edge_index_valid = edge_index[:, mask_valid]
    edge_weights_valid = edge_weights[mask_valid]

    return exp_train, label_train, exp_valid, label_valid, edge_index_train, edge_weights_train, edge_index_valid, edge_weights_valid, inverse, genes

def get_gmt(gmt):
    import pathlib
    root = pathlib.Path(__file__).parent
    gmt_files = {
        "human_gobp": [root / "resources/GO_bp.gmt"],
        "human_immune": [root / "resources/immune.gmt"],
        "human_reactome": [root / "resources/reactome.gmt"],
        "human_tf": [root / "resources/TF.gmt"],
        "mouse_gobp": [root / "resources/m_GO_bp.gmt"],
        "mouse_reactome": [root / "resources/m_reactome.gmt"],
        "mouse_tf": [root / "resources/m_TF.gmt"]
    }
    return gmt_files[gmt][0]

def read_gmt(fname, sep='\t', min_g=0, max_g=5000):
    """
    Read GMT file into dictionary of gene_module:genes.\n
    min_g and max_g are optional gene set size filters.

    Args:
        fname (str): Path to gmt file
        sep (str): Separator used to read gmt file.
        min_g (int): Minimum of gene members in gene module.
        max_g (int): Maximum of gene members in gene module.
    Returns:
        OrderedDict: Dictionary of gene_module:genes.
    """
    dict_pathway = OrderedDict()
    with open(fname) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            val = line.split(sep)
            if min_g <= len(val[2:]) <= max_g:
                dict_pathway[val[0]] = val[2:]
    return dict_pathway

def create_pathway_mask(feature_list, dict_pathway, add_missing=1, fully_connected=True, to_tensor=False):
    """
    Creates a mask of shape [genes,pathways] where (i,j) = 1 if gene i is in pathway j, 0 else.

    Expects a list of genes and pathway dict.
    Note: dict_pathway should be an Ordered dict so that the ordering can be later interpreted.

    Args:
        feature_list (list): List of genes in single-cell dataset.
        dict_pathway (OrderedDict): Dictionary of gene_module:genes.
        add_missing (int): Number of additional, fully connected nodes.
        fully_connected (bool): Whether to fully connect additional nodes or not.
        to_tensor (False): Whether to convert mask to tensor or not.
    Returns:
        torch.tensor/np.array: Gene module mask.
    """
    assert type(dict_pathway) == OrderedDict
    p_mask = np.zeros((len(feature_list), len(dict_pathway)))
    pathway = list()
    for j, k in enumerate(dict_pathway.keys()):
        pathway.append(k)
        for i in range(p_mask.shape[0]):
            if feature_list[i] in dict_pathway[k]:
                p_mask[i,j] = 1.
    # 最后一列加一列全连接
    if add_missing:
        n = 1 if type(add_missing)==bool else add_missing
        # Get non connected genes
        if not fully_connected:
            idx_0 = np.where(np.sum(p_mask, axis=1)==0)
            vec = np.zeros((p_mask.shape[0],n))
            vec[idx_0,:] = 1.
        else:
            vec = np.ones((p_mask.shape[0], n))
        p_mask = np.hstack((p_mask, vec))
        for i in range(n):
            x = 'node %d' % i
            pathway.append(x)
    if to_tensor:
        p_mask = torch.Tensor(p_mask)
    return p_mask,np.array(pathway)

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """
    Train the model and updata weights.
    """
    model.train()
    loss_function = torch.nn.CrossEntropyLoss() 
    accu_loss = torch.zeros(1).to(device) 
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()
    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        exp, label = data
        sample_num += exp.shape[0]
        _,pred,_ = model(exp.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, label.to(device)).sum()
        loss = loss_function(pred, label.to(device))
        loss.backward()
        accu_loss += loss.detach()
        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        optimizer.step() 
        optimizer.zero_grad()
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)
    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        exp, labels = data
        sample_num += exp.shape[0]
        _,pred,_ = model(exp.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        loss = loss_function(pred, labels.to(device))
        accu_loss += loss
        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

def compute_edge_weights(x, edge_index):
    # x is a tensor where each row is a point in a d-dimensional space
    # edge_index is the edge indices as returned by knn_graph
    # Get the coordinates of the two points for each edge
    point1 = x[edge_index[0]]
    point2 = x[edge_index[1]]
    # Compute the Euclidean distance between the points
    edge_weights = torch.sqrt(torch.sum((point1 - point2) ** 2, dim=1))
    return edge_weights

def test():
    # 创建模拟数据
    exp = torch.randn(10, 5)  # 假设有10个样本，每个样本有5个特征
    label = torch.randint(0, 2, (10,))  # 假设有2个类别
    edge_index = torch.randint(0, 10, (2, 20))  # 假设有20条边
    edge_weight = torch.randn(20)  # 每条边有一个权重

    # 创建MyDataSet实例
    dataset = MyDataSet(exp, label, edge_index, edge_weight)
    print(exp)
    print(edge_index)
    # 遍历数据集并打印每个元素的内容
    for i in range(len(dataset)):
        exp_i, label_i, edge_index_i, edge_weight_i = dataset[i]
        print(f"Sample {i}:")
        print(f"Expression: {exp_i}")
        print(f"Label: {label_i}")
        print(f"Edge index: {edge_index_i}")
        print(f"Edge weight: {edge_weight_i}")
        print()

def duplicate_nodes(edge_index, node_idx, nodes_to_duplicate):
    # 创建新的节点索引
    new_node_idx = torch.arange(node_idx.max() + 1, node_idx.max() + 1 + len(nodes_to_duplicate))

    # 创建新的边
    new_edges = []
    for old_node, new_node in zip(nodes_to_duplicate, new_node_idx):
        # 找到旧节点的邻居
        neighbors = edge_index[1][edge_index[0] == old_node]
        # 创建新的边
        new_edges.extend([(new_node, neighbor) for neighbor in neighbors])
        # 找到旧节点的邻居(逆向), 不需要去重
        # neighbors = edge_index[0][edge_index[1] == old_node]
        # new_edges.extend([(neighbor, new_node) for neighbor in neighbors])

    # 将新的边添加到edge_index
    edge_index = torch.cat([edge_index, torch.tensor(new_edges).t().contiguous()], dim=1)

    # 更新node_idx
    node_idx = torch.cat([node_idx, new_node_idx])

    return edge_index, node_idx


def get_nodes_to_duplicate(batches_celltype):
    nodes_to_duplicate_all = []
    sum = 0
    for i in range(len(batches_celltype)):
        # 统计每个标签的数量
        label_counts = {}
        label_counts = batches_celltype[i].value_counts()

        # 找到数量最多的标签的数量
        max_count = label_counts.max()

        # 记录需要复制的节点(行index)
        nodes_to_duplicate = []
        # 获取第i个batch的数据
        batch = pd.DataFrame(batches_celltype[i], columns=['Celltype'])

        # 按标签分组，并获取每个组的行号
        label_indices = batch.groupby('Celltype').indices

        # 遍历每个标签
        for label, count in label_counts.items():
            if count == max_count or count == 0:
                continue
            # 计算需要复制的数量
            num_to_duplicate = max_count - count
            nodes = label_indices[label]
            # 将nodes转换为tensor
            # 找到当前标签的所有节点的行号
            if num_to_duplicate <= len(nodes):
                nodes = nodes[:num_to_duplicate]
            else:
                temp = nodes
                while num_to_duplicate > len(nodes):
                    nodes = np.concatenate((nodes, temp), axis=0)
                nodes = nodes[:num_to_duplicate]
            nodes += sum
            # 添加需要复制的节点
            nodes_to_duplicate.extend(nodes)
        nodes_to_duplicate_all.extend(nodes_to_duplicate)
        sum += len(batches_celltype[i])
    return nodes_to_duplicate_all


def oversample_graph(adata, edge_index_train, node_idx_train, edge_index_valid, node_idx_valid, 
                     coordinates_train, coordinates_valid, train_batch):
    all_batch = adata.obs['batch'].unique().tolist()
    batches_train_celltype = []
    for i in range(len(train_batch)):
        batch_celltype = adata.obs.loc[adata.obs['batch'] == train_batch[i], 'Celltype']
        batches_train_celltype.append(batch_celltype)
    
    # valid_batch必须有顺序！！！！否则nodeid 和原始的数据的nodeid不一致
    # 在数据处理阶段的obs['batch']字符串中不能出现字母
    valid_batch = list(set(all_batch).difference(set(train_batch)))
    valid_batch.sort()
    batches_valid_celltype = []
    for i in range(len(valid_batch)):
        batch_celltype = adata.obs.loc[adata.obs['batch'] == valid_batch[i], 'Celltype']
        batches_valid_celltype.append(batch_celltype)

    nodes_to_duplicate_train = get_nodes_to_duplicate(batches_train_celltype)
    nodes_to_duplicate_valid = get_nodes_to_duplicate(batches_valid_celltype)
    # 如果需要复制的节点为空, 如果是空的可能会有bug
    if len(nodes_to_duplicate_train) == 0:
        new_edge_index_train = edge_index_train
        new_node_idx_train = node_idx_train
    else:
        new_edge_index_train, new_node_idx_train = duplicate_nodes(edge_index_train, node_idx_train, nodes_to_duplicate_train)# 
    if len(nodes_to_duplicate_valid) == 0:
        new_edge_index_valid = edge_index_valid
        new_node_idx_valid = node_idx_valid
    else:
        new_edge_index_valid, new_node_idx_valid = duplicate_nodes(edge_index_valid, node_idx_valid, nodes_to_duplicate_valid)

    # 可视化测试oversample后的图
    # adata = adata[adata.obs['batch'].isin(train_batch)]
    # coordinates_train_output = coordinates_train[nodes_to_duplicate_train]
    # batch_train_output = adata.obs['batch'][nodes_to_duplicate_train]
    # celltype_train_output = adata.obs['Celltype'][nodes_to_duplicate_train]
    # all_output = np.column_stack((coordinates_train_output[:, 0].numpy(), coordinates_train_output[:, 1].numpy(), batch_train_output, celltype_train_output))
    # all_output = pd.DataFrame(all_output, columns=['x', 'y', 'batch', 'Celltype']).reset_index()
    # all_output.to_csv('display_oversample_train.csv')

    # adata = adata[adata.obs['batch'].isin(valid_batch)]
    # # adata.obs['batch'].to_csv('display_oversample_valid_batch.csv')
    # # nodes_to_duplicate_valid = pd.DataFrame(nodes_to_duplicate_valid, columns=['index'])
    # # nodes_to_duplicate_valid.to_csv('display_oversample_valid_nodes.csv')
    # coordinates_valid_output = coordinates_valid[nodes_to_duplicate_valid]
    # batch_valid_output = adata.obs['batch'][nodes_to_duplicate_valid]
    # # batch_valid_output.to_csv('display_oversample_valid_batch_output.csv')
    # celltype_valid_output = adata.obs['Celltype'][nodes_to_duplicate_valid]
    # all_output = np.column_stack((coordinates_valid_output[:, 0].numpy(), coordinates_valid_output[:, 1].numpy(), batch_valid_output, celltype_valid_output))
    # all_output = pd.DataFrame(all_output, columns=['x', 'y', 'batch', 'Celltype']).reset_index()
    # all_output.to_csv('display_oversample_valid.csv')

    coordinates_train = torch.cat([coordinates_train, coordinates_train[nodes_to_duplicate_train]], dim=0)
    coordinates_valid = torch.cat([coordinates_valid, coordinates_valid[nodes_to_duplicate_valid]], dim=0)

    return new_edge_index_train, new_node_idx_train, new_edge_index_valid, new_node_idx_valid, coordinates_train, coordinates_valid, nodes_to_duplicate_train, nodes_to_duplicate_valid


def enhanceGraph(edge_index_train, edge_index_valid, coordinates_train, coordinates_valid, batch_train, batch_valid):
    # 计算每一个点的k近邻的平均距离
    distance_train = compute_edge_weights(coordinates_train, edge_index_train)
    distance_valid = compute_edge_weights(coordinates_valid, edge_index_valid)
    # 计算每一个点的k近邻的平均距离的平均值, 每k个点计算一个平均值，而不是直接计算所有点的平均值
    # count = distance_train[distance_train <= 2].shape[0]
    # count_14142 = distance_train[(distance_train - 1.4142 <= 0.001) & (distance_train - 1.4142 >= -0.001)].shape[0]
    # count_2 = distance_train[distance_train == 2].shape[0]
    # count_14142_small = distance_train[distance_train <= 1.4142].shape[0]
    # test 1
    print(edge_index_train.shape)
    print(edge_index_valid.shape)
    # span_tech
    # edge_index_train = edge_index_train[:, distance_train <= 114.3]
    # edge_index_valid = edge_index_valid[:, distance_valid <= 114.3]
    # fly_14-16h
    # edge_index_train = edge_index_train[:, distance_train <= 20]
    # edge_index_valid = edge_index_valid[:, distance_valid <= 20]
    # dlpfc
    # edge_index_train = edge_index_train[:, distance_train <= 2]
    # edge_index_valid = edge_index_valid[:, distance_valid <= 2]
    # hheart
    # edge_index_train = edge_index_train[:, distance_train <= 1.5]
    # edge_index_valid = edge_index_valid[:, distance_valid <= 1.5]
    # embryo without threshold
    # breast_cancer fail!
    # liver_cancer
    # edge_index_train = edge_index_train[:, distance_train <= 2]
    # edge_index_valid = edge_index_valid[:, distance_valid <= 2]
    # fly_16-18h
    # edge_index_train = edge_index_train[:, distance_train <= 20]
    # edge_index_valid = edge_index_valid[:, distance_valid <= 20]
    # rcc dataset(10x)
    # edge_index_train = edge_index_train[:, distance_train <= 2]
    # edge_index_valid = edge_index_valid[:, distance_valid <= 2]
    # test 2
    # edge_index_train = edge_index_train[:, (distance_train - 1.4142 <= 0.001) & (distance_train - 1.4142 >= -0.001)]
    # edge_index_valid = edge_index_valid[:, (distance_valid - 1.4142 <= 0.001) & (distance_valid - 1.4142 >= -0.001)]
    print('enhance: ', edge_index_train.shape)
    print('enhance: ', edge_index_valid.shape)
    return edge_index_train, edge_index_valid


def mergeGraph(adata, edge_index_train, edge_index_valid):
    # 生成train的数据是，记得生成train和valid两个边集合，节点index分别从0开始
    graph_comm_train = adata.uns['communication_graph']['train_edge_index']
    graph_comm_valid = adata.uns['communication_graph']['valid_edge_index']

    # 转为 tensor 格式，支持 list 或 numpy.ndarray
    if isinstance(graph_comm_train, (list, np.ndarray)):
        graph_comm_train = torch.tensor(graph_comm_train, dtype=torch.long)
    if isinstance(graph_comm_valid, (list, np.ndarray)):
        graph_comm_valid = torch.tensor(graph_comm_valid, dtype=torch.long)

    # 确保都是二维 [2, num_edges]
    if graph_comm_train.shape[0] != 2:
        raise ValueError("Invalid shape for train_edge_index")
    if graph_comm_valid.shape[0] != 2:
        raise ValueError("Invalid shape for valid_edge_index")

    # 合并边
    edge_index_train = torch.cat([edge_index_train, graph_comm_train], dim=1)
    edge_index_valid = torch.cat([edge_index_valid, graph_comm_valid], dim=1)

    return edge_index_train, edge_index_valid


def buildgraph(adata, train_batch, label_name='Celltype', batch_size=64, neighbor=6):
    # all_batch = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    # valid_batch = [4, 13, 18]
    # 获取adata.obs中的'x'和'y'值
    mask = adata.obs['batch'].isin(train_batch)
    x_train = adata.obs.loc[mask, 'x']
    y_train = adata.obs.loc[mask, 'y']
    x_valid = adata.obs.loc[~mask, 'x']
    y_valid = adata.obs.loc[~mask, 'y']
    # 将'x'和'y'值组合成坐标矩阵
    coordinates_train = torch.tensor(list(zip(x_train, y_train)))
    coordinates_valid = torch.tensor(list(zip(x_valid, y_valid)))
    batch_train = adata.obs.loc[mask, 'batch']
    batch_valid = adata.obs.loc[~mask, 'batch']
    
    batch_train = torch.from_numpy(np.array(batch_train).astype(np.int64))
    batch_valid = torch.from_numpy(np.array(batch_valid).astype(np.int64))
    # 使用knn_module函数计算edge_index
    edge_index_train = knn_module(coordinates_train, batch_train, neighbor)
    edge_index_valid = knn_module(coordinates_valid, batch_valid, neighbor)

    # 计算edge_weight
    # edge_weights = compute_edge_weights(coordinates, edge_index)

    edge_index_train, edge_index_valid = enhanceGraph(edge_index_train, edge_index_valid, coordinates_train, coordinates_valid, batch_train, batch_valid)
    # # 构建表达谱相似性knn图
    # count_train = adata.X[mask]
    # count_valid = adata.X[~mask]
    # count_train = torch.tensor(count_train)
    # count_valid = torch.tensor(count_valid)
    # edge_index_train_count_dis = knn_module(count_train, batch_train, neighbor)
    # edge_index_valid_count_dis = knn_module(count_valid, batch_valid, neighbor)

    # # edge_index_train 和 edge_index_train_count_dis concat
    # edge_index_train = torch.cat([edge_index_train, edge_index_train_count_dis], dim=1)
    # edge_index_valid = torch.cat([edge_index_valid, edge_index_valid_count_dis], dim=1)

    node_idx_train = torch.arange(coordinates_train.shape[0])
    node_idx_valid = torch.arange(coordinates_valid.shape[0])

    # 需要debug，在exp_train 和 exp_valid中加入重复的节点, 在label_train 和 label_valid中加入重复的标签
    edge_index_train, node_idx_train, edge_index_valid, node_idx_valid, coordinates_train, coordinates_valid, nodes_to_duplicate_train, nodes_to_duplicate_valid = oversample_graph(adata, edge_index_train, node_idx_train, edge_index_valid, node_idx_valid, coordinates_train, coordinates_valid, train_batch)

    # 合并来自通讯图和图过采样之后的空间图
    print("edge_index_train shape before merge:", edge_index_train.shape)
    print("edge_index_valid shape before merge:", edge_index_valid.shape)
    edge_index_train, edge_index_valid = mergeGraph(adata, edge_index_train, edge_index_valid)
    print("edge_index_train shape after merge:", edge_index_train.shape)
    print("edge_index_valid shape after merge:", edge_index_valid.shape)

    sampler_train = NeighborSampler(
        edge_index_train, node_idx=node_idx_train,
        sizes=[neighbor],  # 指定采样的邻居规模
        # sizes=[-1],  # 指定采样的邻居规模，-1为所有的邻居
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    sampler_valid = NeighborSampler(
        edge_index_valid, node_idx=node_idx_valid,
        sizes=[neighbor],  # 指定采样的邻居规模
        # sizes=[-1],  # 指定采样的邻居规模，-1为所有的邻居
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    exp_gcn = pd.DataFrame(todense(adata), index=np.array(adata.obs_names).tolist(),
                           columns=np.array(adata.var_names).tolist())
    genes = exp_gcn.columns.values
    exp_gcn = np.array(exp_gcn)
    exp_gcn = exp_gcn.astype(np.float32)
    exp_gcn_train = exp_gcn[mask]
    exp_gcn_valid = exp_gcn[~mask]

    # oversample 2 line
    exp_gcn_train = np.concatenate((exp_gcn_train, exp_gcn_train[nodes_to_duplicate_train]), axis=0)
    exp_gcn_valid = np.concatenate((exp_gcn_valid, exp_gcn_valid[nodes_to_duplicate_valid]), axis=0)
    
    exp_gcn_train = torch.from_numpy(np.array(exp_gcn_train).astype(np.float32))
    exp_gcn_valid = torch.from_numpy(np.array(exp_gcn_valid).astype(np.float32))

    label = adata.obs[label_name].astype('str')
    label_encoder = LabelEncoder()
    label = label_encoder.fit_transform(label)
    inverse = label_encoder.inverse_transform(range(0, np.max(label) + 1))
    label = label.astype(np.float32)
    label_train = label[mask]
    label_valid = label[~mask]
    
    # oversample 2 line
    label_train = np.concatenate((label_train, label_train[nodes_to_duplicate_train]), axis=0)
    label_valid = np.concatenate((label_valid, label_valid[nodes_to_duplicate_valid]), axis=0)
    
    label_train = torch.from_numpy(np.array(label_train).astype(np.int64))
    label_valid = torch.from_numpy(np.array(label_valid).astype(np.int64))
    return sampler_train, sampler_valid, exp_gcn_train, exp_gcn_valid, label_train, label_valid, edge_index_train, edge_index_valid, inverse, genes, coordinates_train, coordinates_valid


class GraphLaplacianLoss(nn.Module):
    def __init__(self, lambda_reg):
        super(GraphLaplacianLoss, self).__init__()
        self.lambda_reg = lambda_reg
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, embeddings, pred, labels, laplacian_matrix):
        # 计算交叉熵损失
        ce_loss = self.cross_entropy_loss(pred, labels)

        # 计算拉普拉斯正则化项
        laplacian_reg = torch.trace(torch.matmul(torch.matmul(embeddings.t(), laplacian_matrix), embeddings))

        # 计算总体损失
        total_loss = ce_loss + self.lambda_reg * laplacian_reg

        return total_loss


def compute_laplacian_matrix(edge_index, num_nodes):
    # 创建邻接矩阵
    adj_matrix = torch.zeros((num_nodes, num_nodes))
    adj_matrix[edge_index[0], edge_index[1]] = 1
    adj_matrix[edge_index[1], edge_index[0]] = 1

    # 计算度矩阵
    degree_matrix = torch.diag(torch.sum(adj_matrix, dim=1))
    # 如果度矩阵的对角线元素为0，将其替换为1e-6
    # degree_matrix = torch.where(degree_matrix == 0, torch.tensor(1e-6), degree_matrix)
    diag = torch.diag(degree_matrix)
    degree_matrix = degree_matrix + torch.diag((diag == 0).float() * 1e-6)

    # 计算拉普拉斯矩阵
    # laplacian_matrix = degree_matrix - adj_matrix
    # 另一种计算拉普拉斯矩阵的方法, 但是这种方法在计算拉普拉斯矩阵的逆矩阵时会出现奇异矩阵
    degree_matrix_sqrt_inv = torch.sqrt(torch.inverse(degree_matrix))
    laplacian_matrix = torch.eye(adj_matrix.shape[0]) - torch.matmul(torch.matmul(degree_matrix_sqrt_inv, adj_matrix), degree_matrix_sqrt_inv)

    return laplacian_matrix


def fit_model(adata, gmt_path, project = None, pre_weights='', label_name='Celltype',max_g=300,max_gs=300, mask_ratio = 0.015,n_unannotated = 1,batch_size=4, embed_dim=48,depth=2,num_heads=4,lr=0.001, epochs= 10, lrf=0.01, use_gcn=False, data_batch = [], **kwargs):
    GLOBAL_SEED = 1
    set_seed(GLOBAL_SEED)

    # batch_size, lambda_reg, neighbor
    lambda_reg = 0.000001
    batch_size_gcn = 64
    neighbor = 6

    for key, value in kwargs.items():
        if key == 'batch_size_gcn':
            batch_size_gcn = value
        if key == 'lambda_reg':
            lambda_reg = value
        if key == 'neighbor':
            neighbor = value

    print('batch_size_gcn:', batch_size_gcn)
    print('lambda_reg:', lambda_reg)
    print('neighbor:', neighbor)
    print('num_head:', num_heads)
    print('depth:', depth)
    print('embed_dim:', embed_dim)

    device = 'cuda:0'
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(device)
    today = time.strftime('%Y%m%d',time.localtime(time.time()))
    #train_weights = os.getcwd()+"/weights%s"%today
    project = project or gmt_path.replace('.gmt','')+'_%s'%today
    project_path = os.getcwd()+'/%s'%project
    if os.path.exists(project_path) is False:
        os.makedirs(project_path)

    # train_batch = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17]
    # dlpfc_train_batch = ['151507', '151508', '151509', '151510', '151669', '151670']
    # dlpfc_valid_batch = ['151671', '151672']
    sampler_train, sampler_valid, exp_gcn_train, exp_gcn_valid, label_train, label_valid, edge_index_train, edge_index_valid, inverse, genes, coordinates_train, coordinates_valid = buildgraph(adata, data_batch, label_name, batch_size=batch_size_gcn, neighbor=neighbor)
    tb_writer = SummaryWriter()
    # exp_train, label_train, exp_valid, label_valid, edge_index_train, edge_weights_train, edge_index_valid, edge_weights_valid, inverse,genes = splitDataSet(adata, edge_index, edge_weights, label_name)
    if gmt_path is None:
        mask = np.random.binomial(1,mask_ratio,size=(len(genes), max_gs))
        pathway = list()
        for i in range(max_gs):
            x = 'node %d' % i
            pathway.append(x)
        print('Full connection!')
    else:
        if '.gmt' in gmt_path:
            gmt_path = gmt_path
        else:
            gmt_path = get_gmt(gmt_path)
        reactome_dict = read_gmt(gmt_path, min_g=0, max_g=max_g)
        mask,pathway = create_pathway_mask(feature_list=genes,
                                          dict_pathway=reactome_dict,
                                          add_missing=n_unannotated,
                                          fully_connected=True)
        # 筛选
        pathway = pathway[np.sum(mask,axis=0)>4]
        mask = mask[:,np.sum(mask,axis=0)>4]
        #print(mask.shape)
        pathway = pathway[sorted(np.argsort(np.sum(mask,axis=0))[-min(max_gs,mask.shape[1]):])]
        mask = mask[:,sorted(np.argsort(np.sum(mask,axis=0))[-min(max_gs,mask.shape[1]):])]
        #print(mask.shape)
        print('Mask loaded!')
    np.save(project_path+'/mask.npy',mask)
    pd.DataFrame(pathway).to_csv(project_path+'/pathway.csv') 
    pd.DataFrame(inverse,columns=[label_name]).to_csv(project_path+'/label_dictionary.csv', quoting=None)
    num_classes = np.int64(torch.max(label_train)+1)
    print('use_gcn:{}'.format(use_gcn))
    model = create_model(num_classes=num_classes, num_genes=len(exp_gcn_train[0]),  mask = mask, embed_dim=embed_dim,depth=depth,num_heads=num_heads,has_logits=False, use_gcn=use_gcn).to(device)
    if pre_weights != "":
        assert os.path.exists(pre_weights), "pre_weights file: '{}' not exist.".format(pre_weights)
        preweights_dict = torch.load(pre_weights, map_location=device)
        print(model.load_state_dict(preweights_dict, strict=False))
    #for name, param in model.named_parameters():
    #    if param.requires_grad:
    #        print(name) 
    print('Model builded!')
    pg = [p for p in model.parameters() if p.requires_grad]  
    # optimizer = optim.SGD(pg, lr=lr, momentum=0.9, weight_decay=5E-5) 
    optimizer = optim.Adam(pg, lr=lr, weight_decay=5E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
    loss_function = torch.nn.CrossEntropyLoss()
    # loss_function = GraphLaplacianLoss(lambda_reg=lambda_reg)
    for epoch in range(epochs):
        model.train()
        accu_loss = torch.zeros(1).to(device)
        accu_num = torch.zeros(1).to(device)
        optimizer.zero_grad()
        sample_num = 0
        data_loader = tqdm(sampler_train)
        for step, data in enumerate(data_loader):
            # exp, label, edge_index, edge_weight = data
            _, n_id, adjs  = data
            # Sampled batch
            exp = exp_gcn_train[n_id].to(device)
            edge_index, _, _ = adjs.to(device)
            # Forward pass
            coordinates_input = coordinates_train[n_id].to(device)
            edge_weights = compute_edge_weights(coordinates_input, edge_index)
            sample_num += exp.shape[0]
            # _, pred, _ = model(exp.to(device))
            # 第三个输出为 embeddings 是 attn 模块嵌入的输出，这里不再需要embeddings
            _, pred, _ = model(exp.to(device), edge_index, edge_weights, coordinates_input)
            pred_classes = torch.max(pred, dim=1)[1]
            label_temp = label_train[n_id].to(device)
            accu_num += torch.eq(pred_classes, label_temp).sum().item()
            # embeddings, pred, labels, laplacian_matrix
            # laplacian_matrix = compute_laplacian_matrix(edge_index, num_nodes=exp.shape[0]).to(device)
            # loss = loss_function(embeddings, pred, label_temp, laplacian_matrix)
            loss = loss_function(pred, label_temp)
            loss.backward()
            accu_loss += loss.item()
            data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                                   accu_loss.item() / (step + 1),
                                                                                   accu_num.item() / sample_num)
            # if not torch.isfinite(loss):
            #     print('WARNING: non-finite loss, ending training ', loss)
            #     sys.exit(1)
            optimizer.step()
            optimizer.zero_grad()
            del exp, label_temp, pred, loss, data
            torch.cuda.empty_cache()

        train_loss, train_acc = accu_loss.item() / (step + 1), accu_num.item() / sample_num
        scheduler.step()
        model.eval()
        with torch.no_grad():
            accu_num = torch.zeros(1).to(device)
            accu_loss = torch.zeros(1).to(device)
            sample_num = 0
            data_loader = tqdm(sampler_valid)
            for step, data in enumerate(data_loader):
                # exp, label, edge_index, edge_weight = data
                _, n_id, adjs = data
                # Sampled batch
                exp = exp_gcn_valid[n_id].to(device)
                edge_index, _, _ = adjs.to(device)
                # Forward pass
                coordinate_input = coordinates_valid[n_id].to(device)
                edge_weights = compute_edge_weights(coordinate_input, edge_index)
                # _, pred, _ = model(exp.to(device))
                sample_num += exp.shape[0]
                # 第三个输出为 embeddings 是 attn 模块嵌入的输出，这里不再需要embeddings
                _, pred, _ = model(exp.to(device), edge_index, edge_weights, coordinate_input)
                pred_classes = torch.max(pred, dim=1)[1]
                label_temp = label_valid[n_id].to(device)
                accu_num += torch.eq(pred_classes, label_temp.to(device)).sum().item()
                loss = loss_function(pred, label_temp.to(device)).item()
                # laplacian_matrix = compute_laplacian_matrix(edge_index, num_nodes=exp.shape[0]).to(device)
                # loss = loss_function(embeddings, pred, label_temp, laplacian_matrix)
                accu_loss += loss
                data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                                       accu_loss.item() / (step + 1),
                                                                                       accu_num.item() / sample_num)
                del exp, label_temp, pred, loss, data
                torch.cuda.empty_cache()
            val_loss, val_acc = accu_loss.item() / (step + 1), accu_num.item() / sample_num
        # print("3:{}".format(torch.cuda.memory_allocated(0)))
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        if ((epoch + 1) % 10 == 0):
            if platform.system().lower() == 'windows':
                torch.save(model.state_dict(), project_path+"/model-{}.pth".format(epoch))
            else:
                torch.save(model.state_dict(), "/%s"%project_path+"/model-{}.pth".format(epoch))
        # print("4:{}".format(torch.cuda.memory_allocated(0)))
    print('Training finished!')

#train(adata, gmt_path, pre_weights, batch_size=8, epochs=20)

if __name__ == "__main__":
    getVector()
