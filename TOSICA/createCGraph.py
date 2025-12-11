import scanpy as sc
import pandas as pd
import numpy as np
# from pyscenic.aucell import aucell

from sklearn import metrics

def aucell_manual(expression_matrix, gene_sets, auc_rank_threshold=0.05):
    """
    手动实现 AUCell 算法：评估每个细胞在给定基因集合上的富集分数。
    
    :param expression_matrix: DataFrame，行为细胞，列为基因
    :param gene_sets: dict，如 {'Pathway1': [gene1, gene2, ...], ...}
    :param auc_rank_threshold: 取前多少比例的高表达基因作为排序窗口
    :return: DataFrame，行是细胞，列是路径，值是 AUC 分数
    """
    # 标准化 gene 名
    all_genes = expression_matrix.columns.str.upper().str.replace(' ', '')
    expression_matrix.columns = all_genes
    
    # 结果保存
    auc_scores = pd.DataFrame(index=expression_matrix.index)
    
    # 确定前k个高表达基因
    k = int(len(expression_matrix.columns) * auc_rank_threshold)

    for path_name, genes in gene_sets.items():
        # 处理基因集合
        selected_genes = [g.upper().replace(' ', '') for g in genes if g.upper().replace(' ', '') in all_genes]
        if len(selected_genes) == 0:
            print(f"[Warning] No valid genes found for path {path_name}")
            auc_scores[path_name] = 0
            continue

        scores = []
        for idx, row in expression_matrix.iterrows():
            # 排序当前细胞表达
            ranked_genes = row.sort_values(ascending=False).index.tolist()
            # top_k 基因列表
            top_k_genes = set(ranked_genes[:k])
            # 二值标签向量：1表示基因集中的基因
            labels = [1 if gene in selected_genes else 0 for gene in ranked_genes[:k]]
            scores.append(metrics.auc(range(len(labels)), labels))
        
        auc_scores[path_name] = scores
    
    return auc_scores

def calculatePathEnrichment(adata, path_data):
    """
    计算每个细胞的路径富集分数，并生成新的特征向量。
    :param adata: AnnData 对象，包含细胞表达矩阵
    :param path_data: DataFrame，包含路径及其基因列表
    :return: 更新后的 AnnData 对象
    """
    # 提取基因表达矩阵
    expression_matrix = adata.X
    gene_names = adata.var_names.tolist()
    gene_names = [gene.upper().replace(' ', '') for gene in gene_names]  # 确保基因名是大写的
    expression_matrix = pd.DataFrame(expression_matrix, index=adata.obs_names, columns=gene_names)
    
    # 初始化存储富集分数的字典
    enrichment_scores = {}
    cnt = 0
    # 遍历每个路径
    for path_name, genes in path_data.items():
        # 筛选路径中的基因
        genes = [gene.upper().replace(' ','') for gene in genes]
        selected_genes = [gene for gene in genes if gene in gene_names]
        selected_genes = list(set(selected_genes))  # 去重

        if len(selected_genes) == 0:
            print(f"Warning: No genes found for path '{path_name}' in the expression matrix.")
            cnt += 1
            enrichment_scores[path_name] = pd.Series(0, index=expression_matrix.index)
            continue
        # 计算 AUCell 分数
        scores_df = aucell_manual(expression_matrix, {path_name: selected_genes})
        scores = scores_df.loc[:,path_name]
        enrichment_scores[path_name] = scores
        
    print(f"Warning: {cnt} paths had no genes found in the expression matrix.")
    # # 将富集分数添加到 adata.obs 中
    # for path_name, scores in enrichment_scores.items():
    #     adata.obs[path_name] = scores
    # 将富集分数转为 DataFrame
    path_score_matrix = pd.DataFrame(enrichment_scores, index=expression_matrix.index)
    # 删除全是 0 的路径
    path_score_matrix = path_score_matrix.loc[:, (path_score_matrix != 0).any(axis=0)]
    # 存储到 adata.obsm 中
    adata.obsm['path_scores'] = path_score_matrix
    
    return adata


def getPathScores(adata_name):
    # 获得 path
    path = "/home/lixiangyu/TOSICA-main/TOSICA/resources/output_path4.txt"
    path_data = {}
    with open(path, 'r') as file:
        for line in file:
            values = line.strip().split('\t')
            path_name = values[0]
            path_data[path_name] = values[1:]

    # 计算路径富集分数
    adata = sc.read(adata_name)
    updated_adata = calculatePathEnrichment(adata, path_data)
    # print(updated_adata.obsm['path_scores'])  # 打印更新后的特征向量
    # 输出.obsm['path_scores']中的非零元素个数
    non_zero_elements = np.count_nonzero(adata.obsm['path_scores'])
    print(f"Number of non-zero elements in path_scores: {non_zero_elements}")
    name_ls = adata_name.split('.')
    save_path = name_ls[0] + '_path_scores.' + name_ls[1]
    updated_adata.write(save_path)
    return updated_adata


from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

def filter_top_similarity_edges(edge_sources, edge_targets, similarities, top_percent=0.2):
    """
    保留相似度前 top_percent 的边。
    
    Parameters:
        edge_sources (np.ndarray): 边的起点索引数组
        edge_targets (np.ndarray): 边的终点索引数组
        similarities (np.ndarray): 每条边的相似度
        top_percent (float): 要保留的前百分比（默认20%）
        
    Returns:
        edge_index (np.ndarray): 形状为 [2, num_edges] 的边索引数组
        edge_weights (np.ndarray): 筛选后的边权重数组
    """
    num_edges = len(similarities)
    top_k = int(num_edges * top_percent)

    edge_sources = np.asarray(edge_sources)
    edge_targets = np.asarray(edge_targets)
    similarities = np.asarray(similarities)

    if top_k == 0:
        top_k = 1  # 至少保留一条边

    top_indices = np.argsort(similarities)[-top_k:]

    filtered_edge_index = np.array([
        edge_sources[top_indices],
        edge_targets[top_indices]
    ], dtype=np.int64)
    filtered_weights = similarities[top_indices]

    return filtered_edge_index, filtered_weights


def getCommunicateGraph(adata_name, k=6):
    updated_adata = sc.read(adata_name)
    path_scores = updated_adata.obsm['path_scores']

    # 过滤掉 path_scores 全为 0 的细胞
    valid_mask = ~(path_scores == 0).all(axis=1)
    valid_path_scores = path_scores.loc[valid_mask]

    # 获取 valid_cell_names 及其对应的 AnnData 中的整数索引
    valid_cell_names = valid_path_scores.index.tolist()
    cellname_to_index = {cell_name: i for i, cell_name in enumerate(updated_adata.obs_names)}
    valid_cell_ids = [cellname_to_index[cell_name] for cell_name in valid_cell_names]

    # 使用 KNN 构建图（K=6，使用余弦相似度）
    knn = NearestNeighbors(n_neighbors=k, metric='cosine')
    knn.fit(valid_path_scores.values)
    distances, indices = knn.kneighbors(valid_path_scores.values)

    # 构建图，边用的是 AnnData 中的整数 ID
    edge_sources = []
    edge_targets = []
    adata_distances = []
    for i, neighbors in enumerate(indices):
        source_id = valid_cell_ids[i]
        for j, neighbor_idx in enumerate(neighbors):
            if i == neighbor_idx:
                continue  # 跳过自环
            target_id = valid_cell_ids[neighbor_idx]
            similarity = 1 - distances[i][j]  # 转为余弦相似度
            
            edge_sources.append(source_id)
            edge_targets.append(target_id)
            adata_distances.append(similarity)

    # 构建 edge_index numpy array，形状为 [2, num_edges]
    edge_index = np.array([edge_sources, edge_targets], dtype=np.int64)

    # 只保留distance前20%的边
    edge_index, edge_weights = filter_top_similarity_edges(
        edge_sources, edge_targets, adata_distances, top_percent=0.05
    )

    # 保存为 edge_index 到 uns 中
    updated_adata.uns['communication_graph'] = {'edge_index': edge_index}
    updated_adata.uns['communication_graph']['edge_weights'] = adata_distances

    print(f"Graph edge_index shape: {edge_index.shape} with {edge_index.shape[1]} edges.")

    return updated_adata

import matplotlib.pyplot as plt

def visualCommunicateGraph(adata_name):
    adata = sc.read(adata_name)
    edge_index = adata.uns['communication_graph']['edge_index']
    edge_weights = adata.uns['communication_graph']['edge_weights']

    # Create a graph using NetworkX
    G = nx.Graph()
    
    for i in range(edge_index.shape[1]):
        source = edge_index[0, i]
        target = edge_index[1, i]
        weight = edge_weights[i]
        G.add_edge(source, target, weight=weight)
    
    # Get node positions from adata.obs['x'] and adata.obs['y']
    # 获取 adata.obs_names 的整数索引映射
    cellname_to_index = {cell_name: i for i, cell_name in enumerate(adata.obs_names)}

    # 构建 pos 字典（用整数 ID 为键）
    pos = {
            cellname_to_index[cell_name]: (adata.obs.loc[cell_name, 'x'], adata.obs.loc[cell_name, 'y'])
            for cell_name in adata.obs_names
            }

    # 边权重列表
    weights = [G[u][v]['weight'] for u, v in G.edges()]

    plt.figure(figsize=(12, 10))

    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=150, node_color="skyblue")

    # 绘制边，使用边权重控制颜色
    edge_collection = nx.draw_networkx_edges(
        G, pos, edge_color=weights, edge_cmap=plt.cm.Blues, width=2
    )

    # 添加边权重标签（数字）
    edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=5)

    # 绘制标签
    nx.draw_networkx_labels(G, pos, font_size=6)

    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, 
                               norm=plt.Normalize(vmin=min(weights), vmax=max(weights)))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label("Edge Weight (Cosine Similarity)")

    # plt.figure(figsize=(10, 8))
    # nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue", font_size=10)
    plt.title("Communication Graph")
    save_path = adata_name.replace('.h5ad', '.png')
    plt.savefig(save_path, dpi=300)
    plt.show()


def fliterScores(adata_name):
    adata = sc.read(adata_name)
    # 获取 path_scores
    path_scores = adata.obsm['path_scores']
    print(path_scores.shape)
    # 删除在所有细胞中全为 0 的路径
    non_zero_paths = path_scores.loc[:, (path_scores != 0).any(axis=0)]
    # 更新 adata.obsm['path_scores']
    adata.obsm['path_scores'] = non_zero_paths
    print(non_zero_paths.shape)
    print(non_zero_paths)
    # 输出non_zero_paths的非零元素个数
    non_zero_elements = np.count_nonzero(non_zero_paths)
    print(f"Number of non-zero elements in filtered path_scores: {non_zero_elements}")
    # 输出最大值，最小值和众数
    max_value = non_zero_paths.max().max()
    min_value = non_zero_paths.min().min()
    mode_value = non_zero_paths.mode().iloc[0].max()
    # 输出众数的数量
    mode_count = (non_zero_paths == mode_value).sum().sum()
    print(f"Mode value count: {mode_count}")
    print(f"Max value: {max_value}, Min value: {min_value}, Mode value: {mode_value}")
    # 保存更新后的 adata
    adata.write(adata_name)
    


def CreateTrainData(train_batch, valid_batch, file_pre, train_adata_name, k=6):
    # graph_communicate_train = adata.uns['communication_graph']['train_edge_index']
    # graph_communicate_valid = adata.uns['communication_graph']['valid_edge_index']
    communicate_graph_all = {'train_edge_index': [[], []], 'valid_edge_index': [[], []]}
    node_offset  = 0
    for batch in train_batch:
        adata_name = file_pre + batch + '_path_scores_graph_k_' + str(k) + '.h5ad'
        adata_batch = sc.read(adata_name)
        edge_index_batch = adata_batch.uns['communication_graph']['edge_index']
        edge_weights_batch = adata_batch.uns['communication_graph']['edge_weights']
        # edge_index整体加sum
        edge_index_batch[0, :] += node_offset 
        edge_index_batch[1, :] += node_offset 
        node_offset  += adata_batch.shape[0]
        # 将边和权重添加到 communicate_graph_all，直接将元素加进去，不是形成一个二维列表
        communicate_graph_all['train_edge_index'][0].extend(edge_index_batch[0].tolist())
        communicate_graph_all['train_edge_index'][1].extend(edge_index_batch[1].tolist())
    
    node_offset  = 0
    for batch in valid_batch:
        adata_name = file_pre + batch + '_path_scores_graph_k_' + str(k) + '.h5ad'
        adata_batch = sc.read(adata_name)
        edge_index_batch = adata_batch.uns['communication_graph']['edge_index']
        edge_weights_batch = adata_batch.uns['communication_graph']['edge_weights']
        # edge_index整体加sum
        edge_index_batch[0, :] += node_offset 
        edge_index_batch[1, :] += node_offset 
        node_offset  += adata_batch.shape[0]
        # 将边和权重添加到 communicate_graph_all，直接将元素加进去，不是形成一个二维列表
        communicate_graph_all['valid_edge_index'][0].extend(edge_index_batch[0].tolist())
        communicate_graph_all['valid_edge_index'][1].extend(edge_index_batch[1].tolist())

    train_adata = sc.read(train_adata_name)
    train_adata.uns['communication_graph'] = communicate_graph_all
    train_adata_name_ls = train_adata_name.split('.')
    new_adata_name = train_adata_name_ls[0] + f'_train_valid_CGraph_k_{k}.' + train_adata_name_ls[1]
    train_adata.write(new_adata_name)


if __name__ == "__main__":
    # span query
    # batch_ls = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
    # batch_ls = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
    # train_batch = ['1', '2', '3', '5', '6', '7', '8', '10', '11', '12', '14']
    # valid_batch = ['4', '9', '13']
    # k = 1
    # # file_pre = '/home/lixiangyu/TOSICA-main/data/Fly3d/fly_E16-18h_hvg_3000_query_binning_8_with_0_'
    # file_pre = '/home/lixiangyu/TOSICA-main/data/Fly3d/fly_E14-16h_hvg_3000_span_query_binning_8_with_0_'

    # dlpfc
    # batch_ls = ['151507', '151508', '151509', '151510', 
    #                 '151669', '151670', '151671', '151672', 
    #                 '151673', '151674', '151675', '151676']
    # file_pre = '/home/lixiangyu/TOSICA-main/data/dlpfc/dlpfc_test_hvg_3000_with_ground_truth_and_batch_x_y_without_na_'
    # k = 2

    # embyro
    # batch_ls = ['1', '2', '3']
    # batch_ls = ['2', '3']
    # file_pre = '/home/lixiangyu/TOSICA-main/data/embryo/embryo_query_351_gene_'
    # k = 1

    # liver cancer
    # batch_ls = ['1', '2', '3', 
    #             '4', '5', '6', 
    #             '7', '8', '9', '10', 
    #             '11', '12', '13', 
    #             '14', '15', '16', 
    #             '17', '18', '19', '20', 
    #             '21']
    # file_pre = '/home/lixiangyu/TOSICA-main/data/liver_cancer/liver_cancer_query_'
    # k = 2

    # mob
    batch_ls = ['3', '5']
    file_pre = '/home/lixiangyu/TOSICA-main/data/MOB/mob_query_hvg_5000_3_5_common_'
    k = 2


    for batch in batch_ls:
        # /home/lixiangyu/TOSICA-main/data/Fly3d/fly_E16-18h_hvg_3000_query_binning_8_with_0_1.h5ad
        adata_name = file_pre + f'{batch}.h5ad'
        # 1. 计算路径富集分数并保存path_scores的adata
        # getPathScores(adata_name)


        # 2. 获得 communicate graph
        name_ls = adata_name.split('.')
        adata_name_path_scores = name_ls[0] + '_path_scores.' + name_ls[1]
        adata_name_communicate_graph = name_ls[0] + '_path_scores' + '_graph_k_' + str(k) + '.' + name_ls[1]

        # updated_adata = getCommunicateGraph(adata_name_path_scores, k=k)
        # updated_adata.write(adata_name_communicate_graph)


        # 3. 可视化 communicate graph
        # visualCommunicateGraph(adata_name_communicate_graph)
        # break


    # 4. 根据train_batch生成train data
    # span query
    # train_adata_name = '/home/lixiangyu/TOSICA-main/data/Fly3d/fly_E16-18h_train_hvg_3000_binning_8_with_0.h5ad'
    # CreateTrainData(train_batch, valid_batch, file_pre, train_adata_name, k=k)

    # dlpfc
    # train_adata_file_pre = '/home/lixiangyu/TOSICA-main/data/dlpfc/dlpfc_train_and_valid_hvg_3000_with_ground_truth_and_batch_x_y_without_na_'
    # train_adata_name_ls = ['07_69_73.h5ad', '08_70_74.h5ad',
    #                        '09_71_75.h5ad', '10_72_76.h5ad']
    # dlpfc_all_batch = ['151507', '151508', '151509', '151510', 
    #                    '151669', '151670', '151671', '151672', 
    #                    '151673', '151674', '151675', '151676']
    # train_batch_ls = [['151508', '151509', '151670', '151671', '151674', '151675'],
    #                 ['151509', '151510', '151671', '151672', '151675', '151676'],
    #                 ['151507', '151510', '151669', '151672', '151673', '151676'],
    #                 ['151507', '151508', '151669', '151670', '151673', '151674'],]
    # valid_batch_ls = [['151510', '151672', '151676'],
    #                  ['151507', '151669', '151673'],
    #                  ['151508', '151670', '151674'],
    #                  ['151509', '151671', '151675']]
    # for i, train_adata_name in enumerate(train_adata_name_ls):
    #     train_batch = train_batch_ls[i]
    #     valid_batch = valid_batch_ls[i]
    #     train_adata_name = train_adata_file_pre + train_adata_name
    #     CreateTrainData(train_batch, valid_batch, file_pre, train_adata_name, k=k)

    # embryo
    # train_adata_file_pre = '/home/lixiangyu/TOSICA-main/data/embryo/embryo_train_'
    # train_adata_name_ls = ['1_2.h5ad', '1_3.h5ad', '2_3.h5ad']
    # train_batch_ls = [['3'], ['2'], ['1']]
    # valid_batch_ls = [['3'], ['2'], ['1']]
    # for i, train_adata_name in enumerate(train_adata_name_ls):
    #     train_batch = train_batch_ls[i]
    #     valid_batch = valid_batch_ls[i]
    #     train_adata_name = train_adata_file_pre + train_adata_name
    #     CreateTrainData(train_batch, valid_batch, file_pre, train_adata_name, k=k)

    # liver cancer
    # train_adata_file_pre = '/home/lixiangyu/TOSICA-main/data/liver_cancer/liver_cancer_train_'
    # train_adata_name_ls = ['chc1_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21.h5ad', 
    #                         'hcc2_1_2_3_21.h5ad']
    # train_batch_ls = [['1', '2'],
    #                   ['7', '8', '9']]
    # valid_batch_ls = [['3'],
    #                   ['10']]
    # for i, train_adata_name in enumerate(train_adata_name_ls):
    #     train_batch = train_batch_ls[i]
    #     valid_batch = valid_batch_ls[i]
    #     train_adata_name = train_adata_file_pre + train_adata_name
    #     CreateTrainData(train_batch, valid_batch, file_pre, train_adata_name, k=k)

    # mob
    train_adata_file_pre = '/home/lixiangyu/TOSICA-main/data/MOB/mob_train_hvg_5000_3_5_common_'
    train_adata_name_ls = ['3.h5ad', '5.h5ad']
    train_batch_ls = [['5'],
                      ['3']]
    valid_batch_ls = [['5'],
                      ['3']]                
    for i, train_adata_name in enumerate(train_adata_name_ls):
        train_batch = train_batch_ls[i]
        valid_batch = valid_batch_ls[i]
        train_adata_name = train_adata_file_pre + train_adata_name
        CreateTrainData(train_batch, valid_batch, file_pre, train_adata_name, k=k)

    # 测试代码
    # adata = sc.read('/home/lixiangyu/TOSICA-main/data/Fly3d/fly_E16-18h_hvg_3000_query_binning_8_with_0_13_path_scores_graph_k_3.h5ad')
    # print(adata)
    # print(adata.uns['communication_graph']['edge_index'].shape)
    # print(adata.obs['batch'].value_counts())
    # 输出权重中等于1的数量
    # edge_weights = adata.uns['communication_graph']['edge_weights']
    # equal_one_count = np.sum(np.array(edge_weights) == 1)
    # print(f"Number of edges with weight equal to 1: {equal_one_count}")
    # print(adata.uns['communication_graph']['edge_weights'].max())

    print('done!!')