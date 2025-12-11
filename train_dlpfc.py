import anndata
import pandas as pd
import matplotlib.pyplot as plt
import TOSICA
import scanpy as sc
import numpy as np
import warnings
warnings.filterwarnings ("ignore")

import math

import torch
# print(torch.__version__)
# print(torch.cuda.get_device_capability(device=None),  torch.cuda.get_device_name(device=None))


# 合并 DLPFC
def mergeDLPFC():
    # dlpfc_list = ['507','508','509','510','669','670','671','672','673','674','675','676']
    dlpfc_list = ['673','674','675','676']
    # dlpfc_list = ['507','508','509','510','669','670','671','672']
    counts = []
    clusters = []
    coordinates = []
    adata = sc.read(r"D:\tosica\dlpfc\151" + dlpfc_list[0] + ".h5ad")
    gene_sybmol = adata.var_names
    obs_names_list = []

    for i in range(len(dlpfc_list)):
        adata = sc.read(r"D:\tosica\dlpfc\151" + dlpfc_list[i] + ".h5ad")
        # print(adata)
        counts.append(pd.DataFrame(adata.X.todense()))
        clusters.append(pd.DataFrame(adata.obs['clusters']))
        coordinates.append(pd.DataFrame(pd.concat([adata.obs['array_row'], adata.obs['array_col']], axis=1)))
        obs_names_list.append(pd.DataFrame(adata.obs_names))
        print(i)

    counts_all = pd.concat(counts, axis=0)
    clusters_all = pd.concat(clusters, axis=0)
    coordinates_all = pd.concat(coordinates, axis=0)
    obs_names_all = pd.concat(obs_names_list, axis=0)

    counts_all.index = obs_names_all
    counts_all.columns = gene_sybmol

    adata_all = anndata.AnnData(X=counts_all)
    adata_all.obs['Celltype'] = clusters_all['clusters'].values
    adata_all.var['Gene Symbol'] = gene_sybmol

    adata_all.write('dlpfc_test.h5ad')


def calculate_hvg():
    adata_train = sc.read('dlpfc_train.h5ad')

    sc.pp.highly_variable_genes(adata_train, n_top_genes=3000, inplace=True)
    # adata_train = adata_train[:, adata_train.var.highly_variable]
    adata_test = sc.read('dlpfc_test.h5ad')
    adata_test = adata_test[:, adata_train.var.highly_variable]
    # adata_train.write('dlpfc_train_hvg_3000.h5ad')
    adata_test.write('dlpfc_test_hvg_3000.h5ad')


def checkAdata(adata):
    print(adata)
    print(adata.X)
    print(adata.obs['Celltype'])
    print(adata.obs['Celltype'].value_counts())
    print(adata.obs['Prediction'])
    print(adata.obs['Prediction'].value_counts())


def addPosition():
    # dlpfc_list = ['507','508','509','510','669','670','671','672']
    dlpfc_list = ['673', '674', '675', '676']
    positions_x = []
    positions_y = []
    for i in range(len(dlpfc_list)):
        adata = sc.read(r"D:\tosica\dlpfc\151" + dlpfc_list[i] + "_10xvisium_processed.h5ad")
        # print(adata)
        positions_x.append(pd.DataFrame(adata.obs['array_col']))
        positions_y.append(pd.DataFrame(adata.obs['array_row']))
        print(i)

    positions_x_all = pd.concat(positions_x, axis=0)
    positions_y_all = pd.concat(positions_y, axis=0)
    positions_all = pd.concat([positions_x_all, positions_y_all], axis=1)
    print(positions_all)
    return positions_all



def patch_position_encode(coordinate, count):
    position_encode = np.zeros((count.shape[0], count.shape[1]))
    feature_dim = 3000
    for i in range(int(feature_dim / 2)):
        if i % 2 == 0:
            position_encode[:, i] = coordinate.iloc[:, 0].apply(lambda x: math.sin(x / (math.exp((4 * i) / feature_dim * math.log(10000.0)))))
        else:
            position_encode[:, i] = coordinate.iloc[:, 0].apply(lambda x: math.cos(x / (math.exp((4 * i) / feature_dim * math.log(10000.0)))))

    for j in range(int(feature_dim / 2), feature_dim):
        if j % 2 == 0:
            position_encode[:, j] = coordinate.iloc[:, 1].apply(lambda x: math.sin(x / (math.exp((4 * (j - feature_dim / 2)) / feature_dim * math.log(10000.0)))))
        else:
            position_encode[:, j] = coordinate.iloc[:, 1].apply(lambda x: math.cos(x / (math.exp((4 * (j - feature_dim / 2)) / feature_dim * math.log(10000.0)))))
    # print(position_encode)
    # print(position_encode.shape)
    return position_encode


def calculatePredictionPercentEachBatch(path):
    adata = sc.read(path)
    print(adata)
    label_counts = adata.obs['Celltype'].value_counts()
    print(label_counts)
    for label, count in label_counts.items():
        print(label)
        print((adata.obs.loc[adata.obs['Celltype'] == label, 'match'].sum() / count) * 100)


def saveResult(query_adata, model_weight_path, project, batch_index, pre_batch_dlpfc):
    # pre_batch_dlpfc = ['151510', '151672', '151676']
    new_adata = TOSICA.pre(query_adata, model_weight_path=model_weight_path, project=project, use_gcn=True, data_batch=pre_batch_dlpfc)
    new_adata.obs['match'] = (new_adata.obs['Prediction'].astype(str) == new_adata.obs['Celltype'].astype(str)).astype(int)
    print(new_adata.obs['match'].sum() / len(new_adata.obs['Prediction']))
    new_adata.uns['prediction percent'] = new_adata.obs['match'].sum() / len(new_adata.obs['Prediction'])
    # checkAdata(new_adata)
    new_adata.obs.drop(columns=['x'], inplace=True)
    new_adata.obs.drop(columns=['y'], inplace=True)
    new_adata.obs['x'] = query_adata.obs['x'].values
    new_adata.obs['y'] = query_adata.obs['y'].values
    new_adata.write('predict_result/' + project + '_' + pre_batch_dlpfc[batch_index] + '_result.h5ad')
    print('save successfully!')
    print(project + '_' + pre_batch_dlpfc[batch_index] + '_result.h5ad')
    new_adata = sc.read('predict_result/' + project + '_' + pre_batch_dlpfc[batch_index] + "_result.h5ad")
    print(new_adata)
    print(new_adata.uns['prediction percent'])


def displayResult(project, batch_index, pre_batch_dlpfc):
    # pre_batch_dlpfc = ['151510', '151672', '151676']
    # 读取adata
    adata = sc.read('predict_result/' + project + '_' + pre_batch_dlpfc[batch_index] + '_result.h5ad')
    # 提取x和y坐标，以及Celltype和Prediction字段
    x = adata.obs['x']
    y = adata.obs['y']
    y = -y
    celltypes = adata.obs['Celltype']
    predictions = adata.obs['Prediction']
    # 创建一个颜色字典，为每个Celltype和Prediction分配不同的颜色
    unique_values = pd.concat([celltypes, predictions]).unique()
    # colors = {value: color for value, color in zip(unique_values, plt.cm.rainbow(np.linspace(0, 1, len(unique_values))))}
    colors = {'Layer 1': '#1f77b4', 'Layer 3': '#2c8c2c', 'Layer 5': '#9467bd',
               'WM': '#e377c2', 'Layer 2': '#ff7f0e', 'Layer 4':'#d62728', 'Layer 6': '#8c564b'}
    # 创建两个子图，一个用于Celltype，另一个用于Prediction
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # 画Celltype的散点图
    for celltype in celltypes.unique():
        celltype_coordinates = pd.concat([x[celltypes == celltype], y[celltypes == celltype]], axis=1)
        axes[0].scatter(celltype_coordinates['x'], celltype_coordinates['y'], color=colors[celltype], label=celltype, s=85)
    axes[0].legend()
    axes[0].set_title('Celltype')

    # 画Prediction的散点图
    for prediction in predictions.unique():
        prediction_coordinates = pd.concat([x[predictions == prediction], y[predictions == prediction]], axis=1)
        axes[1].scatter(prediction_coordinates['x'], prediction_coordinates['y'], color=colors[prediction], label=prediction, s=85)
    axes[1].legend()
    axes[1].set_title('Prediction')

    plt.tight_layout()
    plt.show()
    # 将plt保存为png
    plt.savefig('predict_result/' + project + '_' + pre_batch_dlpfc[batch_index] + '_result.png')


def saveGroundTruth():
    dlpfc_list = ['507','508','509','510','669','670','671','672','673','674','675','676']
    for i in range(len(dlpfc_list)):
        adata = sc.read(r"D:\tosica\dlpfc\DLPFC_151" + dlpfc_list[i] + ".h5ad")
        print(adata)
        ground_truth = pd.DataFrame(adata.obs['ground.truth'])
        ground_truth['batch'] = "DLPFC_151" + dlpfc_list[i]
        # ground_truth.to_csv(r'D:\tosica\dlpfc\151' + dlpfc_list[i] + '_ground_truth.csv')
        print(ground_truth['ground.truth'].value_counts())


def configGroundTruth():
    adata = sc.read('dlpfc_train_hvg_3000.h5ad')
    print(adata)
    dlpfc_list = ['673','674','675','676']
    dlpfc_list = ['507','508','509','510','669','670','671','672']
    groundTruth_train = []
    for i in range(len(dlpfc_list)):
        ground_truth = pd.read_csv(r'D:\tosica\dlpfc\151' + dlpfc_list[i] + '_ground_truth.csv')
        groundTruth_train.append(ground_truth)
        print(ground_truth['ground.truth'].value_counts())
    groundTruth_train = pd.concat(groundTruth_train, axis=0)
    adata.obs['Celltype'] = groundTruth_train['ground.truth'].values
    adata.obs['batch'] = groundTruth_train['batch'].values
    # adata.write('dlpfc_train_hvg_3000_with_ground_truth_and_batch.h5ad')
    print(adata)
    print(adata.obs['Celltype'].value_counts())


def train_and_pre(project, epoch, dlpfc_train_batch, pre_batch_dlpfc, adata_name, iftrain=True, batch_size_gcn=64, neighbor=6):
    adata = sc.read(adata_name)
    # adata = sc.read('data/dlpfc_train_hvg_3000_with_ground_truth_and_batch_x_y_without_na.h5ad')
    # adata.obs['batch'].to_csv('batch.csv')
    # pre_weights = "dlpfc_k_6_bs_64_qk_newloss_laplacians_2_lambda_reg_0.00001_embedding_attn_epoch20/model-19.pth"
    batch_size_gcn = batch_size_gcn
    lambda_reg = 0.000001
    neighbor = neighbor
    if iftrain:
        TOSICA.train(adata, gmt_path='human_gobp', label_name='Celltype', epochs=epoch, project=project, use_gcn=True, data_batch=dlpfc_train_batch, batch_size_gcn=batch_size_gcn, lambda_reg=lambda_reg, neighbor=neighbor)
    # TOSICA.train(adata, gmt_path=None, label_name='Celltype', epochs=epoch, project=project, pre_weights=pre_weights, use_gcn=True, data_batch=dlpfc_train_batch)
    model_weight_path = './' + project + '/model-' + str(epoch - 1) + '.pth'
    # /home/lixiangyu/TOSICA-main/data/dlpfc/dlpfc_test_hvg_3000_with_ground_truth_and_batch_x_y_without_na_151507_path_scores_graph_k_2.h5ad
    query_data_path = 'data/dlpfc/dlpfc_test_hvg_3000_with_ground_truth_and_batch_x_y_without_na_'
    for i in range(len(pre_batch_dlpfc)):
        query_adata = sc.read(query_data_path + pre_batch_dlpfc[i] + '_path_scores_graph_k_2.h5ad')
        # query_adata = sc.read('data/dlpfc_test_hvg_3000_with_ground_truth_and_batch_x_y_' + pre_batch_dlpfc[i] + '_without_na.h5ad')
        print(pre_batch_dlpfc[i])
        print(query_adata)
        saveResult(query_adata, model_weight_path, project=project, batch_index=i, pre_batch_dlpfc=pre_batch_dlpfc)


def train_without_pre():
    batch_size_gcn = 64
    # 0.00001, 0.000001
    lambda_reg = [0.00000001, 0.00000005, 0.0000002, 0.0000003, 0.0000005, 0.0000007, 0.0000009]
    adata = sc.read('dlpfc_train_hvg_3000_with_ground_truth_and_batch_x_y_without_na_coverdesign.h5ad')
    for i in range(len(lambda_reg)):
        epoch = 20
        project = 'dlpfc_k_6_bs_64_qk_newloss_laplacians_2_lambda_reg_'+ str(lambda_reg[i]) +'_embedding_attn_epoch' + str(epoch)
        adata = sc.read('dlpfc_train_hvg_3000_with_ground_truth_and_batch_x_y_without_na_coverdesign.h5ad')
        dlpfc_train_batch = ['151507', '151508', '151669', '151670', '151673', '151674']
        TOSICA.train(adata, gmt_path=None, label_name='Celltype', epochs=20, project=project, use_gcn=True, data_batch=dlpfc_train_batch, batch_size_gcn=batch_size_gcn, lambda_reg=lambda_reg[i])


def train_all_dlpfc():
    # 除了需要配置参数，还需要重新整理数据
    dlpfc_all_batch = ['151507', '151508', '151509', '151510', 
                       '151669', '151670', '151671', '151672', 
                       '151673', '151674', '151675', '151676']
    dlpfc_train_batch = [['151507', '151508', '151669', '151670', '151673', '151674'],
                         ['151508', '151509', '151670', '151671', '151674', '151675'],
                         ['151509', '151510', '151671', '151672', '151675', '151676'],
                         ['151507', '151510', '151669', '151672', '151673', '151676']]
    pre_batch_dlpfc = [['151510', '151672', '151676'],
                       ['151507', '151669', '151673'],
                       ['151508', '151670', '151674'],
                       ['151509', '151671', '151675']]
    adata_name_query_batch_list = ['10_72_76', '07_69_73', '08_70_74', '09_71_75']
    for i in range(len(dlpfc_train_batch)):
        try:
            epoch = 20
            project = 'dlpfc_k_6_bs_128_qk_threshold_2_' + adata_name_query_batch_list[i] + 'epoch' + str(epoch)
            adata_name = 'data/dlpfc/dlpfc_train_and_valid_hvg_3000_with_ground_truth_and_batch_x_y_without_na_' + adata_name_query_batch_list[i] + '.h5ad'
            pre_epoch = 20
            print(project)
            train_and_pre(project, pre_epoch, dlpfc_train_batch[i], pre_batch_dlpfc[i], adata_name)
            for j in range(len(pre_batch_dlpfc[i])): 
                displayResult(project, j, pre_batch_dlpfc=pre_batch_dlpfc[i])
                calculatePredictionPercentEachBatch('predict_result/' + project +'_' + pre_batch_dlpfc[i][j] + '_result.h5ad')
        except torch.cuda.OutOfMemoryError:
            print(f"CUDA out of memory encountered for iteration {i}. Skipping this iteration.")
            continue

def train_one_group_dlpfc():
    epoch = 20
    project = 'dlpfc_k_6_bs_128_qk_threshold_2_noise_std1_epoch' + str(epoch)
    adata_name = 'data/dlpfc/dlpfc_train_hvg_3000_with_ground_truth_and_batch_x_y_without_na_10_72_76.h5ad'
    dlpfc_train_batch = ['151507', '151508', '151669', '151670', '151673', '151674']
    # dlpfc_valid_batch = ['151509', '151671', '151675']
    pre_batch_dlpfc = ['151510', '151672', '151676']
    # pre_batch_dlpfc = ['151673', '151674', '151675', '151676']
    # 更换pre_epoch参数，利用不同的epoch参数训练模型
    pre_epoch = 20
    train_and_pre(project, pre_epoch, dlpfc_train_batch, pre_batch_dlpfc, adata_name)
    for i in range(len(pre_batch_dlpfc)): 
        displayResult(project, i, pre_batch_dlpfc=pre_batch_dlpfc)
        calculatePredictionPercentEachBatch('predict_result/' + project +'_' + pre_batch_dlpfc[i] + '_result.h5ad')
    # train_without_pre()


def train_all_dlpfc_50_percent():
    # 除了需要配置参数，还需要重新整理数据
    dlpfc_all_batch = ['151507', '151508', '151509', '151510', 
                       '151669', '151670', '151671', '151672', 
                       '151673', '151674', '151675', '151676']
    dlpfc_train_and_valid_batch = [['151507', '151508', '151669', '151670', '151673', '151674'],
                                   ['151507', '151509', '151669', '151671', '151673', '151675'],
                                   ['151507', '151510', '151669', '151672', '151673', '151676'],
                                   ['151508', '151509', '151670', '151671', '151674', '151675'],
                                   ['151508', '151510', '151670', '151672', '151674', '151676'],
                                   ['151509', '151510', '151671', '151672', '151675', '151676'],]
    dlpfc_train_batch = [['151507', '151669', '151673'],
                         ['151507', '151669', '151673'],
                         ['151507', '151669', '151673'],
                         ['151508', '151670', '151674'],
                         ['151508', '151670', '151674'],
                         ['151509', '151671', '151675']]
    pre_batch_dlpfc = [['151509', '151510', '151671', '151672', '151675', '151676'],
                       ['151508', '151510', '151670', '151672', '151674', '151676'],
                       ['151508', '151509', '151670', '151671', '151674', '151675'],
                       ['151507', '151510', '151669', '151672', '151673', '151676'],
                       ['151507', '151509', '151669', '151671', '151673', '151675'],
                       ['151507', '151508', '151669', '151670', '151673', '151674']]
    adata_name_query_batch_list = ['09_10_71_72_75_76', '08_10_70_72_74_76', '08_09_70_71_74_75', 
                                   '07_10_69_72_73_76', '07_09_69_71_73_75', '07_08_69_70_73_74']
    for i in range(len(dlpfc_train_batch)):
        try:
            epoch = 20
            project = 'dlpfc_k_6_bs_64_qk_threshold_2_debug_percent_50_' + adata_name_query_batch_list[i] + 'epoch' + str(epoch)
            adata_name = 'data/dlpfc/dlpfc_train_and_valid_hvg_3000_with_ground_truth_and_batch_x_y_without_na_' + adata_name_query_batch_list[i] + '.h5ad'
            pre_epoch = 20
            print(project)
            train_and_pre(project, pre_epoch, dlpfc_train_batch[i], pre_batch_dlpfc[i], adata_name)
            for j in range(len(pre_batch_dlpfc[i])): 
                displayResult(project, j, pre_batch_dlpfc=pre_batch_dlpfc[i])
                calculatePredictionPercentEachBatch('predict_result/' + project +'_' + pre_batch_dlpfc[i][j] + '_result.h5ad')
        except torch.cuda.OutOfMemoryError:
            print(f"CUDA out of memory encountered for iteration {i}. Skipping this iteration.")
            continue


def train_all_dlpfc_25_percent():
    bs = 16
    k = 6
    # 除了需要配置参数，还需要重新整理数据
    dlpfc_all_batch = ['151507', '151508', '151509', '151510', 
                       '151669', '151670', '151671', '151672', 
                       '151673', '151674', '151675', '151676']
    dlpfc_train_and_valid_batch = [['151507', '151669', '151673'], 
                                   ['151508', '151670', '151674'],
                                   ['151509', '151671', '151675'],
                                   ['151510', '151672', '151676']]
    dlpfc_train_batch = [['151507', '151669'],
                         ['151508', '151670'],
                         ['151509', '151671'],
                         ['151510', '151672']]
    pre_batch_dlpfc = [['151508', '151509', '151510', '151670', '151671', '151672', '151674', '151675', '151676'],
                       ['151507', '151509', '151510', '151669', '151671', '151672', '151673', '151675', '151676'],
                       ['151507', '151508', '151510', '151669', '151670', '151672', '151673', '151674', '151676'],
                       ['151507', '151508', '151509', '151669', '151670', '151671', '151673', '151674', '151675']]
    adata_name_query_batch_list = ['08_09_10_70_71_72_74_75_76', '07_09_10_69_71_72_73_75_76', 
                                   '07_08_10_69_70_72_73_74_76', '07_08_09_69_70_71_73_74_75']
    batch_size_gcn = bs
    neighbor = k
    for i in range(len(dlpfc_train_batch)):
        try:
            epoch = 20
            project = 'dlpfc_k_' + str(neighbor) + '_bs_' + str(batch_size_gcn) + '_qk_threshold_2_percent_25_' + adata_name_query_batch_list[i] + 'epoch' + str(epoch)
            adata_name = 'data/dlpfc/dlpfc_train_and_valid_hvg_3000_with_ground_truth_and_batch_x_y_without_na_' + adata_name_query_batch_list[i] + '.h5ad'
            pre_epoch = 20
            print(project)
            train_and_pre(project, pre_epoch, dlpfc_train_batch[i], pre_batch_dlpfc[i], adata_name, iftrain=True, batch_size_gcn=batch_size_gcn, neighbor=neighbor)
            for j in range(len(pre_batch_dlpfc[i])): 
                displayResult(project, j, pre_batch_dlpfc=pre_batch_dlpfc[i])
                calculatePredictionPercentEachBatch('predict_result/' + project +'_' + pre_batch_dlpfc[i][j] + '_result.h5ad')
        except torch.cuda.OutOfMemoryError:
            print(f"CUDA out of memory encountered for iteration {i}. Skipping this iteration.")
            continue


def outputStatistic():
    pre_batch_dlpfc = [['151510', '151672', '151676'],
                       ['151507', '151669', '151673'],
                       ['151508', '151670', '151674'],
                       ['151509', '151671', '151675']]
    adata_name_query_batch_list = ['10_72_76', '07_69_73', '08_70_74', '09_71_75']
    percent_list = []
    # batch_ls = [16, 32, 48, 64, 96, 128]
    batch_ls = [64]
    index_list = []
    for bs in batch_ls:
        for i in range(len(pre_batch_dlpfc)):
            epoch = 20
            project = f'dlpfc_k_6_bs_{bs}_qk_threshold_2_CGraph_k_2_human_gobp_' + adata_name_query_batch_list[i] + 'epoch' + str(epoch)
            for j in range(len(pre_batch_dlpfc[i])): 
                adata = sc.read('predict_result/' + project +'_' + pre_batch_dlpfc[i][j] + '_result.h5ad')
                # print(adata)
                print(adata.uns['prediction percent'])
                percent_list.append(adata.uns['prediction percent'])
    
        for i in range(len(pre_batch_dlpfc)):
            for j in range(len(pre_batch_dlpfc[i])):
                index_list.append(pre_batch_dlpfc[i][j])
    
    columns_list = ['percent']
    df = pd.DataFrame(percent_list, index=index_list, columns=columns_list)
    print(df)
    df.to_csv(project + '_CGraph_human_gobp_all_group_percent.csv')
        


def train_all_dlpfc_span_donor():
    bs = 16
    k = 6
    # 除了需要配置参数，还需要重新整理数据
    dlpfc_all_batch = ['151507', '151508', '151509', '151510', 
                       '151669', '151670', '151671', '151672', 
                       '151673', '151674', '151675', '151676']
    dlpfc_train_and_valid_batch = [['151507', '151508', '151509', '151510','151673', '151674', '151675', '151676']]
    dlpfc_train_batch = [['151507', '151508', '151509', '151673', '151674', '151675']]
    pre_batch_dlpfc = [['151669', '151670', '151671', '151672']]
    adata_name_query_batch_list = ['donor_1_3_to_2']
    bs_ls = [16, 32, 64, 128]
    for bs in bs_ls:
        batch_size_gcn = bs
        neighbor = k
        for i in range(len(dlpfc_train_batch)):
            try:
                epoch = 20
                project = 'dlpfc_k_' + str(neighbor) + '_bs_' + str(batch_size_gcn) + '_qk_threshold_2_percent_66_' + adata_name_query_batch_list[i] + 'epoch' + str(epoch)
                adata_name = 'data/dlpfc/dlpfc_train_and_valid_hvg_3000_with_ground_truth_and_batch_x_y_without_na_' + adata_name_query_batch_list[i] + '.h5ad'
                pre_epoch = 20
                print(project)
                train_and_pre(project, pre_epoch, dlpfc_train_batch[i], pre_batch_dlpfc[i], adata_name, iftrain=True, batch_size_gcn=batch_size_gcn, neighbor=neighbor)
                for j in range(len(pre_batch_dlpfc[i])): 
                    displayResult(project, j, pre_batch_dlpfc=pre_batch_dlpfc[i])
                    calculatePredictionPercentEachBatch('predict_result/' + project +'_' + pre_batch_dlpfc[i][j] + '_result.h5ad')
            except torch.cuda.OutOfMemoryError:
                print(f"CUDA out of memory encountered for iteration {i}. Skipping this iteration.")
                continue


def train_percent_75_dlpfc_CGraph():
    # 除了需要配置参数，还需要重新整理数据
    dlpfc_all_batch = ['151507', '151508', '151509', '151510', 
                       '151669', '151670', '151671', '151672', 
                       '151673', '151674', '151675', '151676']
    dlpfc_train_batch = [['151507', '151508', '151669', '151670', '151673', '151674'],
                         ['151508', '151509', '151670', '151671', '151674', '151675'],
                         ['151509', '151510', '151671', '151672', '151675', '151676'],
                         ['151507', '151510', '151669', '151672', '151673', '151676']]
    pre_batch_dlpfc = [['151510', '151672', '151676'],
                       ['151507', '151669', '151673'],
                       ['151508', '151670', '151674'],
                       ['151509', '151671', '151675']]
    adata_name_query_batch_list = ['10_72_76', '07_69_73', '08_70_74', '09_71_75']
    # batch_ls = [16, 32, 48, 64, 96, 128]
    batch_ls = [64]
    for bs in batch_ls:
        for i in range(len(dlpfc_train_batch)):
            try:
                epoch = 20
                project = f'dlpfc_k_6_bs_{bs}_qk_threshold_2_CGraph_k_2_human_gobp_' + adata_name_query_batch_list[i] + 'epoch' + str(epoch)
                adata_name = 'data/dlpfc/dlpfc_train_and_valid_hvg_3000_with_ground_truth_and_batch_x_y_without_na_' + adata_name_query_batch_list[i] + '_train_valid_CGraph_k_2.h5ad'
                pre_epoch = 20
                print(project)
                train_and_pre(project, pre_epoch, dlpfc_train_batch[i], pre_batch_dlpfc[i], adata_name, batch_size_gcn=bs)
                for j in range(len(pre_batch_dlpfc[i])): 
                    displayResult(project, j, pre_batch_dlpfc=pre_batch_dlpfc[i])
                    calculatePredictionPercentEachBatch('predict_result/' + project +'_' + pre_batch_dlpfc[i][j] + '_result.h5ad')
            except torch.cuda.OutOfMemoryError:
                print(f"CUDA out of memory encountered for iteration {i}. Skipping this iteration.")
                continue


if __name__ == '__main__':
    # train_all_dlpfc()
    # train_all_dlpfc_50_percent()
    # train_all_dlpfc_25_percent()
    # outputStatistic()
    # train_all_dlpfc_span_donor()
    train_percent_75_dlpfc_CGraph()
    print('done')