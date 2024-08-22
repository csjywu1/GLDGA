import numpy as np
import torch
import pickle
from model import SGCLDGA
from utils import metrics, scipy_sparse_mat_to_torch_sparse_tensor, mm_auc
# import pandas as pd
# from parser import args
from tqdm import tqdm
import time
import torch.utils.data as data
from utils import TrnData
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc as sklearn_auc, \
    precision_recall_curve
from sklearn import metrics as mt
import torch.nn.functional as F
from utils import get_syn_sim
from sklearn.decomposition import PCA



import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--decay', default=0.99, type=float, help='learning rate')
    parser.add_argument('--batch', default=256, type=int, help='batch size')  # 256
    parser.add_argument('--inter_batch', default=4096, type=int, help='batch size')  # 4096
    parser.add_argument('--note', default=None, type=str, help='note')
    parser.add_argument('--lambda1', default=1e-3, type=float, help='weight of cl loss')  # 0.05
    parser.add_argument('--epoch', default=50, type=int, help='number of epochs')
    parser.add_argument('--d', default=1024, type=int, help='embedding size')#512 0.886
    parser.add_argument('--q', default=5, type=int, help='rank')
    parser.add_argument('--gnn_layer', default=4, type=int, help='number of gnn layers')  # 2
    parser.add_argument('--data', default='yelp', type=str, help='name of dataset')
    parser.add_argument('--dropout', default=0.1, type=float, help='rate for edge dropout')
    parser.add_argument('--temp', default=0.8, type=float, help='temperature in cl loss')  # 0.8
    parser.add_argument('--lambda2', default=1e-5, type=float, help='l2 reg weight')  # 1e-5
    parser.add_argument('--cuda', default='0', type=str, help='the gpu to use')
    return parser.parse_args()
args = parse_args()

device = 'cuda:' + args.cuda
# hyperparameters
d = args.d
l = args.gnn_layer
temp = args.temp
batch_gene = args.batch
inter_batch = args.inter_batch
epoch_no = args.epoch
max_samp = 100#40
lambda_1 = args.lambda1

lambda_2 = args.lambda2
dropout = args.dropout
lr = args.lr
decay = args.decay
svd_q = args.q


def train_val():
    print("lambda_1:",lambda_1,"lambda_2:",lambda_2)
    # load data 加载数据
    f = open('data_val/train_mat', 'rb')
    train = pickle.load(f)
    train = train.astype(np.float64) # 3227,10690
    print(type(train))
    train_csr = (train != 0).astype(np.float32)
    print(type(train_csr))
    f = open('data_val/test_mat', 'rb')
    test = pickle.load(f)
    test = test.astype(np.float64)
    f = open('data_val/val_mat', 'rb')
    val = pickle.load(f)
    val = val.astype(np.float64)
    print('Data loaded.')
    print(type(train))
    print('gene_num:', train.shape[0], 'drug_num:', train.shape[1], 'lambda_1:', lambda_1, 'lambda_2:', lambda_2,
          'temp:',
          temp, 'q:', svd_q)

    # 将训练数据从稀疏矩阵转换为密集矩阵。
    # 然后，处理训练数据的标签，构建一个列表 train_labels，其中每个元素是该行（基因）的非零列（药物）的索引。
    # train_mat = train.todense().A

    train_labels = [[] for i in range(train.shape[0])]
    for i in range(len(train.data)):
        row = train.row[i]
        col = train.col[i]
        train_labels[row].append(col) #3227个标签
    print('Test data processed.')

    # 定义一个变量 epoch_gene，表示每个epoch中基因的数量，取训练数据行数和30000之间的较小值。
    # epoch_gene = min(train.shape[0], 30000)

    # normalizing the adj matrix
    # 对训练数据进行行和列归一化处理，以标准化邻接矩阵的值。
    rowD = np.array(train.sum(1)).squeeze()
    colD = np.array(train.sum(0)).squeeze()

    for i in range(len(train.data)):
        train.data[i] = train.data[i] / pow(rowD[train.row[i]] * colD[train.col[i]], 0.5)

    # construct data loader
    # 构建数据加载器
    train = train.tocoo()
    # matrix = train.todense().A
    train_data = TrnData(train) #3227,10690
    train_loader = data.DataLoader(train_data, batch_size=args.inter_batch, shuffle=True, num_workers=0) #假设 train 是一个稀疏矩阵，包含 37514 个非零元素

    # 归一化邻接矩阵并转换为稀疏张量
    # 将训练数据转换为稀疏张量，并移动到GPU上。
    adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(train)
    adj_norm = adj_norm.coalesce().cuda(torch.device(device)) #3227,10690 #假设 train 是一个稀疏矩阵，包含 37514 个非零元素
    print('Adj matrix normalized.')

    # perform svd reconstruction
    # 对邻接矩阵执行SVD分解，得到左奇异向量、奇异值和右奇异向量。计算并保存 u_mul_s 和 v_mul_s。
    adj = scipy_sparse_mat_to_torch_sparse_tensor(train).coalesce().cuda(torch.device(device)) #3227,10690
    print('Performing SVD...')
    # svd_u, s, svd_v = torch.linalg.svd(adj)
    # 假设 adj 矩阵的大小为 3227 x 10690，且 q 为 5，那么：
    # svd_u 的大小为 3227 x 5。
    # s 的长度为 5。
    # svd_v 的大小为 10690 x 5。
    svd_u, s, svd_v = torch.svd_lowrank(adj, q=svd_q) # 3227, 10690 q=5
    u_mul_s = svd_u @ (torch.diag(s))
    v_mul_s = svd_v @ (torch.diag(s))
    del s
    print('SVD done.')

    # process test set
    # 类似处理训练数据标签的方法，处理测试和验证数据的标签。
    test_labels = [[] for i in range(test.shape[0])] #3227,10690
    for i in range(len(test.data)):
        row = test.row[i]
        col = test.col[i]
        test_labels[row].append(col)
    print('Test data processed.')

    val_labels = [[] for i in range(val.shape[0])] #3227.10690
    for i in range(len(val.data)):
        row = val.row[i]
        col = val.col[i]
        val_labels[row].append(col)
    print('Test data processed.')

    # 初始化损失列表，并构建模型和优化器。
    loss_list = []
    loss_r_list = []
    loss_s_list = []

    model = SGCLDGA(adj_norm.shape[0], adj_norm.shape[1], d, u_mul_s, v_mul_s, svd_u.T, svd_v.T, train_csr, adj_norm,
                     l,
                     temp, lambda_1, lambda_2, dropout, batch_gene, device)
    # model.load_state_dict(torch.load('saved_model.pt'))
    model.cuda(torch.device(device))
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0, lr=lr)
    # optimizer.load_state_dict(torch.load('saved_optim.pt'))

    # 迭代训练过程，对于每个epoch，重置损失累积值，并进行负采样。
    # 在每个批次中，获取基因ID、正样本和负样本，并将其转换为GPU张量。
    # 然后计算损失并进行反向传播和优化。最后，计算并打印每个epoch的平均损失。
    # current_lr = lr


    for epoch in range(epoch_no):
        epoch_loss = 0
        epoch_loss_r = 0
        epoch_loss_s = 0
        # 分别包含 37514 个元素，表示每个非零元素的行和列索引。 正样本
        # self.negs 也包含 3227 个元素，用于存储每个正样本对应的负样本索引。

        # 遍历每一行（基因）索引 self.rows。
        # 对于每个基因，随机生成一个列索引 i_neg。
        # 检查生成的 (u, i_neg) 是否在 self.dokmat 中（即是否是一个非零元素）。如果不是，则将其作为负样本。
        # 将生成的负样本索引存储在 self.negs 中对应的位置。
        train_loader.dataset.neg_sampling()
        # 如果数据集的总大小为 37514，而批量大小为 4096，那么批次数量计算如下： 批次数量 = 10
        for i, batch in enumerate(tqdm(train_loader)): # 这个循环遍历每个批次的数据。tqdm 是一个快速、可扩展的进度条库，用于显示训练进度。
            geneids, pos, neg = batch # 4096， 4096， 4096
            geneids = geneids.long().cuda(torch.device(device)) # 基因ID，是当前批次中所有样本的行索引。
            pos = pos.long().cuda(torch.device(device)) # 正样本药物ID，是与基因ID对应的正样本（实际存在的基因-药物对）。
            neg = neg.long().cuda(torch.device(device))
            iids = torch.concat([pos, neg], dim=0)

            # feed
            optimizer.zero_grad()
            loss, loss_r, loss_s = model(geneids, iids, pos, neg)
            loss.backward()
            optimizer.step()
            # print('batch',batch)
            epoch_loss += loss.cpu().item()
            epoch_loss_r += loss_r.cpu().item()
            epoch_loss_s += loss_s.cpu().item()

            # torch.cuda.empty_cache()
            # print(i, len(train_loader), end='\r')

        batch_no = len(train_loader)
        epoch_loss = epoch_loss / batch_no
        epoch_loss_r = epoch_loss_r / batch_no
        epoch_loss_s = epoch_loss_s / batch_no
        loss_list.append(epoch_loss)
        loss_r_list.append(epoch_loss_r)
        loss_s_list.append(epoch_loss_s)
        print('Epoch:', epoch, 'Loss:', epoch_loss, 'Loss_r:', epoch_loss_r, 'Loss_s:', epoch_loss_s)

train_val()




