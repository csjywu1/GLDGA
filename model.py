import torch
import torch.nn as nn
from utils import sparse_dropout, spmm
import torch.nn.functional as F
import numpy as np

class SGCLDGA(nn.Module):
    def __init__(self, n_u, n_i, d, u_mul_s, v_mul_s, ut, vt, train_csr, adj_norm, l, temp, lambda_1, lambda_2, dropout,
                 batch_user, device):
        super(SGCLDGA, self).__init__()
        # 初始化 nn层
        self.E_g_GNN_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_u, d)))  # 3227， 1024
        self.E_d_GNN_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_i, d)))  # 10690， 1024
        self.mlp1 = nn.Sequential(nn.Linear(639, 1280),
                                  nn.ReLU(),
                                  nn.Linear(1280, 1280),
                                  nn.ReLU(),
                                  nn.Linear(1280, d))

        self.mlp2 = nn.Sequential(nn.Linear(120, 256),
                                  nn.ReLU(),
                                  nn.Linear(256, 256),
                                  nn.ReLU(),
                                  nn.Linear(256, d))

        self.train_csr = train_csr
        self.adj_norm = adj_norm
        self.l = l
        self.E_g_GNN_list = [None] * (l + 1)
        self.E_d_GNN_list = [None] * (l + 1)
        self.E_g_GNN_list[0] = self.E_g_GNN_0
        self.E_d_GNN_list[0] = self.E_d_GNN_0

        # self.new_adj_norm = self.remove_edges(self.adj_norm, 0.01)

        # 初始化 SVD 部分
        self.E_g_SVD_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_u, d)))  # 3227， 1024
        self.E_d_SVD_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_i, d)))  # 10690， 1024
        self.E_g_SVD_list = [None] * (l + 1)
        self.E_d_SVD_list = [None] * (l + 1)
        self.E_g_SVD_list[0] = self.E_g_SVD_0
        self.E_d_SVD_list[0] = self.E_d_SVD_0
        self.temp = temp
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.dropout = dropout
        self.act = nn.LeakyReLU(0.5)
        self.batch_user = batch_user

        self.E_g_GNN = None
        self.E_d_GNN = None

        self.u_mul_s = u_mul_s
        self.v_mul_s = v_mul_s
        self.ut = ut
        self.vt = vt

        self.device = device

        # wu
        # 初始化代码簇 (codebook)
        self.codebook_g = nn.Parameter(nn.init.xavier_uniform_(torch.empty(128, d)))  # 512 个 codebook 向量，每个向量维度为 d
        self.codebook_d = nn.Parameter(nn.init.xavier_uniform_(torch.empty(128, d)))  # 512 个 codebook 向量，每个向量维度为 d

        # 定义用于计算查询和键的MLP网络
        self.query_mlp = nn.Sequential(
            nn.Linear(d, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )

        self.key_mlp = nn.Sequential(
            nn.Linear(d, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )

        # 定义解码器
        self.decoder_g = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d)
        )

        self.decoder_d = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d)
        )

        # 定义扩散模型
        self.diffusion_model = SimpleDiffusionModel(input_dim=d, hidden_dim=256)

        self.concat_mlp = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 1)  # 输出维度为1，用于计算分数
        )

        self.concat_mlp1 = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 1)  # 输出维度为1，用于计算分数
        )

        self.query_mlp1 = nn.Sequential(
            nn.Linear(d, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )

        self.key_mlp1 = nn.Sequential(
            nn.Linear(d, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )

    def forward(self, uids, iids, pos, neg, test=False):
        if test == True:  # testing phase
            u_emb = self.E_g_GNN[uids]
            i_emb = self.E_d_GNN[iids]

            # 合并并通过 MLP
            u_i_concat = torch.cat([u_emb, i_emb], dim=-1).unsqueeze(0)
            pred = self.concat_mlp(u_i_concat).squeeze(-1)

            return pred, self.E_g_GNN, self.E_d_GNN
        else:  # training phase
            # 这个循环遍历从第1层到第self.l层的所有层。
            for layer in range(1, self.l + 1):
                # GNN propagation
                self.E_g_GNN_list[layer] =torch .spmm(sparse_dropout(self.adj_norm, self.dropout), self.E_d_GNN_list[layer - 1])
                self.E_d_GNN_list[layer] = torch.spmm(sparse_dropout(self.adj_norm, self.dropout).transpose(0, 1),
                                                      self.E_g_GNN_list[layer - 1])

                # SVD propagation using new_adj_norm
                # SVD propagation using new_adj_norm
                self.E_g_SVD_list[layer] = torch.spmm(sparse_dropout(self.adj_norm, self.dropout),
                                                      self.E_d_SVD_list[layer - 1])
                self.E_d_SVD_list[layer] = torch.spmm(sparse_dropout(self.adj_norm, self.dropout).transpose(0, 1),
                                                      self.E_g_SVD_list[layer - 1])

            self.E_g_SVD = sum(self.E_g_SVD_list)  # SVD
            self.E_d_SVD = sum(self.E_d_SVD_list)

            # aggregate across layers # 将不同元素进行聚合
            self.E_g_GNN = sum(self.E_g_GNN_list)  # 原始
            self.E_d_GNN = sum(self.E_d_GNN_list)

            # WU:一致性loss
            # 计算查询和键
            # 计算 Q 和 K
            # 计算 Q 和 K
            Q = self.query_mlp(self.E_g_SVD)
            K = self.key_mlp(self.E_g_GNN)

            # 打印 Q 和 K 的初始统计信息
            # print("Initial Q mean:", Q.mean().item(), "std:", Q.std().item())
            # print("Initial K mean:", K.mean().item(), "std:", K.std().item())

            # 正则化 Q 和 K
            Q = F.normalize(Q, p=2, dim=-1)
            K = F.normalize(K, p=2, dim=-1)

            # 打印正则化后的 Q 和 K 的统计信息
            # print("Normalized Q mean:", Q.mean().item(), "std:", Q.std().item())
            # print("Normalized K mean:", K.mean().item(), "std:", K.std().item())

            # 添加偏置项
            bias = 0.5
            d_k = Q.size(-1)
            A = (Q @ K.T)

            # 确保对角线一致性
            A_diag = A.diag()
            I_diag = torch.ones_like(A_diag)
            consistency_loss = F.mse_loss(A_diag, I_diag)

            #wu
            # 计算 E_d_SVD 和 E_g_SVD 的 A_svd
            # Q_svd = self.query_mlp1(self.E_g_SVD)
            # K_svd = self.key_mlp1(self.E_d_SVD)
            #
            # Q_svd = F.normalize(Q_svd, p=2, dim=-1)
            # K_svd = F.normalize(K_svd, p=2, dim=-1)
            #
            # A_svd = Q_svd @ K_svd.T
            #
            # # 计算 E_d_GNN 和 E_g_GNN 的 A_gnn
            # Q_gnn = self.query_mlp1(self.E_g_GNN)
            # K_gnn = self.key_mlp1(self.E_d_GNN)
            #
            # Q_gnn = F.normalize(Q_gnn, p=2, dim=-1)
            # K_gnn = F.normalize(K_gnn, p=2, dim=-1)
            #
            # A_gnn = Q_gnn @ K_gnn.T
            #
            # # 计算 A_svd 和 A_gnn 的余弦相似度
            #
            # # 计算 A_svd 和 A_gnn 的余弦相似度
            # cosine_sim = self.cosine_similarity(A_svd, A_gnn)
            #
            # # 反转余弦相似度以使其最小化，范围为 [0, 2]
            # cosine_loss = 1 - cosine_sim.mean()
            # # 总的一致性损失
            # consistency_loss += cosine_loss

            #交叉熵损失
            u_emb = self.E_g_GNN[uids]  # 4096,1024 #基因
            pos_emb = self.E_d_GNN[pos]  # 4096,1024 # 药物
            neg_emb = self.E_d_GNN[neg]  # 4096,1024 # 药物
            pos_scores = (u_emb * pos_emb).sum(-1)
            neg_scores = (u_emb * neg_emb).sum(-1)

            # 正样本和负样本的得分
            u_pos_concat = torch.cat([u_emb, pos_emb], dim=-1)
            u_neg_concat = torch.cat([u_emb, neg_emb], dim=-1)

            # 通过MLP进行降维
            pos_scores = self.concat_mlp(u_pos_concat).squeeze(-1)
            neg_scores = self.concat_mlp(u_neg_concat).squeeze(-1)

            # 创建标签
            pos_labels = torch.ones_like(pos_scores)
            neg_labels = torch.zeros_like(neg_scores)

            # 计算交叉熵损失
            loss_pos = F.binary_cross_entropy_with_logits(pos_scores, pos_labels)
            loss_neg = F.binary_cross_entropy_with_logits(neg_scores, neg_labels)

            # 总的交叉熵损失
            loss_r = loss_pos + loss_neg
            loss_r = -(pos_scores - neg_scores).sigmoid().log().mean() + loss_r  # 差值越大越好

            # cl loss
            # loss_s 是一种对比损失（contrastive loss），用于增强模型的对比学习效果，确保嵌入空间的结构合理。
            G_u_norm = self.E_g_SVD  # 3227，1024
            E_g_norm = self.E_g_GNN  # 3227, 1024
            G_i_norm = self.E_d_SVD  # 10690， 1024
            E_d_norm = self.E_d_GNN  # 10690， 1024

            neg_score = torch.log(torch.exp(G_u_norm[uids] @ E_g_norm.T / self.temp).sum(1) + 1e-8).mean()  # SVD
            neg_score += torch.log(torch.exp(G_i_norm[iids] @ E_d_norm.T / self.temp).sum(1) + 1e-8).mean()  # SVD
            # 对应位置的相乘
            pos_score = (torch.clamp((G_u_norm[uids] * E_g_norm[uids]).sum(1) / self.temp, -5.0, 5.0)).mean() + \
                        (
                            torch.clamp((G_i_norm[iids] * E_d_norm[iids]).sum(1) / self.temp, -5.0, 5.0)).mean()
            loss_s = -pos_score + neg_score  # pos_score 越大越好，neg_score越小越好

            #交叉熵损失
            # bpr loss 改为交叉熵损失
            # u_emb = self.E_g_GNN[uids]  # 4096,1024 #基因
            # pos_emb = self.E_d_GNN[pos]  # 4096,1024 # 药物
            #
            # u_emb1 = self.E_g_SVD[uids]  # 4096,1024 #基因
            # pos_emb1 = self.E_d_SVD[pos]  # 4096,1024 # 药物
            #
            # # 合并 u_emb 和 pos_emb
            # u_pos_concat = torch.cat([u_emb, pos_emb], dim=-1)
            # u_pos_concat1 = torch.cat([u_emb1, pos_emb1], dim=-1)
            #
            # # 通过MLP进行降维
            # u_pos_concat = self.concat_mlp(u_pos_concat).squeeze(-1)
            # u_pos_concat1 = self.concat_mlp1(u_pos_concat1).squeeze(-1)
            #
            # # 计算正样本的得分，并应用sigmoid函数确保输出在0到1之间
            # pos_scores = torch.sigmoid(u_pos_concat)
            # pos_scores1 = torch.sigmoid(u_pos_concat1)
            # #
            # # # 将分数转化为概率分布
            # pos_scores_log = F.log_softmax(pos_scores, dim=-1)  # 计算log_softmax
            # pos_scores1_prob = F.softmax(pos_scores1, dim=-1)  # 计算softmax
            # #
            # # # 计算KL散度
            # kl_loss = F.kl_div(pos_scores_log, pos_scores1_prob, reduction='batchmean')
            #
            # # 总的交叉熵损失
            # loss_r = kl_loss

            # reg loss
            # loss_reg 是正则化损失，用于避免模型过拟合，通过对所有参数施加 L2 正则化来实现。
            loss_reg = 0
            for param in self.parameters():
                loss_reg += param.norm(2).square()
            loss_reg *= self.lambda_2

            neg_emb = self.E_d_GNN[neg]
            ## Diffusion loss
            # diffusion_loss, pos_scores3 = self.compute_diffusion_loss(u_emb, pos_emb, neg_emb)
            # kl_loss = F.kl_div(pos_scores, pos_scores1, reduction='batchmean')
            # kl_loss = F.kl_div(pos_scores, pos_scores3, reduction='batchmean')
            # loss_pos3 = F.binary_cross_entropy_with_logits(pos_scores3, pos_labels)

            # total loss
            loss = loss_reg + self.lambda_1 * loss_s + consistency_loss + loss_r   #loss_r +
                   # + loss_reg + consistency_loss + diffusion_loss   # 将 VQ-VAE 损失添加到总损失中
            # loss_r = torch.tensor(0)

            return loss, loss_r, self.lambda_1 * loss_s

    def cosine_similarity(self, A, B):
        A_norm = F.normalize(A, p=2, dim=-1)
        B_norm = F.normalize(B, p=2, dim=-1)
        return torch.sum(A_norm * B_norm, dim=-1)

    # 向量量化
    def quantize(self, x, codebook):
        x_expanded = x.unsqueeze(1)  # 扩展维度以进行广播
        distances = torch.sum((x_expanded - codebook) ** 2, dim=2)  # 计算距离
        indices = torch.argmin(distances, dim=1)  # 找到最近的代码簇向量
        quantized = codebook[indices]  # 获取量化后的向量
        return quantized, indices

    def compute_diffusion_loss(self, u_emb, pos_emb, neg_emb):
        t = torch.rand(u_emb.size(0), 1).to(u_emb.device)  # 随机时间步

        noise = torch.randn_like(u_emb)
        noisy_u_emb = u_emb + noise * torch.sqrt(t)
        denoised_u_emb = self.diffusion_model(noisy_u_emb, t)
        diffusion_loss = F.mse_loss(denoised_u_emb, u_emb)

        noise = torch.randn_like(pos_emb)
        noisy_pos_emb = pos_emb + noise * torch.sqrt(t)
        denoised_pos_emb = self.diffusion_model(noisy_pos_emb, t)
        diffusion_loss += F.mse_loss(denoised_pos_emb, pos_emb)

        # 计算去噪后的BPR损失
        pos_scores = (denoised_u_emb * denoised_pos_emb).sum(-1)
        # neg_scores = (denoised_u_emb * denoised_neg_emb).sum(-1)
        # bpr_loss = -(pos_scores - neg_scores).sigmoid().log().mean()

        # 返回总的扩散损失，包括去噪后的BPR损失
        return diffusion_loss, pos_scores  #diffusion_loss +

    def remove_edges(self, adj_norm, percentage):
        # 将稀疏矩阵转换为 COO 格式
        indices = adj_norm._indices()
        values = adj_norm._values()
        size = adj_norm.size()

        # 计算边的总数
        num_edges = values.size(0)

        # 随机选择 percentage 的边进行删除
        num_edges_to_remove = int(num_edges * percentage)
        perm = torch.randperm(num_edges)
        edges_to_remove = perm[:num_edges_to_remove]

        # 删除选定的边
        mask = torch.ones(num_edges, dtype=torch.bool, device=adj_norm.device)
        mask[edges_to_remove] = False

        new_indices = indices[:, mask]
        new_values = values[mask]

        # 构建新的稀疏矩阵
        new_adj_norm = torch.sparse_coo_tensor(new_indices, new_values, size)

        # 输出新矩阵的边数
        new_num_edges = new_values.size(0)
        print(f"New number of edges: {new_num_edges}")

        return new_adj_norm




class SimpleDiffusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimpleDiffusionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim + 1, hidden_dim)  # +1 是因为我们将时间步 t 与输入拼接在一起
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, t):
        t_expanded = t.expand_as(x[:, :1])  # 扩展时间步 t 的维度以与输入 x 匹配
        x = torch.cat([x, t_expanded], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
