import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import scipy.io as io
import random
import sklearn.utils
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from sklearn import preprocessing
import math


# 修改编码器部分，解码器用nri，编码器是参考resnet18络，还是用高斯分布，多任务优化

class MinNormSolver:
    MAX_ITER = 200
    #STOP_CRIT = 1e-5
    STOP_CRIT = 0.0001

    def _min_norm_element_from2(v1v1, v1v2, v2v2):
        """
        Analytical solution for min_{c} |cx_1 + (1-c)x_2|_2^2
        d is the distance (objective) optimzed
        v1v1 = <x1,x1>
        v1v2 = <x1,x2>
        v2v2 = <x2,x2>
        """
        if v1v2 >= v1v1:
            # Case: Fig 1, third column
            gamma = 0.999
            cost = v1v1
            return gamma, cost
        if v1v2 >= v2v2:
            # Case: Fig 1, first column
            gamma = 0.001
            cost = v2v2
            return gamma, cost
        # Case: Fig 1, second column
        gamma = -1.0 * ((v1v2 - v2v2) / (v1v1 + v2v2 - 2 * v1v2))
        cost = v2v2 + gamma * (v1v2 - v2v2)
        return gamma, cost

    def _min_norm_2d(vecs, dps):
        """
        Find the minimum norm solution as combination of two points
        This is correct only in 2D
        ie. min_c |\sum c_i x_i|_2^2 st. \sum c_i = 1 , 1 >= c_1 >= 0 for all i, c_i + c_j = 1.0 for some i, j
        """
        dmin = 1e8
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                if (i, j) not in dps:
                    dps[(i, j)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i, j)] += torch.mul(vecs[i][k], vecs[j][k]).sum().data.cpu()
                    dps[(j, i)] = dps[(i, j)]
                if (i, i) not in dps:
                    dps[(i, i)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i, i)] += torch.mul(vecs[i][k], vecs[i][k]).sum().data.cpu()
                if (j, j) not in dps:
                    dps[(j, j)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(j, j)] += torch.mul(vecs[j][k], vecs[j][k]).sum().data.cpu()
                c, d = MinNormSolver._min_norm_element_from2(dps[(i, i)], dps[(i, j)], dps[(j, j)])
                if d < dmin:
                    dmin = d
                    sol = [(i, j), c, d]
        return sol, dps

    def _projection2simplex(y):
        """
        Given y, it solves argmin_z |y-z|_2 st \sum z = 1 , 1 >= z_i >= 0 for all i
        """
        m = len(y)
        sorted_y = np.flip(np.sort(y), axis=0)
        tmpsum = 0.0
        tmax_f = (np.sum(y) - 1.0) / m
        for i in range(m - 1):
            tmpsum += sorted_y[i]
            tmax = (tmpsum - 1) / (i + 1.0)
            if tmax > sorted_y[i + 1]:
                tmax_f = tmax
                break
        return np.maximum(y - tmax_f, np.zeros(y.shape))

    def _next_point(cur_val, grad, n):
        proj_grad = grad - (np.sum(grad) / n)
        tm1 = -1.0 * cur_val[proj_grad < 0] / proj_grad[proj_grad < 0]
        tm2 = (1.0 - cur_val[proj_grad > 0]) / (proj_grad[proj_grad > 0])

        skippers = np.sum(tm1 < 1e-7) + np.sum(tm2 < 1e-7)
        t = 1
        if len(tm1[tm1 > 1e-7]) > 0:
            t = np.min(tm1[tm1 > 1e-7])
        if len(tm2[tm2 > 1e-7]) > 0:
            t = min(t, np.min(tm2[tm2 > 1e-7]))

        next_point = proj_grad * t + cur_val
        next_point = MinNormSolver._projection2simplex(next_point)
        return next_point

    def find_min_norm_element(vecs):
        """
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution, and then run the projected gradient descent until convergence
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)

        n = len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec, init_sol[2]

        iter_count = 0

        grad_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                grad_mat[i, j] = dps[(i, j)]

        while iter_count < MinNormSolver.MAX_ITER:
            grad_dir = -1.0 * np.dot(grad_mat, sol_vec)
            new_point = MinNormSolver._next_point(sol_vec, grad_dir, n)
            # Re-compute the inner products for line search
            v1v1 = 0.0
            v1v2 = 0.0
            v2v2 = 0.0
            for i in range(n):
                for j in range(n):
                    v1v1 += sol_vec[i] * sol_vec[j] * dps[(i, j)]
                    v1v2 += sol_vec[i] * new_point[j] * dps[(i, j)]
                    v2v2 += new_point[i] * new_point[j] * dps[(i, j)]
            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc * sol_vec + (1 - nc) * new_point
            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec

    def find_min_norm_element_FW(vecs):
        """
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution, and then run the Frank Wolfe until convergence
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)

        n = len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec, init_sol[2]

        iter_count = 0

        grad_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                grad_mat[i, j] = dps[(i, j)]

        while iter_count < MinNormSolver.MAX_ITER:
            t_iter = np.argmin(np.dot(grad_mat, sol_vec))

            v1v1 = np.dot(sol_vec, np.dot(grad_mat, sol_vec))
            v1v2 = np.dot(sol_vec, grad_mat[:, t_iter])
            v2v2 = grad_mat[t_iter, t_iter]

            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc * sol_vec
            new_sol_vec[t_iter] += 1 - nc

            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec


def gradient_normalizers(grads, losses, normalization_type):
    gn = {}
    if normalization_type == 'l2':
        for t in grads:
            gn[t] = np.sqrt(np.sum([gr.pow(2).sum().data.cpu() for gr in grads[t]]))
    elif normalization_type == 'loss':
        for t in grads:
            gn[t] = losses[t]
    elif normalization_type == 'loss+':
        for t in grads:
            gn[t] = losses[t] * np.sqrt(np.sum([gr.pow(2).sum().data.cpu() for gr in grads[t]]))
    elif normalization_type == 'none':
        for t in grads:
            gn[t] = 1.0
    else:
        print('ERROR: Invalid Normalization Type')
    return gn


def my_softmax(input, axis=1):
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input, dim=0)
    return soft_max_1d.transpose(axis, 0)


class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        # self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = F.relu(self.fc1(inputs))
        # x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.relu(self.fc2(x))
        return self.batch_norm(x)
        # return x


class CNN(nn.Module):
    def __init__(self, n_in, n_hid, n_out, channels):
        super(CNN, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2))
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 2))
        self.pool5 = nn.MaxPool2d(kernel_size=(1, 2))

        # self.conv1 = nn.Conv1d(n_in, n_hid, kernel_size=7, stride=1, padding=6)
        self.conv1 = nn.Conv2d(n_in, n_hid, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv11 = nn.Conv2d(n_hid, n_hid, kernel_size=(1, 3), stride=1, padding=(0, 1))
        # self.conv12 = nn.Conv2d(n_hid, n_hid, kernel_size=(1, 3), stride=1, padding=(0, 1))
        # self.conv13 = nn.Conv2d(n_hid, n_hid, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(n_hid)
        # self.conv2 = nn.Conv1d(n_hid, n_hid, kernel_size=7, stride=1, padding=6)
        self.conv2 = nn.Conv2d(n_hid, n_hid, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv21 = nn.Conv2d(n_hid, n_hid, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv22 = nn.Conv2d(n_hid, n_hid, kernel_size=(1, 3), stride=1, padding=(0, 1))
        # self.conv23 = nn.Conv2d(n_hid, n_hid, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(n_hid)
        self.bn21 = nn.BatchNorm2d(n_hid)
        self.bn3 = nn.BatchNorm2d(n_hid * 2)
        self.bn31 = nn.BatchNorm2d(n_hid * 2)
        self.bn4 = nn.BatchNorm2d(n_hid * 2)
        self.bn41 = nn.BatchNorm2d(n_hid * 2)
        self.bn5 = nn.BatchNorm2d(n_hid * 4)
        self.bn51 = nn.BatchNorm2d(n_hid * 4)
        self.bn6 = nn.BatchNorm2d(n_hid * 4)
        self.bn61 = nn.BatchNorm2d(n_hid * 4)
        # self.conv3 = nn.Conv2d(n_hid, n_hid, kernel_size=(1, 5), stride=1, padding=(0, 2))
        self.conv3 = nn.Conv2d(n_hid, n_hid * 2, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.conv31 = nn.Conv2d(n_hid * 2, n_hid * 2, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv32 = nn.Conv2d(n_hid, n_hid, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv33 = nn.Conv2d(n_hid, n_hid, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv4 = nn.Conv2d(n_hid * 2, n_hid * 2, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv41 = nn.Conv2d(n_hid * 2, n_hid * 2, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv42 = nn.Conv2d(n_hid, n_hid, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv43 = nn.Conv2d(n_hid, n_hid, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv44 = nn.Conv2d(n_hid, n_hid, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv5 = nn.Conv2d(n_hid * 2, n_hid * 4, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.conv51 = nn.Conv2d(n_hid * 4, n_hid * 4, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv52 = nn.Conv2d(n_hid * 4, n_hid * 4, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv53 = nn.Conv2d(n_hid * 4, n_hid * 4, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv6 = nn.Conv2d(n_hid * 4, n_hid * 4, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv61 = nn.Conv2d(n_hid * 4, n_hid * 4, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv62 = nn.Conv2d(n_hid * 4, n_hid * 4, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv63 = nn.Conv2d(n_hid * 4, n_hid * 4, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.jiangwei1 = nn.Linear(64, 8)
        self.jiangwei2 = nn.Linear(8, 1)
        self.skipconv2 = nn.Conv2d(n_hid, n_hid * 2, kernel_size=(1, 1), stride=(1, 2))
        self.skipconv4 = nn.Conv2d(n_hid * 2, n_hid * 4, kernel_size=(1, 1), stride=(1, 2))

    def forward(self, inputs):
        # Input shape: [num_sims * num_edges, num_dims, num_timesteps]

        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv11(x))
        # x = F.relu(self.conv12(x))
        # x = F.relu(self.conv13(x))
        x = self.bn1(x)
        x = self.pool1(x)
        xskip1 = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn21(self.conv21(x))
        x = F.relu(x + xskip1)
        # print(np.shape(x))
        # x = F.relu(self.conv22(x))
        # x = F.relu(self.conv23(x))
        # x = self.bn2(x)
        # x = F.relu(self.conv3(x))
        xskip2 = self.skipconv2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn31(self.conv31(x))
        x = F.relu(x + xskip2)
        # x = F.relu(self.conv32(x))
        # x = F.relu(self.conv33(x))
        # x = self.bn3(x)
        # x = self.pool3(x)
        xskip3 = x
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn41(self.conv41(x))
        x = F.relu(x + xskip3)
        # x = F.relu(self.conv42(x))
        # x = F.relu(self.conv43(x))
        # x = F.relu(self.conv44(x))
        # x = self.bn4(x)
        xskip4 = self.skipconv4(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.bn51(self.conv51(x))
        x = F.relu(x + xskip4)
        xskip5 = x
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.bn61(self.conv61(x))
        x = F.relu(x + xskip5)
        x = x.permute(0, 2, 1, 3)
        edge_prob = F.relu(self.jiangwei1(x))
        edge_prob = self.jiangwei2(edge_prob)

        return edge_prob


class CNNEncoder(nn.Module):
    def __init__(self, channels, n_hid, n_out, do_prob=0., factor=True):
        super(CNNEncoder, self).__init__()
        self.dropout_prob = do_prob

        self.factor = factor

        self.cnn = CNN(1 * 2, n_hid, n_hid, channels)
        self.mlp1 = MLP(n_hid * 4, n_hid * 4, n_hid * 4, do_prob)
        self.mlp2 = MLP(n_hid, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
        # self.fc_out = nn.Linear(n_hid, 8)
        # self.fc_out1 = nn.Linear(8, n_out)
        # self.fc_out2 = nn.Linear(8, n_out)
        self.fc_out1 = nn.Linear(n_hid * 4, n_out)
        self.fc_out2 = nn.Linear(n_hid * 4, n_out)

    def node2edge_temporal(self, inputs, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.

        x = inputs.view(inputs.size(0), inputs.size(1), -1)

        receivers = torch.matmul(rel_rec, x)
        receivers = receivers.view(inputs.size(0) * receivers.size(1),
                                   inputs.size(2), inputs.size(3))
        receivers = receivers.transpose(2, 1)

        senders = torch.matmul(rel_send, x)
        senders = senders.view(inputs.size(0) * senders.size(1),
                               inputs.size(2),
                               inputs.size(3))
        senders = senders.transpose(2, 1)

        # receivers and senders have shape:
        # [num_sims * num_edges, num_dims, num_timesteps]
        edges = torch.cat([senders, receivers], dim=1)
        return edges

    def forward(self, inputs, rel_rec, rel_send):
        # Input has shape: [num_sims, num_atoms, num_timesteps, num_dims]
        data1 = inputs.unsqueeze(3)
        edges = self.node2edge_temporal(data1, rel_rec, rel_send)
        x = edges.view(inputs.size(0), inputs.size(1) * inputs.size(1), edges.size(1), edges.size(2))
        x = x.permute(0, 2, 1, 3)
        x = self.cnn(x)
        x = x.view(inputs.size(0), inputs.size(1) * inputs.size(1), -1)
        x = self.mlp1(x)

        return self.fc_out1(x), self.fc_out2(x)


class MLPDecoder(nn.Module):
    """MLP decoder module."""

    def __init__(self, n_in_node, edge_types, msg_hid, msg_out, n_hid,
                 do_prob=0., skip_first=False):
        super(MLPDecoder, self).__init__()
        self.lag = n_in_node

        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * n_in_node, msg_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(msg_hid, msg_out) for _ in range(edge_types)])
        self.msg_out_shape = msg_out
        self.skip_first_edge_type = skip_first

        self.out_fc1 = nn.Linear(n_in_node + msg_out, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, 1)

        print('Using learned interaction net decoder.')

        self.dropout_prob = do_prob

    def single_step_forward(self, single_timestep_inputs, rel_rec, rel_send,
                            single_timestep_rel_type):

        receivers = torch.matmul(rel_rec, single_timestep_inputs)
        senders = torch.matmul(rel_send, single_timestep_inputs)
        pre_msg = torch.cat([senders, receivers], dim=-1)

        all_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1),
                               pre_msg.size(2), self.msg_out_shape)
        if single_timestep_inputs.is_cuda:
            all_msgs = all_msgs.cuda()

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0

        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.relu(self.msg_fc2[i](msg))
            msg = msg * single_timestep_rel_type[:, :, :, i:i + 1]
            all_msgs += msg

        # Aggregate all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()

        # Skip connection
        aug_inputs = torch.cat([single_timestep_inputs, agg_msgs], dim=-1)

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(aug_inputs)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        return pred

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()  # 计算标准差
        #  从标准的正态分布中随机采样一个eps
        eps = torch.FloatTensor(std.size()).normal_()
        eps = eps.cuda()
        # eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, inputs, mu, logvar, rel_rec, rel_send):

        rel_type = reparametrize(mu, logvar)
        time_length = inputs.size(2)
        data = None

        for i in range(self.lag):
            if i == 0:
                data = inputs[:, :, i:(time_length - self.lag + i), :]
            else:
                data = torch.cat([data, inputs[:, :, i:(time_length - self.lag + i), :]], dim=-1)
        data = data.transpose(1, 2).contiguous()
        sizes = [rel_type.size(0), data.size(1), rel_type.size(1),
                 rel_type.size(2)]
        curr_rel_type = rel_type.unsqueeze(1).expand(sizes)
        pred_all = self.single_step_forward(data, rel_rec, rel_send, curr_rel_type)

        return pred_all.transpose(1, 2).contiguous()


class KLdecoder(nn.Module):
    """MLP decoder module."""

    def __init__(self, edge_types):
        super(KLdecoder, self).__init__()
        self.yjunzhi = nn.Parameter(torch.zeros((2, 19 * 19, edge_types)))

    def forward(self, mu):
        z_prior_mean = torch.unsqueeze(mu, 1)
        z_prior_mean = z_prior_mean - torch.unsqueeze(self.yjunzhi, 0)

        return z_prior_mean


class MLPclassification(nn.Module):
    """MLP decoder module."""

    def __init__(self, edge_types):
        super(MLPclassification, self).__init__()
        self.yinbianliangfenleiceng1 = nn.Linear(19 * 19 * edge_types, 19 * edge_types * 4)
        self.yinbianliangfenleiceng2 = nn.Linear(19 * edge_types * 4, 8)
        self.yinbianliangfenleiceng3 = nn.Linear(8, 2)
        # self.yjunzhi = nn.Parameter(torch.ones((2, 19 * 19, edge_types))/edge_types)
        # self.yjunzhi = nn.Parameter(torch.rand((2, 19 * 19, edge_types)))

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()  # 计算标准差
        #  从标准的正态分布中随机采样一个eps
        eps = torch.FloatTensor(std.size()).normal_()
        eps = eps.cuda()
        # eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, mu, logvar):
        # NOTE: Assumes that we have the same graph across all samples.
        rel_type = reparametrize(mu, logvar)
        yinbianliangfenlei = rel_type.reshape(-1, rel_type.size(1) * rel_type.size(2))
        yfenlei = F.relu(self.yinbianliangfenleiceng1(yinbianliangfenlei))
        # yfenlei = F.dropout(yfenlei,p=0.5)
        yfenlei = F.relu(self.yinbianliangfenleiceng2(yfenlei))
        # yfenlei = F.dropout(yfenlei,p=0.5)
        # yfenlei = self.yinbianliangfenleiceng3(yfenlei)
        yfenlei = F.softmax(self.yinbianliangfenleiceng3(yfenlei), dim=1)

        return yfenlei


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def nll_gaussian(preds, target, variance, add_const=False):
    neg_log_p = ((preds - target) ** 2 / (2 * variance))
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0) * target.size(1) * target.size(2))


def kl_loss_function(logvar, z_prior_mean, zhenshiylabels):
    logvar = torch.unsqueeze(logvar, 1)
    kl_loss = - 0.5 * (logvar - torch.square(z_prior_mean) - logvar.exp())
    kl_loss = kl_loss.view(kl_loss.size(0), kl_loss.size(1), kl_loss.size(2) * kl_loss.size(3))
    size1 = kl_loss.size(2)
    kl_loss = torch.sum(torch.matmul(torch.unsqueeze(zhenshiylabels, 1), kl_loss), 2)
    kl_loss = torch.sum(kl_loss) / size1

    return kl_loss / logvar.size(0)


def clf_loss_function(yfenlei, zhenshiylabels):
    cat_loss = torch.sum(-zhenshiylabels * torch.log(yfenlei + 1e-7), 1)
    cat_loss = torch.sum(cat_loss)

    return cat_loss / yfenlei.size(0)


def reparametrize(mu, logvar):
    std = logvar.mul(0.5).exp_()  # 计算标准差
    #  从标准的正态分布中随机采样一个eps
    eps = torch.FloatTensor(std.size()).normal_()
    eps = eps.cuda()
    # eps = Variable(eps)
    return eps.mul(std).add_(mu)


def main():
    channels = 19
    timepoints = 512
    off_diag = np.ones([channels, channels])
    # off_diag = np.ones([channels, channels]) - np.eye(channels)
    '''这里是计算节点入边的onehot编码，以便在网络中利用矩阵乘法提取指定的时间序列节点，假设有5个节点，
       对节点1，用来和其他时间序列拼接的入边onehot编码如下所示
                                        [1 0 0 0 0]
                                        [1 0 0 0 0]
                                        [1 0 0 0 0]
                                        [1 0 0 0 0]'''
    rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
    '''这里是计算节点出边的onehot编码，以便在网络中利用矩阵乘法提取指定的时间序列节点，假设有5个节点，
       对节点1，其他时间序列和节点1拼接的出边onehot编码如下所示
                                            [0 1 0 0 0]
                                            [0 0 1 0 0]
                                            [0 0 0 1 0]
                                            [0 0 0 0 1]'''
    rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
    rel_rec = torch.FloatTensor(rel_rec)
    rel_send = torch.FloatTensor(rel_send)
    rel_rec = rel_rec.cuda()
    rel_send = rel_send.cuda()
    randlist1 = []
    randlist2 = []
    # wenjianlujing1 = '/root/autodl-tmp/HBN_restshiyan1/eeg19daoepoch1/'
    wenjianlujing1 = '/root/autodl-tmp/Malaysia mddshuju/eeg19daoepoch/'
    wenjianlujing2 = '/root/autodl-tmp/Malaysia mddshuju/eeg19daoepoch/jieguojilu/1/'
    for a18 in range(27):
        randlist1.append(a18 + 1)
    for a19 in range(28):
        randlist2.append(a19 + 1)
    randmdd = randlist1[:]
    randnomal = randlist2[:]
    randomliuyi = randmdd + randnomal
    print(randomliuyi)
    mddshizhehuafen = [2, 2, 2, 3, 3, 3, 3, 3, 3, 3]
    normalshizhehuafen = [2, 2, 3, 3, 3, 3, 3, 3, 3, 3]

    accuracywuzhe = []
    sensitivitywuzhe = []
    specificitywuzhe = []
    precisionwuzhe = []
    f1scorewuzhe = []
    for shizheid in range(10):
        tpcount = 0
        tncount = 0
        fpcount = 0
        fncount = 0
        mddtrainbeishi = None
        normaltrainbeishi = None
        mddtestbeishi = None
        normaltestbeishi = None
        shizhetraintrails = None
        shizhetesttrails = None
        mddindex = 0
        normalindex = 0
        patience = 60
        patiencecounter = 0
        best_score = None
        #if shizheid not in [1, 5, 7, 8]:
            # if shizheid not in [4,6,8,9] :
        #if shizheid not in [0] :
        if shizheid in [100]:
            continue
        else:
            for huafen in range(10):
                mddslice = randmdd[mddindex: mddindex + mddshizhehuafen[huafen]]
                normalslice = randnomal[normalindex: normalindex + normalshizhehuafen[huafen]]
                mddindex = mddindex + mddshizhehuafen[huafen]
                normalindex = normalindex + normalshizhehuafen[huafen]
                if huafen == shizheid:
                    mddtestbeishi = mddslice
                    normaltestbeishi = normalslice
                elif mddtrainbeishi is None:
                    mddtrainbeishi = mddslice
                    normaltrainbeishi = normalslice
                else:
                    mddtrainbeishi = mddtrainbeishi + mddslice
                    normaltrainbeishi = normaltrainbeishi + normalslice
            for i in range(len(mddtestbeishi)):
                hbneeg1 = io.loadmat(wenjianlujing1 + 'MDD' + str(mddtestbeishi[i]))
                trails = hbneeg1['MDD' + str(mddtestbeishi[i])][:, :, :150]
                trails = np.transpose(trails, [2, 0, 1])
                if i == 0:
                    shizhetesttrails = trails
                else:
                    shizhetesttrails = np.concatenate((shizhetesttrails, trails), axis=0)
            ymddtest = np.ones((np.shape(shizhetesttrails)[0], 1))
            for i in range(len(normaltestbeishi)):
                hbneeg1 = io.loadmat(wenjianlujing1 + 'Normal' + str(normaltestbeishi[i]))
                trails = hbneeg1['Normal' + str(normaltestbeishi[i])][:, :, :150]
                trails = np.transpose(trails, [2, 0, 1])
                shizhetesttrails = np.concatenate((shizhetesttrails, trails), axis=0)
            ynormaltest = np.zeros((np.shape(shizhetesttrails)[0] - len(ymddtest), 1))
            yshizhetest = np.concatenate((ymddtest, ynormaltest), axis=0)
            for i in range(len(mddtrainbeishi)):
                hbneeg1 = io.loadmat(wenjianlujing1 + 'MDD' + str(mddtrainbeishi[i]))
                trails = hbneeg1['MDD' + str(mddtrainbeishi[i])][:, :, :150]
                trails = np.transpose(trails, [2, 0, 1])
                if i == 0:
                    shizhetraintrails = trails
                else:
                    shizhetraintrails = np.concatenate((shizhetraintrails, trails), axis=0)
            ymddtrain = np.ones((np.shape(shizhetraintrails)[0], 1))
            for i in range(len(normaltrainbeishi)):
                hbneeg1 = io.loadmat(wenjianlujing1 + 'Normal' + str(normaltrainbeishi[i]))
                trails = hbneeg1['Normal' + str(normaltrainbeishi[i])][:, :, :150]
                trails = np.transpose(trails, [2, 0, 1])
                shizhetraintrails = np.concatenate((shizhetraintrails, trails), axis=0)
            ynormaltrain = np.zeros((np.shape(shizhetraintrails)[0] - len(ymddtrain), 1))
            yshizhetrain = np.concatenate((ymddtrain, ynormaltrain), axis=0)

            shiyueeglrx_train, shiyueeglry_train = sklearn.utils.shuffle(shizhetraintrails, yshizhetrain)
            shiyueeglrx_train = (shiyueeglrx_train - np.mean(shiyueeglrx_train, axis=2, keepdims=True)) / np.std(
                shiyueeglrx_train, axis=2, keepdims=True)
            shiyueeglrx_test = (shizhetesttrails - np.mean(shizhetesttrails, axis=2, keepdims=True)) / np.std(
                shizhetesttrails, axis=2, keepdims=True)
            shiyueeglrx_train = torch.tensor(shiyueeglrx_train)
            shiyueeglry_train = torch.tensor(shiyueeglry_train)
            shiyueeglrx_test = torch.tensor(shiyueeglrx_test)
            shiyueeglry_test = torch.tensor(yshizhetest)
            print(np.shape(shiyueeglrx_train))
            print(np.shape(shiyueeglry_train))
            print(np.shape(shiyueeglrx_test))
            print(np.shape(shiyueeglry_test))

            train_eegeog1 = TensorDataset(shiyueeglrx_train, shiyueeglry_train)
            test_eegeog1 = TensorDataset(shiyueeglrx_test, shiyueeglry_test)
            train_eegeog1_load = DataLoader(dataset=train_eegeog1, batch_size=3)
            test_eegeog1_load = DataLoader(dataset=test_eegeog1, batch_size=3)
            onehotbianma = preprocessing.OneHotEncoder()
            onehotbianma.fit([[0], [1]])

            dims = 1
            lag = 4
            encoder_hidden = 64
            edge_types = 1
            encoder_dropout = 0.
            decoder_hidden = 128
            decoder_dropout = 0.
            prediction_steps = 1
            tau = 0.5
            encoder = CNNEncoder(dims, encoder_hidden, edge_types, encoder_dropout, factor=False).cuda()
            decoder = MLPDecoder(n_in_node=lag,
                                 edge_types=edge_types,
                                 msg_hid=decoder_hidden,
                                 msg_out=decoder_hidden,
                                 n_hid=decoder_hidden,
                                 do_prob=decoder_dropout
                                 ).cuda()
            clfnet = MLPclassification(edge_types=edge_types).cuda()
            kldecoder = KLdecoder(edge_types=edge_types).cuda()
            model = {}
            model['en'] = encoder
            model['de'] = decoder
            model['kl'] = kldecoder
            model['clf'] = clfnet
            tasks = ['de', 'kl', 'clf']
            # clf_loss = nn.CrossEntropyLoss()
            model_params = []
            for m in model:
                model_params += model[m].parameters()
            # optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.0002)
            optimizer = optim.Adam(model_params, lr=0.0002)
            # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 16], gamma=0.1, verbose=True)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, verbose=True)
            for epoch in range(105):  # loop over the dataset multiple times

                logfile = open(wenjianlujing2+'vallog.txt','a')
                runningloss_de = 0.0
                runningloss_cat = 0.0
                runningloss_kl = 0.0
                for m in model:
                    model[m].train()
                loss_data = {}
                grads = {}
                scale = {}
                for b1, data in enumerate(train_eegeog1_load, 0):
                    inputs, labels = data
                    randseed1 = np.random.randint(1,10000000)
                    # zhenshiylabels = onehotbianma.transform(labels).toarray()
                    # zhenshiylabels = torch.tensor(zhenshiylabels)
                    # zhenshiylabels = zhenshiylabels.float()
                    zhenshiylabels_oh = onehotbianma.transform(labels).toarray()
                    zhenshiylabels_oh = torch.tensor(zhenshiylabels_oh)
                    zhenshiylabels_oh = zhenshiylabels_oh.float()
                    zhenshiylabels_oh = zhenshiylabels_oh.cuda()
                    inputs = inputs.float()
                    inputs = torch.unsqueeze(inputs, 3)
                    inputs = inputs.cuda()

                    optimizer.zero_grad()
                    mu, logvar = model['en'](inputs, rel_rec, rel_send)
                    mu_variable = Variable(mu.data.clone(), requires_grad=True)
                    logvar_variable = Variable(logvar.data.clone(), requires_grad=True)
                    target = inputs[:, :, lag:, :]
                    for t in tasks:
                        optimizer.zero_grad()
                        if t == 'clf':
                            torch.manual_seed(randseed1)
                            out_t = model[t](mu_variable, logvar_variable)
                            loss = clf_loss_function(out_t, zhenshiylabels_oh)
                        elif t == 'kl':
                            out_t = model[t](mu_variable)
                            loss = kl_loss_function(logvar_variable, out_t, zhenshiylabels_oh)
                        else:
                            torch.manual_seed(randseed1)
                            out_t = model[t](inputs, mu_variable, logvar_variable, rel_rec, rel_send)
                            loss = nll_gaussian(out_t, target, variance=1)
                        loss_data[t] = loss.data
                        loss.backward()
                        grads[t] = []
                        grads[t].append(Variable(mu_variable.grad.data.clone(), requires_grad=False))
                        grads[t].append(Variable(logvar_variable.grad.data.clone(), requires_grad=False))
                        mu_variable.grad.data.zero_()
                        logvar_variable.grad.data.zero_()

                    gn = gradient_normalizers(grads, loss_data, 'l2')
                    for t in tasks:
                        for gr_i in range(len(grads[t])):
                            grads[t][gr_i] = grads[t][gr_i] / gn[t]
                    sol, min_norm = MinNormSolver.find_min_norm_element_FW([grads[t] for t in tasks])
                    for i, t in enumerate(tasks):
                        scale[t] = float(sol[i])

                    optimizer.zero_grad()
                    mu, logvar = model['en'](inputs, rel_rec, rel_send)
                    for t in tasks:
                        if t == 'clf':
                            torch.manual_seed(randseed1)
                            out_t = model[t](mu, logvar)
                            loss_t = clf_loss_function(out_t, zhenshiylabels_oh)
                            loss_data[t] = loss_t.data
                            loss = loss + scale[t] * loss_t
                        elif t == 'kl':
                            out_t = model[t](mu)
                            loss_t = kl_loss_function(logvar, out_t, zhenshiylabels_oh)
                            loss_data[t] = loss_t.data
                            loss = loss + scale[t] * loss_t
                        else:
                            torch.manual_seed(randseed1)
                            out_t = model[t](inputs, mu, logvar, rel_rec, rel_send)
                            loss_t = nll_gaussian(out_t, target, variance=1)
                            loss_data[t] = loss_t.data
                            loss = scale[t] * loss_t
                    loss.backward()
                    optimizer.step()
                    runningloss_de += loss_data['de'].item()
                    runningloss_cat += loss_data['clf'].item()
                    runningloss_kl += loss_data['kl'].item()
                    if b1 % 200 == 199:  # print every mini-batches
                        print(
                            'counts: %d, epochs: %d, batchs: %5d,   loss_de: %.4f, loss_kl: %.4f, loss_cat: %.4f' %
                            (shizheid + 1, epoch + 1, b1 + 1, runningloss_de / 200, runningloss_kl / 200, runningloss_cat / 200))
                        print(
                            'counts: %d, epochs: %d, batchs: %5d,   loss_de: %.4f, loss_kl: %.4f, loss_cat: %.4f' %
                            (shizheid + 1, epoch + 1, b1 + 1, runningloss_de / 200, runningloss_kl / 200, runningloss_cat / 200),file=logfile)
                        runningloss_de = 0.0
                        runningloss_cat = 0.0
                        runningloss_kl = 0.0
                scheduler.step()

                runningloss_de = 0.0
                runningloss_cat = 0.0
                runningloss_kl = 0.0
                for m in model:
                    model[m].eval()
                with torch.no_grad():
                    for i, data in enumerate(test_eegeog1_load, 0):
                        inputs, labels = data
                        zhenshiylabels_oh = onehotbianma.transform(labels).toarray()
                        zhenshiylabels_oh = torch.tensor(zhenshiylabels_oh)
                        zhenshiylabels_oh = zhenshiylabels_oh.float()
                        zhenshiylabels_oh = zhenshiylabels_oh.cuda()
                        inputs = inputs.float()
                        inputs = torch.unsqueeze(inputs, 3)
                        inputs = inputs.cuda()

                        mu, logvar = model['en'](inputs, rel_rec, rel_send)
                        target = inputs[:, :, lag:, :]
                        for t in tasks:
                            if t == 'clf':
                                out_t = model[t](mu, logvar)
                                loss_t = clf_loss_function(out_t, zhenshiylabels_oh)
                                loss_data[t] = loss_t.data
                            elif t == 'kl':
                                out_t = model[t](mu)
                                loss_t = kl_loss_function(logvar, out_t, zhenshiylabels_oh)
                                loss_data[t] = loss_t.data
                            else:
                                out_t = model[t](inputs, mu, logvar, rel_rec, rel_send)
                                loss_t = nll_gaussian(out_t, target, variance=1)
                                loss_data[t] = loss_t.data
                        runningloss_de += loss_data['de'].item()
                        runningloss_cat += loss_data['clf'].item()
                        runningloss_kl += loss_data['kl'].item()
                print('counts: %d, epochs: %d, loss_de: %.4f, loss_kl: %.4f, loss_cat: %.4f' %
                      (shizheid + 1, epoch + 1, runningloss_de / (i + 1), runningloss_kl / (i + 1), runningloss_cat / (i + 1)))
                print('counts: %d, epochs: %d, loss_de: %.4f, loss_kl: %.4f, loss_cat: %.4f' %
                      (shizheid + 1, epoch + 1, runningloss_de / (i + 1), runningloss_kl / (i + 1), runningloss_cat / (i + 1)),file=logfile)

                correct = 0
                total = 0
                with torch.no_grad():
                    for i, data in enumerate(test_eegeog1_load, 0):
                        inputs, labels = data
                        labels = labels.cuda().squeeze(1)
                        inputs = inputs.float()
                        inputs = torch.unsqueeze(inputs, 3)
                        inputs = inputs.cuda()
                        mu, logvar = model['en'](inputs, rel_rec, rel_send)
                        out_t = model['clf'](mu, logvar)
                        _, predicted = torch.max(out_t, 1)
                        correct += (predicted == labels.long()).sum().item()
                        total += inputs.size(0)
                print('Accuracy of the network on the test images: %f %%' % (
                        100 * correct / total))
                print(correct)
                print(total)
                print('Accuracy of the network on the test images: %f %%' % (
                        100 * correct / total),file=logfile)
                print(correct,file=logfile)
                print(total,file=logfile)
                
                score = correct / total
                if best_score is None:
                    best_score = score
                    torch.save({'encoder_state_dict': encoder.state_dict(),
                                'decoder_state_dict': decoder.state_dict(),
                                'kldecoder_state_dict': kldecoder.state_dict(),
                                'clfnet_state_dict': clfnet.state_dict()},
                               f=wenjianlujing2 + 'shifoldmodel' + str(shizheid+1) +'.pt')
                elif score <= best_score:
                    patiencecounter += 1
                    if patiencecounter >= patience:
                        break
                else:
                    best_score = score
                    torch.save({'encoder_state_dict': encoder.state_dict(),
                                'decoder_state_dict': decoder.state_dict(),
                                'kldecoder_state_dict': kldecoder.state_dict(),
                                'clfnet_state_dict': clfnet.state_dict()},
                               f=wenjianlujing2 + 'shifoldmodel' + str(shizheid+1) + '.pt')
                    patiencecounter = 0
                logfile.close()


if __name__ == '__main__':
    main()




