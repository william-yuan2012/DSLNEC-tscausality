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


class MLP(nn.Module):

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob


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
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        return self.batch_norm(x)


class CNN(nn.Module):
    def __init__(self, n_in, n_hid, n_out, channels):
        super(CNN, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2))
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 2))
        self.pool5 = nn.MaxPool2d(kernel_size=(1, 2))

        self.conv1 = nn.Conv2d(n_in, n_hid, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv11 = nn.Conv2d(n_hid, n_hid, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(n_hid)
        self.conv2 = nn.Conv2d(n_hid, n_hid, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv21 = nn.Conv2d(n_hid, n_hid, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv22 = nn.Conv2d(n_hid, n_hid, kernel_size=(1, 3), stride=1, padding=(0, 1))
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
        x = self.bn1(x)
        x = self.pool1(x)
        xskip1 = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn21(self.conv21(x))
        x = F.relu(x + xskip1)
        xskip2 = self.skipconv2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn31(self.conv31(x))
        x = F.relu(x + xskip2)
        xskip3 = x
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn41(self.conv41(x))
        x = F.relu(x + xskip3)
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

        edges = torch.cat([senders, receivers], dim=1)
        return edges

    def forward(self, inputs, rel_rec, rel_send):
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
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        eps = eps.cuda()
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

    def __init__(self, edge_types):
        super(KLdecoder, self).__init__()
        self.yjunzhi = nn.Parameter(torch.zeros((2, 19 * 19, edge_types)))

    def forward(self, mu):
        z_prior_mean = torch.unsqueeze(mu, 1)
        z_prior_mean = z_prior_mean - torch.unsqueeze(self.yjunzhi, 0)

        return z_prior_mean


class MLPclassification(nn.Module):

    def __init__(self, edge_types):
        super(MLPclassification, self).__init__()
        self.yinbianliangfenleiceng1 = nn.Linear(19 * 19 * edge_types, 19 * edge_types * 4)
        self.yinbianliangfenleiceng2 = nn.Linear(19 * edge_types * 4, 8)
        self.yinbianliangfenleiceng3 = nn.Linear(8, 2)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        eps = eps.cuda()
        return eps.mul(std).add_(mu)

    def forward(self, mu, logvar):
        rel_type = reparametrize(mu, logvar)
        yinbianliangfenlei = rel_type.reshape(-1, rel_type.size(1) * rel_type.size(2))
        yfenlei = F.relu(self.yinbianliangfenleiceng1(yinbianliangfenlei))
        yfenlei = F.relu(self.yinbianliangfenleiceng2(yfenlei))
        yfenlei = F.softmax(self.yinbianliangfenleiceng3(yfenlei), dim=1)

        return yfenlei







