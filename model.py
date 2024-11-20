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


class skipmodule(nn.Module):
    def __init__(self, n_hid, factor):
        super(skipmodule, self).__init__()

        self.identicalfactor = factor

        self.conv_identical1 = nn.Conv2d(n_hid, n_hid, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv_identical2 = nn.Conv2d(n_hid, n_hid, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.bn_identical1 = nn.BatchNorm2d(n_hid)
        self.bn_identical2 = nn.BatchNorm2d(n_hid)

        self.conv_unsqueeze1 = nn.Conv2d(n_hid, n_hid*2, kernel_size=(1, 3), stride=(1,2), padding=(0, 1))
        self.conv_unsqueeze2 = nn.Conv2d(n_hid*2, n_hid*2, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.bn_unsqueeze1 = nn.BatchNorm2d(n_hid * 2)
        self.bn_unsqueeze2 = nn.BatchNorm2d(n_hid * 2)
        self.skipconv = nn.Conv2d(n_hid, n_hid * 2, kernel_size=(1, 1), stride=(1, 2))

    def forward(self, x):
        if self.identicalfactor:
            xskip = x
            x = F.relu(self.bn_identical1(self.conv_identical1(x)))
            x = self.bn_identical2(self.conv_identical2(x))
            x = F.relu(x + xskip)
        else:
            xskip = self.skipconv(x)
            x = F.relu(self.bn_unsqueeze1(self.conv_unsqueeze1(x)))
            x = self.bn_unsqueeze2(self.conv_unsqueeze2(x))
            x = F.relu(x + xskip)

        return x


class MLP(nn.Module):
    def __init__(self, n_in, n_hid, n_out):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.batch_norm(x)


class CNN(nn.Module):
    def __init__(self, n_in, n_hid, timepoints):
        super(CNN, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=(1,2))
        self.conv1 = nn.Conv2d(n_in, n_hid, kernel_size=(1,3), stride=1, padding=(0,1))
        self.conv2 = nn.Conv2d(n_hid, n_hid, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.bn = nn.BatchNorm2d(n_hid)

        if timepoints % 8 == 0:
            self.fc1 = nn.Linear(int(timepoints/8),8)
        else:
            self.fc1 = nn.Linear(int(timepoints / 8) + 1, 8)
        self.fc2 = nn.Linear(8, 1)

        self.convmodule1 = skipmodule(n_hid, factor=True)
        self.convmodule2 = skipmodule(n_hid, factor=False)
        self.convmodule3 = skipmodule(n_hid * 2, factor=True)
        self.convmodule4 = skipmodule(n_hid * 2, factor=False)
        self.convmodule5 = skipmodule(n_hid * 4, factor=True)


    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.bn(x)
        x = self.pool(x)

        x = self.convmodule1(x)
        x = self.convmodule2(x)
        x = self.convmodule3(x)
        x = self.convmodule4(x)
        x = self.convmodule5(x)

        x = x.permute(0, 2, 1, 3)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class GraphEncodingNet(nn.Module):
    def __init__(self, timepoints, n_hid, n_out):
        super(GraphEncodingNet, self).__init__()

        self.cnn = CNN(2, n_hid, timepoints)
        self.mlp1 = MLP(n_hid*4, n_hid*4, n_hid*4)
        self.fc_out1 = nn.Linear(n_hid*4, n_out)
        self.fc_out2 = nn.Linear(n_hid*4, n_out)


    def node2edge_temporal(self, inputs, rel_rec, rel_send):

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

        data = inputs.unsqueeze(3)
        edges = self.node2edge_temporal(data, rel_rec, rel_send)
        x = edges.view(inputs.size(0), inputs.size(1) * inputs.size(1), edges.size(1), edges.size(2))
        x = x.permute(0, 2, 1, 3)
        x = self.cnn(x)
        x = x.view(inputs.size(0), inputs.size(1) * inputs.size(1), -1)
        x = self.mlp1(x)

        return self.fc_out1(x).squeeze(2), self.fc_out2(x).squeeze(2)


class GraphDecodingNet(nn.Module):
    def __init__(self, lag, msg_hid):
        super(GraphDecodingNet, self).__init__()

        self.msg_fc1 = nn.Linear(2 * lag, msg_hid)
        self.msg_fc2 = nn.Linear(msg_hid, msg_hid)
        self.msg_out_shape = msg_hid

        self.out_fc1 = nn.Linear(lag + msg_hid, msg_hid)
        self.out_fc2 = nn.Linear(msg_hid, msg_hid)
        self.out_fc3 = nn.Linear(msg_hid, 1)

    def single_step_forward(self, single_timestep_inputs, rel_rec, rel_send,
                            single_timestep_edges):

        receivers = torch.matmul(rel_rec, single_timestep_inputs)
        senders = torch.matmul(rel_send, single_timestep_inputs)
        pre_msg = torch.cat([senders, receivers], dim=-1)

        msg = F.relu(self.msg_fc1(pre_msg))
        msg = F.relu(self.msg_fc2(msg))
        msg = msg * single_timestep_edges.unsqueeze(3)

        # Aggregate all msgs to receiver
        agg_msgs = msg.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()
        aug_inputs = torch.cat([single_timestep_inputs, agg_msgs], dim=-1)

        pred = F.relu(self.out_fc1(aug_inputs))
        pred = F.relu(self.out_fc2(pred))
        pred = self.out_fc3(pred)

        return pred.squeeze(3)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        eps = eps.cuda()
        return eps.mul(std).add_(mu)

    def forward(self, inputs, mu, logvar, rel_rec, rel_send):

        edges = reparametrize(mu, logvar)
        inputs = inputs.unsqueeze(3)
        time_length = inputs.size(2)
        data = None

        for i in range(self.lag):
            if i == 0:
                data = inputs[:,:,i:(time_length-self.lag+i),:]
            else:
                data = torch.cat([data, inputs[:,:,i:(time_length-self.lag+i),:]], dim=-1)
        data = data.transpose(1, 2).contiguous()
        sizes = [edges.size(0), data.size(1), edges.size(1)]
        newedges = edges.unsqueeze(1).expand(sizes)
        pred_all = self.single_step_forward(data, rel_rec, rel_send, newedges)

        return pred_all.transpose(1, 2).contiguous()


class KLdecoder(nn.Module):

    def __init__(self, channels):
        super(KLdecoder, self).__init__()
        self.ymean = nn.Parameter(torch.zeros((2, channels * channels)))

    def forward(self, mu):
        z_prior_mean = torch.unsqueeze(mu, 1)
        z_prior_mean = z_prior_mean - torch.unsqueeze(self.ymean, 0)

        return z_prior_mean


class MLPclassification(nn.Module):

    def __init__(self, channels):
        super(MLPclassification, self).__init__()
        self.fc1 = nn.Linear(channels * channels, channels * 4)
        self.fc2 = nn.Linear(channels * 4, 8)
        self.fc3 = nn.Linear(8, 2)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        eps = eps.cuda()
        return eps.mul(std).add_(mu)

    def forward(self, mu, logvar):
        edges = reparametrize(mu, logvar)

        y = F.relu(self.fc1(edges))
        y = F.relu(self.fc2(y))
        y = F.softmax(self.fc3(y), dim=1)

        return y







