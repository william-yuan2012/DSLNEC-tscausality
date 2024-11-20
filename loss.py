import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def nll_gaussian(preds, target):
    neg_log_p = ((preds - target) ** 2 / 2 )
    return neg_log_p.sum() / (target.size(0) * target.size(1) * target.size(2))


def kl_loss_function(logvar, z_prior_mean, ylabels):
    logvar = torch.unsqueeze(logvar, 1)
    kl_loss = - 0.5 * (logvar - torch.square(z_prior_mean) - logvar.exp())
    size1 = kl_loss.size(2)
    kl_loss = torch.sum(torch.matmul(torch.unsqueeze(ylabels, 1), kl_loss), 2)
    kl_loss = torch.sum(kl_loss) / size1

    return kl_loss / logvar.size(0)


def clf_loss_function(ypred, ylabels):
    cat_loss = torch.sum(-ylabels * torch.log(ypred + 1e-7), 1)
    cat_loss = torch.sum(cat_loss)

    return cat_loss / ypred.size(0)