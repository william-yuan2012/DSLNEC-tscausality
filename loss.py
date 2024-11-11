import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



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