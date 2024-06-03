#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

import torch.nn as nn
import torch
from torch.nn import functional as F
import math
import torchsnooper

_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
}


def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]


class KLRegression(nn.Module):
    """KL-divergence loss for probabilistic regression.
    It is computed using Monte Carlo (MC) samples from an arbitrary distribution."""

    def __init__(self, eps=0.0):
        super().__init__()
        self.eps = eps

    def forward(self, scores, sample_density, gt_density, mc_dim=-1):
        """Args:
            scores: predicted score values
            sample_density: probability density of the sample distribution
            gt_density: probability density of the ground truth distribution
            mc_dim: dimension of the MC samples"""

        exp_val = scores - torch.log(sample_density + self.eps)

        L = torch.logsumexp(exp_val, dim=mc_dim) - math.log(scores.shape[mc_dim]) - \
            torch.mean(scores * (gt_density / (sample_density + self.eps)), dim=mc_dim)

        return L.mean()


class MLRegression(nn.Module):
    """Maximum likelihood loss for probabilistic regression.
    It is computed using Monte Carlo (MC) samples from an arbitrary distribution."""

    def __init__(self, eps=0.0):
        super().__init__()
        self.eps = eps

    def forward(self, scores, sample_density, gt_density=None, mc_dim=-1):
        """Args:
            scores: predicted score values. First sample must be ground-truth
            sample_density: probability density of the sample distribution
            gt_density: not used
            mc_dim: dimension of the MC samples. Only mc_dim=1 supported"""

        assert mc_dim == 1
        assert (sample_density[:, 0, ...] == -1).all()

        exp_val = scores[:, 1:, ...] - torch.log(sample_density[:, 1:, ...] + self.eps)

        L = torch.logsumexp(exp_val, dim=mc_dim) - math.log(scores.shape[mc_dim] - 1) - scores[:, 0, ...]
        loss = L.mean()
        return loss


class KLRegressionGrid(nn.Module):
    """KL-divergence loss for probabilistic regression.
    It is computed using the grid integration strategy."""

    def forward(self, scores, gt_density, grid_dim=-1, grid_scale=1.0):
        """Args:
            scores: predicted score values
            gt_density: probability density of the ground truth distribution
            grid_dim: dimension(s) of the grid
            grid_scale: area of one grid cell"""

        score_corr = grid_scale * torch.sum(scores * gt_density, dim=grid_dim)

        L = torch.logsumexp(scores, dim=grid_dim) + math.log(grid_scale) - score_corr

        return L.mean()


class LBHinge(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """

    def __init__(self, error_metric=nn.MSELoss(), threshold=None, clip=None):
        super().__init__()
        self.error_metric = error_metric
        self.threshold = threshold if threshold is not None else -100
        self.clip = clip

    def forward(self, prediction, label, target_bb=None):
        negative_mask = (label < self.threshold).float()
        positive_mask = (1.0 - negative_mask)

        prediction = negative_mask * F.relu(prediction) + positive_mask * prediction

        loss = self.error_metric(prediction, positive_mask * label)

        if self.clip is not None:
            loss = torch.min(loss, torch.tensor([self.clip], device=loss.device))
        return loss


class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, reduction='mean', alpha=1.):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets, mask=None):
        inputs = inputs.narrow(1, 0, targets.shape[1])

        outputs = torch.log_softmax(inputs, dim=1)
        labels = torch.softmax(targets, dim=1)

        loss = (outputs * labels).mean(dim=1)

        if mask is not None:
            loss = loss * mask.float()

        if self.reduction == 'mean':
            # outputs = -torch.mean(loss) #  torch.mean(torch.stack(a))
            outputs = -loss.mean()
        elif self.reduction == 'sum':
            outputs = -torch.sum(loss)
        else:
            outputs = -loss

        return outputs


# @torchsnooper.snoop()
class MemoryMoCo(nn.Module):
    """Fixed-size queue with momentum encoder"""

    # T = 0.2 achieve best result?
    def __init__(self, inputSize, outputSize, K, T=0.07, use_softmax=False):
        super(MemoryMoCo, self).__init__()
        self.outputSize = outputSize
        self.inputSize = int(inputSize)
        self.queueSize = int(K)
        self.T = T
        self.index = 0
        self.use_softmax = use_softmax

        self.register_buffer('params', torch.tensor([-1]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(self.queueSize, inputSize) * (2 * stdv) + (-stdv))
        # self.register_buffer('spatial_memory', torch.rand(self.queueSize, inputSize).mul_(2 * stdv).add_(-stdv))
        # print('using queue shape: ({},{})'.format(self.queueSize, inputSize))
        # ==========LRU Cache==============
        # self.memory_bank = LRUCache(10000)
        # self.criterion = nn.CrossEntropyLoss()

    def forward(self, q, k, n, indexs=None):
        # n, sn,
        batchSize = q.shape[0]
        k = k.detach()

        Z = self.params[0].item()

        # pos logit
        l_pos = torch.bmm(q.view(batchSize, 1, -1), k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)

        # # neg logit
        # # queue = self.memory_bank.get_queue(self.queueSize, indexs)
        queue = self.memory.clone()
        # print('queue:', queue)
        # print('q', q)
        l_neg = torch.mm(queue.detach(), q.transpose(1, 0))
        l_neg = l_neg.transpose(0, 1)
        # out = torch.cat((l_pos, l_neg), dim=1)

        # other negative
        l_neg_2 = torch.bmm(q.view(batchSize, 1, -1), n.view(batchSize, -1, 1))
        l_neg_2 = l_neg_2.view(batchSize, 1)
        #
        # strong negative
        # l_s_neg = torch.bmm(q.view(batchSize, 1, -1), sn.view(batchSize, -1, 1))
        # l_s_neg = l_s_neg.view(batchSize, 1)

        out = torch.cat((l_pos, l_neg, l_neg_2), dim=1)
        # out = torch.cat((l_pos, l_neg, l_neg_2, l_s_neg), dim=1)

        if self.use_softmax:
            out = torch.div(out, self.T)
            out = out.squeeze().contiguous()
        else:
            out = torch.exp(torch.div(out, self.T))
            if Z < 0:
                self.params[0] = out.mean() * self.outputSize
                Z = self.params[0].clone().detach().item()
                print("normalization constant Z is set to {:.1f}".format(Z))
            # compute the out
            out = torch.div(out, Z).squeeze().contiguous()

        # label = torch.zeros([batchSize]).cuda().long()
        # loss = []
        # for i in range(batchSize):
        #     loss.append(self.criterion(out[i].unsqueeze(0), label[i].unsqueeze(0)))
        # print(loss)
        # self.memory_bank.batch_set(indexs, k, loss)
        # self.memory = self.memory_bank.update_queue(self.memory)
        # print(self.memory_bank.link)
        # update memory
        with torch.no_grad():
            out_ids = torch.arange(batchSize).cuda()
            out_ids += self.index
            out_ids = torch.fmod(out_ids, self.queueSize)  # 1 fmod 1.5 = 1  2 fmod 1.5 = 0.5
            out_ids = out_ids.long()
            self.memory.index_copy_(0, out_ids, k)
            self.index = (self.index + batchSize) % self.queueSize
        # add for spatial memory

        return out


eps = 1e-7

#
class NCECriterion(nn.Module):
    """
    Eq. (12): L_{NCE}
    """
    def __init__(self, n_data):
        super(NCECriterion, self).__init__()
        self.n_data = n_data

    def forward(self, x):
        # print('x', x)
        bsz = x.shape[0]
        m = x.size(1) - 1
        # print(m, 'm')
        # noise distribution
        Pn = 1 / float(self.n_data)
        # print('pn', Pn)
        # loss for positive pair
        P_pos = x.select(1, 0)
        #print('P_pos', P_pos)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()
        #print('log_D1', log_D1)
        # loss for K negative pair
        P_neg = x.narrow(1, 1, m)
        #print('P_neg', P_neg)
        #log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps))
        #print('log_D0', log_D0)
        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz
        #print('loss', loss)
        return loss


class NCESoftmaxLoss(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""

    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        bsz = x.shape[0]
        x = x.squeeze()
        label = torch.zeros([bsz]).cuda().long()
        loss = self.criterion(x, label)
        return loss


class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()
        self.margin = 0.2

    def forward(self, anchor, positive, negative):
        # pos_dist = (anchor - positive).pow(2).sum(1)
        # neg_dist = (anchor - negative).pow(2).sum(1)
        pos_dist = (anchor - positive)
        neg_dist = (anchor - negative)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()
