import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import math
# from .swin_transformer import swin_transformer_tiny, swin_transformer_small, swin_transformer_base
from Graph.resnet import resnet18, resnet50, resnet101
from Graph.graph import create_e_matrix
from Graph.graph_edge_model import GEM


# from Graph.basic_block import *


def bn_init(bn):
    bn.weight.data.fill_(1)
    bn.bias.data.zero_()


# Gated GCN Used to Learn Multi-dimensional Edge Features and Node Features
class GNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(GNN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        # GNN Matrix: E x N
        # Start Matrix Item:  define the source node of one edge
        # End Matrix Item:  define the target node of one edge
        # Algorithm details in Residual Gated Graph Convnets: arXiv preprint arXiv:1711.07553
        # or Benchmarking Graph Neural Networks: arXiv preprint arXiv:2003.00982v3

        start, end = create_e_matrix(self.num_classes)
        self.start = Variable(start, requires_grad=False)
        self.end = Variable(end, requires_grad=False)

        dim_in = self.in_channels
        dim_out = self.in_channels

        self.U1 = nn.Linear(dim_in, dim_out, bias=False)
        self.V1 = nn.Linear(dim_in, dim_out, bias=False)
        self.A1 = nn.Linear(dim_in, dim_out, bias=False)
        self.B1 = nn.Linear(dim_in, dim_out, bias=False)
        self.E1 = nn.Linear(dim_in, dim_out, bias=False)

        self.U2 = nn.Linear(dim_in, dim_out, bias=False)
        self.V2 = nn.Linear(dim_in, dim_out, bias=False)
        self.A2 = nn.Linear(dim_in, dim_out, bias=False)
        self.B2 = nn.Linear(dim_in, dim_out, bias=False)
        self.E2 = nn.Linear(dim_in, dim_out, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(2)
        self.bnv1 = nn.BatchNorm2d(num_classes)
        self.bne1 = nn.BatchNorm2d(num_classes * num_classes)

        self.bnv2 = nn.BatchNorm2d(num_classes)
        self.bne2 = nn.BatchNorm2d(num_classes * num_classes)

        self.act = nn.ReLU()

        self.init_weights_linear(dim_in, 1)

    def init_weights_linear(self, dim_in, gain):
        # conv1
        scale = gain * np.sqrt(2.0 / dim_in)
        self.U1.weight.data.normal_(0, scale)
        self.V1.weight.data.normal_(0, scale)
        self.A1.weight.data.normal_(0, scale)
        self.B1.weight.data.normal_(0, scale)
        self.E1.weight.data.normal_(0, scale)

        self.U2.weight.data.normal_(0, scale)
        self.V2.weight.data.normal_(0, scale)
        self.A2.weight.data.normal_(0, scale)
        self.B2.weight.data.normal_(0, scale)
        self.E2.weight.data.normal_(0, scale)

        bn_init(self.bnv1)
        bn_init(self.bne1)
        bn_init(self.bnv2)
        bn_init(self.bne2)

    def forward(self, x, edge):
        # device
        dev = x.get_device()
        if dev >= 0:
            start = self.start.to(dev)
            end = self.end.to(dev)

        # GNN Layer 1:
        res = x
        Vix = self.A1(x)  # V x d_out
        Vjx = self.B1(x)  # V x d_out
        e = self.E1(edge)  # E x d_out\
        # Vix = Vix.permute(1, 0, 2, 3)
        # Vjx = Vjx.permute(1, 0, 2, 3)
        # print('end:', end.shape)  # [25, 5]
        # print('Vix:', Vix.shape)  # [8, 5, 8, 2048]
        # print('start:', start.shape)  # [25, 5]
        # print('Vjx:', Vjx.shape)  # [8, 5, 8, 2048]
        # print('e:', e.shape)  # [8, 25, 8, 2048]

        # edge = edge + self.act(self.bne1(
        #     torch.einsum('ev, bvc -> bec', (end, Vix)).permute(1, 0, 2, 3) + torch.einsum('ev, bvc -> bec', (start, Vjx)).permute(1, 0, 2, 3) + e))  # E x d_out
        edge = edge + self.act(self.bne1(
            torch.einsum('ev, bvtc -> betc', (end, Vix)) + torch.einsum('ev, bvtc -> betc', (start, Vjx)) + e))  # E x d_out

        e = self.sigmoid(edge)
        b, _, t, c = e.shape
        e = e.view(b, self.num_classes, self.num_classes, t, c)
        e = self.softmax(e)
        e = e.view(b, -1, t, c)

        Ujx = self.V1(x)  # V x H_out
        # Ujx = Ujx.permute(1, 0, 2, 3)
        Ujx = torch.einsum('ev, bvtc -> betc', (start, Ujx))  # E x H_out
        Uix = self.U1(x)  # V x H_out
        # Uix = Uix.permute(1, 0, 2, 3)
        # e = e.permute(1, 0, 2, 3)
        x = Uix + torch.einsum('ve, betc -> bvtc', (end.t(), e * Ujx)) / self.num_classes  # V x H_out
        x = self.act(res + self.bnv1(x))
        res = x

        # GNN Layer 2:
        Vix = self.A2(x)  # V x d_out
        Vjx = self.B2(x)  # V x d_out
        e = self.E2(edge)  # E x d_out
        edge = edge + self.act(self.bne2(
            torch.einsum('ev, bvtc -> betc', (end, Vix)) + torch.einsum('ev, bvtc -> betc', (start, Vjx)) + e))  # E x d_out

        e = self.sigmoid(edge)
        # print('e.shape:',e.shape)  # [8, 25, 8, 2048]
        b, _, t, c = e.shape
        e = e.view(b, self.num_classes, self.num_classes, t, c)
        e = self.softmax(e)
        e = e.view(b, -1, t, c)

        Ujx = self.V2(x)  # V x H_out
        Ujx = torch.einsum('ev, bvtc -> betc', (start, Ujx))  # E x H_out
        Uix = self.U2(x)  # V x H_out
        x = Uix + torch.einsum('ve, betc -> bvtc', (end.t(), e * Ujx)) / self.num_classes  # V x H_out
        x = self.act(res + self.bnv2(x))
        return x, edge


class Action(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Action, self).__init__()
        # The head of network
        # Input: the feature maps x from backbone
        # Output: the AU recognition probabilities cl And the logits cl_edge of edge features for classification
        # Modules: 1. AFG extracts individual Au feature maps U_1 ---- U_N
        #          2. GEM: graph edge modeling for learning multi-dimensional edge features
        #          3. Gated-GCN for graph learning with node and multi-dimensional edge features
        # sc: individually calculate cosine similarity between node features and a trainable vector.
        # edge fc: for edge prediction

        self.in_channels = in_channels
        self.num_classes = num_classes
        # class_linear_layers = []
        # for i in range(self.num_classes):
        #     layer = LinearBlock(self.in_channels, self.in_channels)
        #     class_linear_layers += [layer]
        # self.class_linears = nn.ModuleList(class_linear_layers)
        self.edge_extractor = GEM(self.in_channels, self.num_classes)
        self.gnn = GNN(self.in_channels, self.num_classes)
        self.sc = nn.Parameter(torch.FloatTensor(torch.zeros(self.num_classes, self.in_channels)))
        self.edge_fc = nn.Linear(self.in_channels, 1)
        self.relu = nn.ReLU()

        nn.init.xavier_uniform_(self.edge_fc.weight)
        nn.init.xavier_uniform_(self.sc)

    def forward(self, af, ef, gf, attf, sf):
        # AFG
        f_u = []
        # for i, layer in enumerate(self.class_linears):
        #     f_u.append(layer(x).unsqueeze(1))
        # print('af:', af)
        # print('af:', ef)
        # print('af:', gf)
        # print('af:', attf)
        # print('af:', sf)
        b,c,t,h,w = af[0].shape
        af = af[0].view(b, c, t, -1).permute(0, 2, 3, 1).cuda()  # b, t, d, c
        ef = ef.view(b, c, t, -1).permute(0, 2, 3, 1).cuda()
        gf = gf.view(b, c, t, -1).permute(0, 2, 3, 1).cuda()
        attf = attf.view(b, c, t, -1).permute(0, 2, 3, 1).cuda()
        sf = sf.view(b, c, t, -1).permute(0, 2, 3, 1).cuda()
        # f_u.append(af)
        # f_u.append(ef)
        # f_u.append(gf)
        # f_u.append(attf)
        # f_u.append(sf)
        f_u.append(af.unsqueeze(1))
        f_u.append(ef.unsqueeze(1))
        f_u.append(gf.unsqueeze(1))
        f_u.append(attf.unsqueeze(1))
        f_u.append(sf.unsqueeze(1))

        # f_u = torch.stack(f_u)
        f_u = torch.cat(f_u, dim=1)
        # print('f_u:', f_u)  # b,n,t,d,c
        # print('f_u.shape:', f_u.shape)  # b,n,t,d,c
        f_v = f_u.mean(dim=-2)

        # MEFL
        f_e = self.edge_extractor(f_u, af)
        f_e = f_e.mean(dim=-2)
        f_v, f_e = self.gnn(f_v, f_e)
        # print('f_v.shape, f_e.shape:', f_v.shape, f_e.shape)  # [8, 5, 8, 2048] [8, 25, 8, 2048]
        b, n, t, c = f_v.shape
        sc = self.sc
        sc = sc.unsqueeze(-2).repeat(1, 1, 1, t, 1)
        sc = self.relu(sc)
        sc = F.normalize(sc, p=2, dim=-1)
        cl = F.normalize(f_v, p=2, dim=-1)
        cl = (cl * sc.view(1, n, t, c)).sum(dim=-1, keepdim=False)
        cl_edge = self.edge_fc(f_e)
        return cl, cl_edge


class Build_Graph(nn.Module):
    def __init__(self, num_classes=5, backbone=''):
        super(Build_Graph, self).__init__()
        self.out_channels = 2048
        self.num_class = num_classes
        # if 'transformer' in backbone:
        #     if backbone == 'swin_transformer_tiny':
        #         self.backbone = swin_transformer_tiny()
        #     elif backbone == 'swin_transformer_small':
        #         self.backbone = swin_transformer_small()
        #     else:
        #         self.backbone = swin_transformer_base()
        #     self.in_channels = self.backbone.num_features
        #     self.out_channels = self.in_channels // 2
        #     self.backbone.head = None
        #
        # elif 'resnet' in backbone:
        #     if backbone == 'resnet18':
        #         self.backbone = resnet18()
        #     elif backbone == 'resnet101':
        #         self.backbone = resnet101()
        #     else:
        #         self.backbone = resnet50()
        #     self.in_channels = self.backbone.fc.weight.shape[1]
        #     self.out_channels = self.in_channels // 4
        #     self.backbone.fc = None
        # else:
        #     raise Exception("Error: wrong backbone name: ", backbone)
        #
        # self.global_linear = LinearBlock(self.in_channels, self.out_channels)
        self.action = Action(self.out_channels, num_classes)

    def forward(self, af, ef, gf, attf, sf):
        # x: b d c
        # x = self.backbone(x)
        # x = self.global_linear(x)
        cl, cl_edge = self.action(af, ef, gf, attf, sf)
        return cl, cl_edge
