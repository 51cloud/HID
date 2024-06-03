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
        # print('res:', res, res.shape)  # torch.Size([2, 5, 8, 2048]) 非 NAN  # 1.5784e-02,  1.1932e+00,  8.0635e-02,  ...,  1.4120e-01,
        # print('Vix:', Vix, Vix.shape)  # torch.Size([2, 5, 8, 2048]) 非 nan  # 1.2863e-01,  6.5604e-01, -4.0832e-01,  ..., -5.0598e-01,
        # print('Vjx:', Vjx, Vjx.shape)  # torch.Size([2, 5, 8, 2048]) 非 nan  # 4.4875e-01, -3.3083e-01,  4.4048e-01,  ..., -5.7026e-02,
        # print('e:', e, e.shape)  # torch.Size([2, 25, 8, 2048]) 非nan  # -1.8385e+00,  3.4862e+00,  1.7302e-01,  ...,  1.1704e+00,
        # Vix = Vix.permute(1, 0, 2, 3)
        # Vjx = Vjx.permute(1, 0, 2, 3)
        # print('end:', end.shape)  # [25, 5]
        # print('Vix:', Vix.shape)  # [8, 5, 8, 2048]
        # print('start:', start.shape)  # [25, 5]
        # print('Vjx:', Vjx.shape)  # [8, 5, 8, 2048]
        # print('e:', e.shape)  # [8, 25, 8, 2048]

        # edge = edge + self.act(self.bne1(
        #     torch.einsum('ev, bvc -> bec', (end, Vix)).permute(1, 0, 2, 3) + torch.einsum('ev, bvc -> bec', (start, Vjx)).permute(1, 0, 2, 3) + e))  # E x d_out
        # print('einsum:', torch.einsum('ev, bvtc -> betc', (end, Vix)) + torch.einsum('ev, bvtc -> betc', (start, Vjx)) + e)
        # print('einsum_bn:', self.bne1(
        #     torch.einsum('ev, bvtc -> betc', (end, Vix)) + torch.einsum('ev, bvtc -> betc', (start, Vjx)) + e))  # nan
        # print('einsum_act', self.act(self.bne1(
        #     torch.einsum('ev, bvtc -> betc', (end, Vix)) + torch.einsum('ev, bvtc -> betc', (start, Vjx)) + e)))

        # edge = edge + self.act(self.bne1(
        #     torch.einsum('ev, bvtc -> betc', (end, Vix)) + torch.einsum('ev, bvtc -> betc', (start, Vjx)) + e))  # E x d_out
        edge1 = torch.einsum('ev, bvtc -> betc', (end, Vix)) + torch.einsum('ev, bvtc -> betc', (start, Vjx)) + e
        # print('edge1:', edge1, edge1.shape)  # 2.0876e+30,  1.3555e+30, -9.4971e+29,  ..., -2.9862e+30,
        edge1 = self.bne1(edge1)
        edge1 = torch.where(torch.isnan(edge1),
                                 torch.tensor(0.0, dtype=edge1.dtype, device=edge1.device), edge1)

        edge1 = self.act(edge1)
        edge = edge + edge1
        # print('edge:', edge, edge.shape)  # torch.Size([2, 25, 8, 2048] 有NAN  3.9328e-01, -5.0946e-01,  4.5619e-01,  ...,  3.8598e-01,
        e = self.sigmoid(edge)
        # print('e::', e, e.shape)  # [0.5971, 0.3753, 0.6121,  ..., 0.5953, 0.8062, 0.5467]
        b, _, t, c = e.shape
        e = e.view(b, self.num_classes, self.num_classes, t, c)
        # print('e1:', e, e.shape)  # [0.9128, 0.5782, 0.4992,  ..., 0.3489, 0.5390, 0.4477]
        e = self.softmax(e)
        # print('e2:', e, e.shape)  # [0.2113, 0.2148, 0.1586,  ..., 0.1671, 0.2693, 0.1495]
        e = e.view(b, -1, t, c)
        # print('e3:', e, e.shape)  # [0.2353, 0.1538, 0.1985,  ..., 0.1794, 0.1539, 0.1484]

        Ujx = self.V1(x)  # V x H_out
        # print('Ujx:', Ujx, Ujx.shape)  # 4.8079e+29, -4.4709e+28,  1.2543e+29,  ...,  2.1132e+29
        # Ujx = Ujx.permute(1, 0, 2, 3)
        Ujx = torch.einsum('ev, bvtc -> betc', (start, Ujx))  # E x H_out
        # print('Ujx1:', Ujx, Ujx.shape)  # 2.8378e+30, -2.1079e+29,  1.9573e+30,  ...,  8.3779e+29
        Uix = self.U1(x)  # V x H_out
        # print('Uix:', Uix, Uix.shape)  # 2.0028e+29,  2.6633e+29,  1.5088e+29,  ...,  4.4597e+29
        # Uix = Uix.permute(1, 0, 2, 3)
        # e = e.permute(1, 0, 2, 3)
        x = Uix + torch.einsum('ve, betc -> bvtc', (end.t(), e * Ujx)) / self.num_classes  # V x H_out
        # print('x:', x, x.shape)  # 1.7767e+30, -2.2196e+29, -7.4000e+29,  ..., -7.0435e+29
        x1 = self.bnv1(x)
        # print('x1:', x1, x1.shape)  # nan, nan, nan,  ..., nan, nan, nan
        x1 = torch.where(torch.isnan(x1),
                            torch.tensor(0.0, dtype=x1.dtype, device=x1.device), x1)
        x = self.act(res + x1)
        # print('x2:', x, x.shape)  # 1.5784e-02, 1.1932e+00, 8.0635e-02,  ..., 1.4120e-01
        res = x

        # GNN Layer 2:
        Vix = self.A2(x)  # V x d_out
        Vjx = self.B2(x)  # V x d_out
        e = self.E2(edge)  # E x d_out
        # edge = edge + self.act(self.bne2(
        #     torch.einsum('ev, bvtc -> betc', (end, Vix)) + torch.einsum('ev, bvtc -> betc', (start, Vjx)) + e))  # E x d_out
        edge2 = torch.einsum('ev, bvtc -> betc', (end, Vix)) + torch.einsum('ev, bvtc -> betc', (start, Vjx)) + e
        edge2 = self.bne2(edge2)
        edge2 = torch.where(torch.isnan(edge2),
                            torch.tensor(0.0, dtype=edge2.dtype, device=edge2.device), edge2)

        edge2 = self.act(edge2)
        edge = edge + edge2

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
        # 定义异常值的阈值
        # threshold_lower = 1.0e+7  # 异常值下限阈值
        # threshold_upper = 1.0e+7  # 异常值上限阈值

        # # 将 sf 中小于等于下限阈值的值替换为 0.1
        # sf[sf <= threshold_lower] = 0.01

        # # 将 sf 中大于等于上限阈值的值替换为 0.1
        # sf[sf >= threshold_upper] = 0.01
        # for i, layer in enumerate(self.class_linears):
        #     f_u.append(layer(x).unsqueeze(1))
        # print('af:', af)  # torch.Size([4, 2048, 8, 7, 7])  # 0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 7.4266e-01,
        # print('ef:', ef)  # torch.Size([4, 2048, 8, 64, 64])  # 0.0000e+00,  0.0000e+00,  0.0000e+00,  ..., -3.5516e-04,
        # print('gf:', gf)  # torch.Size([4, 2048, 8, 64, 64])  # -7.0781e-05,  6.1215e-04,  5.5643e-04,  ..., -1.0003e-04
        # print('attf:', attf)  # torch.Size([4, 2048, 8, 64, 64])  #0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00, -4.3506e-12, -3.0423e-12
        # print('sf:', sf)  # torch.Size([4, 2048, 8, 64, 64])  # 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00, 0.0000e+00, -3.6135e+31
        b, c, t, h, w = af.shape
        af = af.view(b, c, t, -1).permute(0, 2, 3, 1).cuda()  # b, t, d, c
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
        # print('f_u:', f_u)  # b,n,t,d,c #  f_u.shape: torch.Size([4, 40, 49, 2048])  # 非nan  # -5.2484e+29,  5.7947e+29, -3.3703e+29,  ..., -7.3745e+29
        # print('f_u.shape:', f_u.shape)  # b,n,t,d,c
        f_v = f_u.mean(dim=-2)
        # 将 sf 中小于等于下限阈值的值替换为 0.1
        # f_v[f_v <= threshold_lower] = 0.01

        # # 将 sf 中大于等于上限阈值的值替换为 0.1
        # f_v[f_v >= threshold_upper] = 0.01
        # print('f_v1:',
        #       f_v)  # 非NAN  #   # 8.8572e+28, -9.5343e+28, -2.6875e+29,  ...,  8.7302e+28, 1.7112e+29,  1.0462e+29
        # MEFL
        # print('f_u:', f_u)  # b,n,t,d,c #  f_u.shape: torch.Size([4, 40, 49, 2048])  # 非nan  # -5.2484e+29,  5.7947e+29, -3.3703e+29,  ..., -7.3745e+29
        # print('f_u.shape:', f_u.shape)  # f_u.shape: torch.Size([1, 5, 8, 4096, 1])
        # print('af:', af)  # b,n,t,d,c #  f_u.shape: torch.Size([4, 40, 49, 2048])  # 非nan  # -5.2484e+29,  5.7947e+29, -3.3703e+29,  ..., -7.3745e+29
        # print('af.shape:', af.shape)  # torch.Size([1, 8, 4096, 1])
        f_e = self.edge_extractor(f_u, af)
        # print('f_e1:', f_e)  # 非NAN  # 2.8457e+00,  8.1892e-01, -6.9378e-02,  ..., -2.1276e-01
        f_e = f_e.mean(dim=-2)
        # print('f_v.shape:', f_v, f_v.shape)  # torch.Size([2, 5, 8, 2048])  # -5.2484e+29,  5.7947e+29, -3.3703e+29,  ..., -7.3745e+29
        # print('f_e.shape:', f_e, f_e.shape)  # torch.Size([2, 25, 8, 2048])  # 1.1828e+00,  2.5511e-01, -9.2326e-01,  ...,  5.3438e-01
        f_v, f_e = self.gnn(f_v, f_e)
        # print('f_v:', f_v) # nan
        # print('f_e:', f_e)
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
        # af = af.cpu().detach().numpy()
        # threshold_lower = 1.0e-5
        # threshold_upper = 1.0e+5
        # nan_indices = np.where(np.isnan(af))
        # lower_indices = np.where(abs(af) <= threshold_lower)
        # upper_indices = np.where(abs(af) >= threshold_upper)
        
        # # 替换异常值为随机数
        # if len(nan_indices[0]) > 0:
        #     af[nan_indices] = np.random.uniform(0.001, 0.01, size=len(nan_indices[0]))

        # if len(lower_indices[0]) > 0:
        #     af[lower_indices] = np.random.uniform(0.001, 0.01, size=len(lower_indices[0]))
        
        # if len(upper_indices[0]) > 0:
        #     af[upper_indices] = np.random.uniform(0.001, 0.01, size=len(upper_indices[0]))
        # af = torch.from_numpy(af).cuda()
        cl, cl_edge = self.action(af, ef, gf, attf, sf)
        return cl, cl_edge
