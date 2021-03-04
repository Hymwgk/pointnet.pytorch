from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    """这里是数据最开始的处理
          用于预测一个三维旋转矩阵
    """
    def __init__(self):
        super(STN3d, self).__init__()
        #1d卷积层 训练阶段输出(batch size,64,2500)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        #训练阶段输出(batch size,128,2500)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        #三个线性层
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        #x.shape()=(32,3,2500)
        batchsize = x.size()[0]
        #先经过三个1d卷积，输出Channel=64
        x = F.relu(self.bn1(self.conv1(x)))
        #输出Channel=128
        x = F.relu(self.bn2(self.conv2(x)))
        #输出Channel=1024的数据  (batch_size,1024,2500)
        x = F.relu(self.bn3(self.conv3(x)))
        #print(x.size())
        # size=(batch_size,1024,1)
        x = torch.max(x, 2, keepdim=True)[0]

        #torch.Size([32, 1024])，每一行都是一个点云的压缩（Channel=1024拓展，维度压缩为1；相当于只留1个点，
        # 该点包含了一帧点云2500个点的综合信息）
        x = x.view(-1, 1024)

        #再利用3个全连接层，压缩点的Channel
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        #最终x的每一行，都是一帧点云的压缩过的FeatureMap，可以认为每帧点云
        # 由一个维度为9的向量表示
        x = self.fc3(x)

        #torch.Size([32, 9]) 这个是为了构建一个3*3单位矩阵
        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        
        if x.is_cuda:
            iden = iden.cuda()
        #为什么要+iden ?
        x = x + iden
        x = x.view(-1, 3, 3)
        #x是什么？作者想让它代表的是一个旋转矩阵
        return x


class STNkd(nn.Module):
    """
    预测一个64*64的矩阵，在特征空间中对
    其中一个FeatureMap做特征变换；目的是将不同输入点云的特征进行对准align
    """
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        #x.size()=(batch_size,4096)
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    """
    这个模块，主要是实现分割之外的所有功能
    """
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        #构造一个旋转矩阵预测网络
        self.stn = STN3d()
        #
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        #一帧点云里的点数
        n_pts = x.size()[2]
        #返回一个预测旋转矩阵
        trans = self.stn(x)
        #将点云转置回到原始的形状;
        # size=(batch_size,2500,3)
        x = x.transpose(2, 1)
        #对batch内的每个点云都做旋转
        x = torch.bmm(x, trans)
        #之后再转置回到(batch_size,3,2500)
        x = x.transpose(2, 1)

        #进行1d卷积,将点云Channel拓展到64
        #shape=(batch_size,64,2500)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        #再次拓展Channel
        #shape=(batch_size,128,2500)
        x = F.relu(self.bn2(self.conv2(x)))
        #再次拓展
        #shape=(batch_size,1024,2500)
        x = self.bn3(self.conv3(x))
        #x.size()=(batch_size,1024,1) 起到的作用实际上就是Maxpool，
        #这个maxpool将2500*3的点云，压缩成为一个维度为1024的向量,在这里实现了顺序无关，maxpool
        #这个向量就代表的是全局的特征向量
        x = torch.max(x, 2, keepdim=True)[0]
        #展开为(batch,1024) 的全局特征向量
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            #x.size()=(batch_size,1024,2500)
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            #每个点的Channel 都拼接局部＋全局特征；构成一个(batch_size,64+1024，2500)的点集合
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat


class PointNetDenseCls(nn.Module):
    """
    PointNet 点云分割网络模型
    """
    def __init__(self, k = 2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform=feature_transform
        #导入一个子模块，将特征变换那里，单独写成一个小的网络
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        #输出x.size=([batch,64+1024=1088,2500]) 压缩Channel
        x, trans, trans_feat = self.feat(x)
        #输出x.size=([batch_size,512, 2500])  压缩Channel
        x = F.relu(self.bn1(self.conv1(x)))
        #输出x.size=(batch_size,256,2500)  压缩Channel
        x = F.relu(self.bn2(self.conv2(x)))
        #继续压缩Channel，x.size=(batch_size,128,2500) 
        x = F.relu(self.bn3(self.conv3(x)))
        #最后一次压缩，x.size=(batch_size,k类,2500)
        x = self.conv4(x)
        #转置后x.size=(batch_size,2500,k)
        x = x.transpose(2,1).contiguous()
        
        #x.view(-1,self.k).size()=(batch_size*2500,4)
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        #重新reshape成为原来的形状(batch_size,2500,k)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feat

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k = 3)
    out, _, _ = seg(sim_data)
    print('seg', out.size())
