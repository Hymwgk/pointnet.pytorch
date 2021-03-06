from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset
from pointnet.model import PointNetDenseCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

#读取外部参数
parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
#设置训练多少轮
parser.add_argument(
    '--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='seg', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--class_choice', type=str, default='Chair', help="class_choice")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=False,
    class_choice=[opt.class_choice])

#创建dataloader
dataloader = torch.utils.data.DataLoader(
    dataset,#设置数据集对象
    batch_size=opt.batchSize,#
    shuffle=True,#设置将会随机打乱样本的顺序
    num_workers=int(opt.workers))#设置使用几个线程去加载数据

test_dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=False,
    class_choice=[opt.class_choice],
    split='test',
    data_augmentation=False)

testdataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))


print(len(dataset), len(test_dataset))
num_classes = dataset.num_seg_classes
#打印出指定的模型，对应的分割数量
print('classes', num_classes)
try:
    #尝试创建一个存放分割结果的文件夹
    os.makedirs(opt.outf)
except OSError:
    pass

blue = lambda x: '\033[94m' + x + '\033[0m'

#在这里，实例化PointNet分割网络
classifier = PointNetDenseCls(k=num_classes, feature_transform=opt.feature_transform)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

#使用Adam优化器，对分割网络中的参数进行优化，初始学习率是0.001，
optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

#将模型参数等等移至GPU中
classifier.cuda()
#计算有多少个batch
num_batch = len(dataset) / opt.batchSize

for epoch in range(opt.nepoch):
    scheduler.step()
    for i, data in enumerate(dataloader, 0):
        #
        points, target = data
        
        #为什么要transpose？
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        #清空模型参数的梯度
        optimizer.zero_grad()
        #将模型设置到训练模式
        classifier = classifier.train()
        #将样本数据tensor送入网络；返回的：
        #pred:   size()=(batchsize, n_pts, self.k) 每行都是一个点对应k个类别的概率
        #trans: 预测出的3*3空间变换矩阵
        #trans_feat:预测出的6*6空间变换矩阵
        pred, trans, trans_feat = classifier(points)
        #size()=(batchsize*n_pts, self.k)=(80000,4)
        pred = pred.view(-1, num_classes)
        #size()=(batchsize*n_pts)=(80000)每个值都是一个点的类别
        target = target.view(-1, 1)[:, 0] - 1
        #print(pred.size(), target.size())
        loss = F.nll_loss(pred, target)
        #为loss函数添加惩罚项（正则化项）
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        #添加惩罚项之后才进行反向传播
        loss.backward()
        #对权重进行更新，对学习率进行衰减
        optimizer.step()
        #按行找每个点的最大概率值，找最大值对应的索引，这个索引就是输出的结果认为是
        pred_choice = pred.data.max(1)[1]
        #
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item()/float(opt.batchSize * 2500)))
        #每隔10次迭代   验证一次
        if i % 10 == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, target = data
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            pred, _, _ = classifier(points)
            pred = pred.view(-1, num_classes)
            target = target.view(-1, 1)[:, 0] - 1
            loss = F.nll_loss(pred, target)
            #求出每个点在k类别中的最大概率，并返回索引
            pred_choice = pred.data.max(1)[1]
            #求出正确分割点的数量
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize * 2500)))

    torch.save(classifier.state_dict(), '%s/seg_model_%s_%d.pth' % (opt.outf, opt.class_choice, epoch))



## benchmark mIOU
shape_ious = []
for i,data in tqdm(enumerate(testdataloader, 0)):
    points, target = data
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(2)[1]

    pred_np = pred_choice.cpu().data.numpy()
    target_np = target.cpu().data.numpy() - 1

    for shape_idx in range(target_np.shape[0]):
        parts = range(num_classes)#np.unique(target_np[shape_idx])
        part_ious = []
        for part in parts:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            if U == 0:
                iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))

print("mIOU for class {}: {}".format(opt.class_choice, np.mean(shape_ious)))