from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm 
import json
from plyfile import PlyData, PlyElement

def get_segmentation_classes(root):
    catfile = os.path.join(root, 'synsetoffset2category.txt')
    cat = {}
    meta = {}

    with open(catfile, 'r') as f:
        for line in f:
            ls = line.strip().split()
            cat[ls[0]] = ls[1]

    for item in cat:
        dir_seg = os.path.join(root, cat[item], 'points_label')
        dir_point = os.path.join(root, cat[item], 'points')
        fns = sorted(os.listdir(dir_point))
        meta[item] = []
        for fn in fns:
            token = (os.path.splitext(os.path.basename(fn))[0])
            meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg')))
    
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'w') as f:
        for item in cat:
            datapath = []
            num_seg_classes = 0
            for fn in meta[item]:
                datapath.append((item, fn[0], fn[1]))

            for i in tqdm(range(len(datapath))):
                l = len(np.unique(np.loadtxt(datapath[i][-1]).astype(np.uint8)))
                if l > num_seg_classes:
                    num_seg_classes = l

            print("category {} num segmentation classes {}".format(item, num_seg_classes))
            f.write("{}\t{}\n".format(item, num_seg_classes))

def gen_modelnet_id(root):
    classes = []
    with open(os.path.join(root, 'train.txt'), 'r') as f:
        for line in f:
            classes.append(line.strip().split('/')[0])
    classes = np.unique(classes)
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/modelnet_id.txt'), 'w') as f:
        for i in range(len(classes)):
            f.write('{}\t{}\n'.format(classes[i], i))

class ShapeNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 classification=False,
                 class_choice=None,
                 split='train', #默认是训练模式
                 data_augmentation=True):
        #初始化各种参数
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.data_augmentation = data_augmentation
        self.classification = classification
        self.seg_classes = {}
        
        #只读模式打开self.catfile指向的文件
        with open(self.catfile, 'r') as f:
            for line in f:
                #把每一行文字，转为列表，strip()是删除换行符
                ls = line.strip().split()
                #ls[0]是模型名称，ls[1]是对应的数据文件夹名称
                #'Airplane': 02691156
                self.cat[ls[0]] = ls[1]
        #print(self.cat)
        #筛选出指定的类
        #print(class_choice)
        if class_choice[0] is not None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}
        #02691156: Airplane
        self.id2cat = {v: k for k, v in self.cat.items()}

        self.meta = {}
        #获取不同模式对应的文件列表
        splitfile = os.path.join(self.root, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
        #from IPython import embed; embed()
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []

        for file in filelist:
            _, category, uuid = file.split('/')
            if category in self.cat.values():
                #Airplane:('路径/02691156/points/1a04e3eab45ca15dd86060f189eb133.pts',
                #                     '路径/02691156/points_label/1a04e3eab45ca15dd86060f189eb133.seg')
                #模型名称:('某角度的点云路径','对应的label路径')
                self.meta[self.id2cat[category]].append((os.path.join(self.root, category, 'points', uuid+'.pts'),
                                        os.path.join(self.root, category, 'points_label', uuid+'.seg')))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                #('模型名称'，'某角度的点云路径','对应的label路径')
                self.datapath.append((item, fn[0], fn[1]))
        print(sorted(self.cat))
        print(len(self.cat))
        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        print(self.classes)
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                #'Airplane':4             模型:分割数量ground_truth
                self.seg_classes[ls[0]] = int(ls[1])
        self.num_seg_classes = self.seg_classes[list(self.cat.keys())[0]]
        print(self.seg_classes, self.num_seg_classes)

    def __getitem__(self, index):
        """根据给定的index，返回对应的样本数据+标签
                1.从原始点云中随机抽取一定数量的点
                2.将点云均值点作为新的坐标系原点，将抽样点云进行平移
                3.找到最远点距离，利用其对重抽样点云坐标进行归一化
        """
        #直接按顺序读[('模型名称'，'某角度的点云路径','对应的label路径')...]
        fn = self.datapath[index]
        #cls是什么意思？搞不明白，打印出来是排序后的序号，但是有什么作用呢？
        cls = self.classes[self.datapath[index][0]]
        #读取点云数据，以np.float32形式
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        #分割的ground_truth,里面是对应点云数据中，每个点所属的类的标签
        seg = np.loadtxt(fn[2]).astype(np.int64)
        #print(point_set.shape, seg.shape)

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        #resample，行(点)按照choice抽取，列（三维坐标）的话，全都要
        point_set = point_set[choice, :]
        #求出点集的均值中心，然后将所有点的坐标转换到以均值中心为原点
        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        #找出点云中距离均值中心最远点的距离
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        #然后进行缩放，归一化
        point_set = point_set / dist #scale


        #是否进行数据增强
        if self.data_augmentation:
            #随机从0~pi/2中抽取一个角度
            theta = np.random.uniform(0,np.pi*2)
            #print(theta)
            #构造旋转矩阵
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            #绕着y轴旋转theta角度
            point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation 对数据进行随机角度的旋转
            point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter 对数据做随机抖动，添加随机噪声

        seg = seg[choice]
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        if self.classification:
            return point_set, cls
        else:
            return point_set, seg

    def __len__(self):
        return len(self.datapath)

class ModelNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        self.fns = []
        with open(os.path.join(root, '{}.txt'.format(self.split)), 'r') as f:
            for line in f:
                self.fns.append(line.strip())

        self.cat = {}
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/modelnet_id.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = int(ls[1])

        print(self.cat)
        self.classes = list(self.cat.keys())

    def __getitem__(self, index):
        fn = self.fns[index]
        cls = self.cat[fn.split('/')[0]]
        with open(os.path.join(self.root, fn), 'rb') as f:
            plydata = PlyData.read(f)
        pts = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
        choice = np.random.choice(len(pts), self.npoints, replace=True)
        point_set = pts[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        point_set = torch.from_numpy(point_set.astype(np.float32))
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        return point_set, cls


    def __len__(self):
        return len(self.fns)

if __name__ == '__main__':
    dataset = sys.argv[1]
    datapath = sys.argv[2]

    if dataset == 'shapenet':
        d = ShapeNetDataset(root = datapath, class_choice = ['Chair'])
        print(len(d))
        ps, seg = d[0]
        print(ps.size(), ps.type(), seg.size(),seg.type())

        d = ShapeNetDataset(root = datapath, classification = True)
        print(len(d))
        ps, cls = d[0]
        print(ps.size(), ps.type(), cls.size(),cls.type())
        # get_segmentation_classes(datapath)

    if dataset == 'modelnet':
        gen_modelnet_id(datapath)
        d = ModelNetDataset(root=datapath)
        print(len(d))
        print(d[0])

