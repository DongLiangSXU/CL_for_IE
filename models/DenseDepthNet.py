import torch
import torch.nn as nn
import torchvision
import scipy.misc as sc
import numpy as np
from torchvision import models
from torchvision import datasets, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
import h5py
import argparse
import matplotlib
import numpy as np

import skimage.measure
windowpath = '/home/ubuntu/PycharmProjects/DTTD_master/bestmodel/usemodel/'

class Upproject(nn.Module):
   def __init__(self,in_channels,nf):
       super(Upproject,self).__init__()
       # self.upsample = F.upsample_bilinear
       self.upsample = F.interpolate
       self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=nf,stride=1,kernel_size=3,padding=1,bias=True)
       # self.relu = nn.LeakyReLU(0.2,inplace=True)
       self.conv2 = nn.Conv2d(in_channels=nf,out_channels=nf,kernel_size=3,stride=1,padding=1,bias=True)
       self.relu2 = nn.LeakyReLU(0.2)


   def forward(self, input, to_cat):
        shape_out = input.data.size()
        shape_out = shape_out[2:4]
        # print(shape_out)
        x1 = self.upsample(input,size=(shape_out[0]*2,shape_out[1]*2),mode='bilinear',align_corners=True)
        x1 = torch.cat([x1, to_cat], dim=1)
        # x1 = self.upsample(x1,size=(shape_out[0]*2,shape_out[1]*2))
        x2 = self.conv1(x1)
        # x2 = self.relu(x2)
        x3 = self.conv2(x2)
        x3 = self.relu2(x3)
        return x3

class DenseNet_pytorch(nn.Module):
    def __init__(self,in_channels,out_channels,):
        super(DenseNet_pytorch, self).__init__()
        # self.model = models.resnet34(pretrained=False)
        self.model = models.densenet169(pretrained=False)
        # self.model.load_state_dict(torch.load(Windows_filepath+'densenet169-b2777c0a.pth'))
        self.conv0 = self.model.features.conv0
        self.norm0 = self.model.features.norm0
        self.relu0 = self.model.features.relu0
        self.pool0 = self.model.features.pool0

        ############# Block1-down 64-64  ##############
        self.dense_block1 = self.model.features.denseblock1
        self.trans_block1 = self.model.features.transition1

        ############# Block2-down 32-32  ##############
        self.dense_block2 = self.model.features.denseblock2
        self.trans_block2 = self.model.features.transition2

        ############# Block3-down  16-16 ##############
        self.dense_block3 = self.model.features.denseblock3
        self.trans_block3 = self.model.features.transition3

        ############# Block4-down  16-16 ##############
        self.dense_block4 = self.model.features.denseblock4


        self.model_out = self.model.features.norm5
        self.model_relu = F.relu

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_out_channels = 1664
        self.midconv = nn.Conv2d(in_channels=self.model_out_channels,out_channels=self.model_out_channels,kernel_size=1,stride=1,padding=0,bias=True)
        # self.midrelu = nn.LeakyReLU(0.2,inplace=True)
        # 输出：1664
        self.up1 = Upproject(1920,832)
        self.up2 = Upproject(960,416)
        self.up3 = Upproject(480,208)
        self.up4 = Upproject(272, 104)
        self.finalconv = nn.Conv2d(in_channels=104,out_channels=1,kernel_size=3,stride=1,padding=1,bias=True)


    def forward(self, x):
        tempx = x
        shape_out = x.data.size()
        shape_out = shape_out[2:4]

        x0 = self.relu0(self.norm0(self.conv0(x)))
        tx1 =x0
        x0=self.pool0(x0)
        tx2 = x0
        x1 = self.trans_block1(self.dense_block1(x0))
        tx3 = x1
        x2 = self.trans_block2(self.dense_block2(x1))
        tx4 =x2

        x3 = self.trans_block3(self.dense_block3(x2))

        x4 = self.dense_block4(x3)
        finnalout = self.model_out(x4)
        finnalout = self.model_relu(finnalout)

        mid = self.midconv(finnalout)
        # output:640*8*8
        up1 = self.up1(mid, tx4)
        # output:256*16*16
        up2 = self.up2(up1, tx3)
        # output:128*40*40
        up3 = self.up3(up2, tx2)
        # output:64*80*80
        up4 = self.up4(up3, tx1)

        result = self.finalconv(up4)

        return result

def nyu_resize(img, resolution=512, padding=6):
    from skimage.transform import resize
    return resize(img, (resolution, resolution), preserve_range=True, mode='reflect', anti_aliasing=True )

def DepthNorm(x, maxDepth):
    return maxDepth / x

def test():

    save_path = '/home/ubuntu/PycharmProjects/dataset/data/fcval/'
    res_path = '/home/ubuntu/PycharmProjects/DTTD_master/depth_eve/kt/nohaze/'

    # name = save_path + '555.h5'
    score_all = 0
    cout = 0
    for i in range(100):
        num = str(i)
        name = save_path+num+'.h5'
        cout = cout+1
        print(cout)
        f = h5py.File(name, mode='r')
        haze = f['haze'][:]
        gt_tran = f['tran'][:]
        gt = f['gt'][:]


        depth = f['depth'][:]
        # depth = nyu_resize(depth,128)
        # sc.imsave(res_path+'temp.png',gt)
        # gt = sc.imread(res_path+'temp.png')/256


        input = transforms.ToTensor()(gt)
        input = input.float()
        input = torch.unsqueeze(input, 0)
        input = input.cuda()


        net = DenseNet_pytorch(3,3)
        net.cuda()

        # net.load_state_dict(torch.load('/home/ubuntu/PycharmProjects/DTTD_master/bestmodel/newDensedepth_NET1.pth'))
        net.load_state_dict(torch.load(windowpath+'my.pth'))
        # net.load_state_dict(torch.load('/home/ubuntu/PycharmProjects/DTTD_master/train_val/depth_sample/' + 'DenseDepthFUN_KITTI0.pth'))

        net.eval()
        out= net(input)

        out = out.cpu().detach().numpy()
        out = out[0, 0, :, :]
        result = out
        # result = out.detach().numpy()
        # result = np.swapaxes(result, 0, 2)
        # result = np.swapaxes(result, 0, 1)
        # result = result[:, :, 0]
        # result = DepthNorm(result,1000)
        result = np.clip(DepthNorm(result, maxDepth=1000), 10, 1000) / 1000
        result = result*10
        # print(np.max(result))
        result = (result-np.min(result))/(np.max(result)-np.min(result))

        depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.imshow(result)
        # plt.subplot(1, 2, 2)
        # plt.imshow(depth)
        # plt.show()

        score = skimage.measure.compare_ssim(result,depth)
        score_all+=score
        print(score_all/cout)
    print(score_all/100)


def eve():

    save_path = '/home/ubuntu/PycharmProjects/DTTD_master/kt97/'
    savepath = '/home/ubuntu/PycharmProjects/DTTD_master/resultsfrom3method/dcpdn/'
    # res_path = '/home/ubuntu/PycharmProjects/DTTD_master/depth_eve/kt/fun/'
    res_path = '/home/ubuntu/PycharmProjects/DTTD_master/depth_eve/3method/dcp/'
    c =0
    k = 0
    for i in range(1956):
        i=i+1



        num = str(c)

        coun = str(i)

        print(coun)
        name = save_path+coun+'.h5'
        f = h5py.File(name, mode='r')
        # gt = f['gt'][:]
        # haze = f['haze'][:]
        # gtdepth = f['depth'][:]
        # spdepth = f['sdepth'][:]
        spo = f['sodepth'][:]

        name = savepath+coun+'.h5'
        f1 = h5py.File(name, mode='r')
        haze = f1['dehaze'][:]




        input = transforms.ToTensor()(haze)
        input = input.float()
        input = torch.unsqueeze(input, 0)
        input = input.cuda()


        net = DenseNet_pytorch(3,3)
        net.cuda()

        # net.load_state_dict(torch.load('/home/ubuntu/PycharmProjects/DTTD_master/train_val/depth_sample/DenseDepthFUN_KITTI1.pth'))
        net.load_state_dict(
            torch.load('/home/ubuntu/PycharmProjects/DTTD_master/bestmodel/usemodel/kitti.pth'))

        net.eval()

        out= net(input)

        out = out.cpu().detach().numpy()
        out = out[0, 0, :, :]

        result = np.clip(DepthNorm(out, maxDepth=8000), 10, 8000) / 8000

        sname = res_path+num+'.h5'

        save = h5py.File(sname, mode='w')
        save.create_dataset('depth',data=result)
        # save.create_dataset('gt_depth', data=gtdepth)
        # save.create_dataset('s_depth', data=spdepth)
        save.create_dataset('sodepth',data=spo)
        c = c+1
        # k = k+1



def scale_up(scale, images):
    from skimage.transform import resize
    scaled = []

    for i in range(len(images)):
        img = images[i]
        output_shape = (scale * img.shape[0], scale * img.shape[1])
        scaled.append(resize(img, output_shape, order=1, preserve_range=True, mode='reflect', anti_aliasing=True))

    return np.stack(scaled)

def evenyu():

    save_path = '/home/ubuntu/PycharmProjects/dataset/data/eigntest/'
    res_path = '/home/ubuntu/PycharmProjects/DTTD_master/depth_eve/nyu/fun/'

    for i in range(100):
        # i = i+30
        num = str(i)
        print(num)
        name = save_path+num+'.h5'
        f = h5py.File(name, mode='r')
        gt = f['gt'][:]
        haze = f['haze'][:]
        gtdepth = f['orgindepth'][:]
        d1 = f['depth1'][:]
        trans = f['tran'][:]
        ato = f['ato'][:]


        input = transforms.ToTensor()(haze)
        input = input.float()
        input = torch.unsqueeze(input, 0)
        input = input.cuda()


        net = DenseNet_pytorch(3,3)
        net.cuda()

        net.load_state_dict(torch.load('/home/ubuntu/PycharmProjects/DTTD_master/DIGDdepth0.pth'))
        # net.load_state_dict(
        #     torch.load('/home/ubuntu/PycharmProjects/DTTD_master/bestmodel/usemodel/my.pth'))

        net.eval()

        out= net(input)

        out = out.cpu().detach().numpy()
        out = out[0, 0, :, :]


        out = np.clip(DepthNorm(out, maxDepth=1000), 10, 1000) / 1000

        predepth = np.zeros([1, 256, 256])
        predepth[0, :, :] = out
        predepth = scale_up(2, predepth)
        out = predepth[0, :, :]
        out = out * 10
        print(np.max(out))
        print(np.max(gtdepth))

        sname = res_path+num+'.h5'

        save = h5py.File(sname, mode='w')
        save.create_dataset('depth',data=out)
        save.create_dataset('gt_depth', data=gtdepth)

        # save.create_dataset('s_depth', data=spdepth)


class NormDepth(nn.Module):
    def __init__(self,iskt=False,isfun=False):
        super(NormDepth, self).__init__()
        self.upsample = F.upsample_bilinear
        self.depth_es = DenseNet_pytorch(3, 3)

        # if iskt:
        self.isfun = isfun

        if iskt:
            self.depth_es.load_state_dict(torch.load('/home/lyd16/PycharmProjects/wangbin_Project/PDLD_master/usemodel/kitti.pth'))
            if self.isfun:
                self.depth_es.load_state_dict(torch.load('DenseDepthFUN_KITTI0.pth'))
                print('----------------------')
        # print('-------------------------------')
        else:
            self.depth_es.load_state_dict(torch.load('/home/lyd16/PycharmProjects/wangbin_Project/PDLD_master/usemodel/my.pth'))
        # self.depth_es.eval()
        self.threeshold = nn.Threshold(1, 1)


    def forward(self, input):


        depth = self.depth_es(input)

        shape = input.data.size()
        shape = shape[2:4]
        depth = self.upsample(depth, size=shape)


        depth = self.threeshold(depth)
        depth = 1 / depth


        return depth

