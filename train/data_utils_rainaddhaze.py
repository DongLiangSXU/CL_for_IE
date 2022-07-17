import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import os,sys
sys.path.append('../data_loader')
sys.path.append('../../../../Dark')
import numpy as np
import torch
import random
from PIL import Image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from metrics import *
from option import opt
BS=opt.bs
print(BS)
crop_size='whole_img'
if opt.crop:
    crop_size=opt.crop_size
from PIL import Image as pi


def get_target(tfile):

    small_t_np = np.asarray(tfile) / 255
    small_t_np = small_t_np[:, :, 0]
    flag = np.min(small_t_np)
    renumber = 0
    if flag <= 1/3:
        renumber = 2
    elif flag <= 2/3:
        renumber = 1
    elif flag <= 1:
        renumber = 0

    return renumber


# 1_1_0.90179.png 1_1.png
class Rain_Dataset(data.Dataset):
    def __init__(self,path,train,ishaze=False,size=crop_size,format='.png'):
        super(Rain_Dataset,self).__init__()
        self.size=size
        print('crop size',size)
        self.train=train
        self.ishaze = ishaze
        self.format=format
        self.rain_imgs_dir=os.listdir(os.path.join(path, '../../../../Dark/cvpr2023/lowq'))
        self.rain_imgs=[os.path.join(path, '../../../../Dark/cvpr2023/lowq', img) for img in self.rain_imgs_dir]
        self.clear_dir=os.path.join(path,'clear')


    def __getitem__(self, index):

        rain=Image.open(self.rain_imgs[index])

        if isinstance(self.size,int):
            while rain.size[0]<self.size or rain.size[1]<self.size :
                index=random.randint(0,len(self.rain_imgs))
                rain=Image.open(self.rain_imgs[index])

        rainfullpath=self.rain_imgs[index]


        clear_name=rainfullpath.split('/')[-1]
        if self.ishaze:
            clear_name = clear_name.split('_')[0]+'.png'
        clear = Image.open(os.path.join(self.clear_dir, clear_name))




        if not isinstance(self.size,str):
            i,j,h,w=tfs.RandomCrop.get_params(rain,output_size=(self.size,self.size))
            rain=FF.crop(rain,i,j,h,w)
            clear=FF.crop(clear,i,j,h,w)
            # rain = rain.resize((self.size, self.size), pi.ANTIALIAS)
            # clear = clear.resize((self.size, self.size), pi.ANTIALIAS)



        rain,clear=self.augData(rain.convert("RGB") ,clear.convert("RGB"))
        return rain,clear,clear_name


    def augData(self,data,target):
        if self.train:
            rand_hor=random.randint(0,1)
            rand_rot=random.randint(0,3)
            data=tfs.RandomHorizontalFlip(rand_hor)(data)
            target=tfs.RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data=FF.rotate(data,90*rand_rot)
                target=FF.rotate(target,90*rand_rot)

        data=tfs.ToTensor()(data)
        target=tfs.ToTensor()(target)
        return data,target


    def __len__(self):
        return len(self.rain_imgs)

import os
pwd=os.getcwd()
print(pwd)
path='/home/lyd16/PycharmProjects/wangbin_Project/data/Rain200H/crop/train'#path to your 'data' folder
testpath = '/home/lyd16/PycharmProjects/wangbin_Project/data/unsup_test/'
hazypath = '/home/lyd16/PycharmProjects/wangbin_Project/data/RESIDE_s'

Rain_train_loader=DataLoader(dataset=Rain_Dataset(path,train=True,size=crop_size),batch_size=BS,shuffle=True)
LOW_train_loader = DataLoader(dataset=Rain_Dataset(hazypath,ishaze=True,train=True,size=crop_size),batch_size=BS,shuffle=True)
ALL_test_loader=DataLoader(dataset=Rain_Dataset(testpath,train=False,size='whole img'),batch_size=1,shuffle=False)

# OTS_train_loader=DataLoader(dataset=RESIDE_Dataset(path+'/RESIDE/OTS',train=True,format='.jpg'),batch_size=BS,shuffle=True)
# OTS_test_loader=DataLoader(dataset=RESIDE_Dataset(path+'/RESIDE/SOTS/outdoor',train=False,size='whole img',format='.png'),batch_size=1,shuffle=False)

if __name__ == "__main__":
    pass

