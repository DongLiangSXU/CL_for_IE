import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import os,sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import random
from PIL import Image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from metrics import *
from option import opt
import torchvision.utils as vutils

BS=opt.bs
print(BS)
crop_size='whole_img'
if opt.crop:
    crop_size=opt.crop_size
from PIL import Image as pi

# 1_1_0.90179.png 1_1.png
class RAIN_Dataset(data.Dataset):
    def __init__(self,path,train,ishaze=False,islow=False,size=crop_size,format='.png'):
        super(RAIN_Dataset,self).__init__()
        self.size=size
        print('crop size',size)
        self.train=train
        self.format=format
        self.ishaze=ishaze
        self.islow = islow
        self.lowq_imgs_dir=os.listdir(os.path.join(path,'lowq'))
        self.lowq_imgs=[os.path.join(path,'lowq',img) for img in self.lowq_imgs_dir]
        self.clear_dir=os.path.join(path,'clear')
        self.peddepthmask = False


    def __getitem__(self, index):
        # print(index,len(self.haze_imgs))

        rain=Image.open(self.lowq_imgs[index])

        if isinstance(self.size,int):
            while rain.size[0]<self.size or rain.size[1]<self.size :
                index=random.randint(0,len(self.lowq_imgs)-1)
                rain=Image.open(self.lowq_imgs[index])

        rainfullpath=self.lowq_imgs[index]


        clear_name=rainfullpath.split('/')[-1]
        if self.ishaze:
            clear_name = clear_name.split('_')[0]+'.png'
        clear = Image.open(os.path.join(self.clear_dir, clear_name))

        if not isinstance(self.size,str):
            i,j,h,w=tfs.RandomCrop.get_params(rain,output_size=(self.size,self.size))
            rain=FF.crop(rain,i,j,h,w)
            clear=FF.crop(clear,i,j,h,w)
        else:
            h,w = rain.size[1],rain.size[0]
            tempv = 1
            rain=FF.crop(rain,0,0,(h//tempv)*tempv,(w//tempv)*tempv)
            clear=FF.crop(clear,0,0,(h//tempv)*tempv,(w//tempv)*tempv)



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

        return len(self.lowq_imgs)

import os
pwd=os.getcwd()
print(pwd)

# rain_train = '/home/lyd16/PycharmProjects/wangbin_Project/data/Rain200H/train/'
# rain_test = '/home/lyd16/PycharmProjects/wangbin_Project/data/Rain200H/test/'
#
# ITS_train_loader=DataLoader(dataset=RAIN_Dataset(rain_train,train=True,size=crop_size),batch_size=BS,shuffle=True)
# ITS_test_loader=DataLoader(dataset=RAIN_Dataset(rain_test,train=False,size='whole img'),batch_size=1,shuffle=False)

rainpath='/home/lyd16/PycharmProjects/wangbin_Project/data/Rain200H/crop/train'#path to your 'data' folder
testpath = '/home/lyd16/PycharmProjects/wangbin_Project/data/unsup_test/'
hazypath = '/home/lyd16/PycharmProjects/wangbin_Project/data/RESIDE_s'
lolpath = '/home/lyd16/PycharmProjects/wangbin_Project/data/LOL/our485'

Rain_train_loader=DataLoader(dataset=RAIN_Dataset(rainpath,train=True,size=crop_size),batch_size=BS,shuffle=True)
Haze_train_loader = DataLoader(dataset=RAIN_Dataset(hazypath,ishaze=True,train=True,size=crop_size),batch_size=BS,shuffle=True)
LL_train_loader = DataLoader(dataset=RAIN_Dataset(lolpath,islow=True,train=True,size=crop_size),batch_size=BS,shuffle=True)
ALL_test_loader=DataLoader(dataset=RAIN_Dataset(testpath,train=False,size='whole img'),batch_size=1,shuffle=False)



if __name__ == "__main__":
    pass
