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
from models import *
from option import opt
import torchvision.utils as vutils
# depthnet = NormDepth(iskt=False)
# depthnet = depthnet.to(opt.device)
# depthnet = depthnet.eval()
BS=opt.bs
print(BS)
crop_size='whole_img'
if opt.crop:
    crop_size=opt.crop_size
from PIL import Image as pi

# 1_1_0.90179.png 1_1.png
class LOL_Dataset(data.Dataset):
    def __init__(self,path,train,size=crop_size,format='.png'):
        super(LOL_Dataset,self).__init__()
        self.size=size
        print('crop size',size)
        self.train=train
        self.format=format
        self.lowq_imgs_dir=os.listdir(os.path.join(path,'low'))
        self.lowq_imgs=[os.path.join(path,'low',img) for img in self.lowq_imgs_dir]
        self.clear_dir=os.path.join(path,'high')



    def __getitem__(self, index):
        # print(index,len(self.haze_imgs))

        lowmap=Image.open(self.lowq_imgs[index])

        if isinstance(self.size,int):
            while lowmap.size[0]<self.size or lowmap.size[1]<self.size :
                index=random.randint(0,len(self.lowq_imgs)-1)
                rain=Image.open(self.lowq_imgs[index])

        lowfullpath=self.lowq_imgs[index]

        clear_name=lowfullpath.split('/')[-1]

        clear = Image.open(os.path.join(self.clear_dir, clear_name))

        if not isinstance(self.size,str):
            i,j,h,w=tfs.RandomCrop.get_params(lowmap,output_size=(self.size,self.size))
            lowmap=FF.crop(lowmap,i,j,h,w)
            clear=FF.crop(clear,i,j,h,w)


        low,high=self.augData(lowmap.convert("RGB") ,clear.convert("RGB"))

        return low,high,clear_name


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


testpath = '/image-text-match/OVD/wangbin10/CL_for_IE-main/datasets/LOL/eval15/'
lolpath = '/image-text-match/OVD/wangbin10/CL_for_IE-main/datasets/LOL/our485/'

LL_train_loader = DataLoader(dataset=LOL_Dataset(lolpath,train=True,size=crop_size),batch_size=BS,shuffle=True)
LL_test_loader=DataLoader(dataset=LOL_Dataset(testpath,train=False,size='whole img'),batch_size=1,shuffle=False)



if __name__ == "__main__":
    pass
