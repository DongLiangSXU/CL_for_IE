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
class Noise_Dataset(data.Dataset):
    def __init__(self,path,train,sigma=50,size=crop_size,format='.png'):
        super(Noise_Dataset,self).__init__()
        self.size=size
        print('crop size',size)
        self.train=train
        self.format=format
        self.clearpath = path
        self.clear_imgs_dir=os.listdir(path)
        self.clear_imgs=[os.path.join(path, img) for img in self.clear_imgs_dir]
        self.sigma = sigma


    def _add_gaussian_noise(self, clean_patch, sigma):
        # noise = torch.randn(*(clean_patch.shape))
        # clean_patch = self.toTensor(clean_patch)

        clean_patch = np.asarray(clean_patch)
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * sigma, 0, 255).astype(np.uint8)
        noisy_patch = pi.fromarray(noisy_patch)
        # noisy_patch = torch.clamp(clean_patch + noise * sigma, 0, 255).type(torch.int32)
        return noisy_patch


    def __getitem__(self, index):

        clear=Image.open(self.clear_imgs[index]).convert('RGB')

        if isinstance(self.size,int):
            while clear.size[0]<self.size or clear.size[1]<self.size :
                index=random.randint(0,len(self.clear_imgs))

        rainfullpath=self.clear_imgs[index]

        clear_name=rainfullpath.split('/')[-1]

        if not isinstance(self.size,str):
            i,j,h,w=tfs.RandomCrop.get_params(clear,output_size=(self.size,self.size))
            clear=FF.crop(clear,i,j,h,w)
        else:
            h,w = clear.size[1],clear.size[0]
            clear = FF.crop(clear, 0, 0, h-1, w-1)

        if self.train:
            degration_type = random.randint(0, 3)
            if degration_type == 0:
                sigma = 15
            elif degration_type ==1:
                sigma = 25
            else:
                sigma = 50
            noise = self._add_gaussian_noise(clear,sigma)
        else:
            noise = self._add_gaussian_noise(clear,self.sigma)

        noise,clear=self.augData(noise.convert("RGB") ,clear.convert("RGB"))
        return noise,clear,clear_name


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
        return len(self.clear_imgs)

import os
pwd=os.getcwd()
print(pwd)
path='/home/lyd16/lsj/data/denoise/'#path to your 'data' folde
testpath = '/home/lyd16/lsj/data/CBSD68/original_png/'

Noise_train_loader=DataLoader(dataset=Noise_Dataset(path,train=True,size=crop_size),batch_size=BS,shuffle=True)
Noise_test_loader=DataLoader(dataset=Noise_Dataset(testpath,sigma=50,train=False,size='whole img'),batch_size=1,shuffle=False)

# OTS_train_loader=DataLoader(dataset=RESIDE_Dataset(path+'/RESIDE/OTS',train=True,format='.jpg'),batch_size=BS,shuffle=True)
# OTS_test_loader=DataLoader(dataset=RESIDE_Dataset(path+'/RESIDE/SOTS/outdoor',train=False,size='whole img',format='.png'),batch_size=1,shuffle=False)

if __name__ == "__main__":
    pass

