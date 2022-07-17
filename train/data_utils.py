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
depthnet = NormDepth(iskt=False)
depthnet = depthnet.to(opt.device)
depthnet = depthnet.eval()
BS=opt.bs
print(BS)
crop_size='whole_img'
if opt.crop:
    crop_size=opt.crop_size
from PIL import Image as pi

def get_target(tfile):

    width, height = tfile.size
    size = 128
    crops_s = []
    label_numbers = []
    num = 0

    for i in range(0, height - size, 128):
        for k in range(0, width - size, 128):
            num = num+1
            box = (k, i, k + size, i + size)
            small_t = tfile.crop(box)
            small_t_np = np.asarray(small_t)/255
            small_t_np = small_t_np[:,:,0]
            flag = np.median(small_t_np)
            if flag<=0.33:
                renumber = 0
            elif flag<=0.66:
                renumber = 1
            elif flag<=1:
                renumber = 2
            label_numbers.append(renumber)
    target_num = min(label_numbers)
    # print(len(crops_s))
    return target_num

import dehaze

def low_tran_mask(img):
    width,height = img.size

    img = img.resize((width//4, height//4),Image.ANTIALIAS)
    img = np.asarray(img)/255.
    img = np.power(img,0.25)
    img = img*255
    img = pi.fromarray(img.astype('uint8')).convert('RGB')

    # img.save('/home/lyd16/PycharmProjects/wangbin_Project/FFA-Net-master/smallhaze.png')


    haze = np.asarray(img)

    I = np.asarray(img) / 255

    dark = dehaze.DarkChannel(I, 15);
    A = dehaze.AtmLight(I, dark);

    dc, a = dehaze.get_dc_A(I, 111, 0.001, 0.95, 0.80)

    A = A - A + a
    te = dehaze.TransmissionEstimate(I, A, 15);
    t = dehaze.TransmissionRefine(haze, te);

    tfile = t
    # img.save('/home/lyd16/PycharmProjects/wangbin_Project/FFA-Net-master/haze.png')
    # print(tfile.shape)
    ht,wt = tfile.shape[0],tfile.shape[1]
    maxv = np.max(tfile)
    minv = np.min(tfile)
    tfile_norm = (tfile-minv)/(maxv-minv)
    mask1 = np.where(tfile_norm>= 0.8, np.ones_like(tfile_norm), np.zeros_like(tfile_norm))
    mask2 = np.where((tfile_norm>= 0.6) & (tfile_norm<0.8), np.ones_like(tfile_norm), np.zeros_like(tfile_norm))
    mask3 = np.where((tfile_norm>= 0.4) & (tfile_norm<0.6), np.ones_like(tfile_norm), np.zeros_like(tfile_norm))
    mask4 = np.where((tfile_norm>= 0.2) & (tfile_norm<0.4), np.ones_like(tfile_norm), np.zeros_like(tfile_norm))
    mask5 = np.where((tfile_norm>= 0.0) & (tfile_norm<0.2), np.ones_like(tfile_norm), np.zeros_like(tfile_norm))
    # B x 3 x 1 x 25 x 25
    mask1 = np.resize(mask1,[1,1,ht,wt])
    mask2 = np.resize(mask2, [1,1, ht, wt])
    mask3 = np.resize(mask3, [1,1, ht, wt])
    mask4 = np.resize(mask4, [1,1, ht, wt])
    mask5 = np.resize(mask5, [1,1, ht, wt])
    mask1 = torch.from_numpy(np.ascontiguousarray(mask1)).to(torch.float)
    mask2 = torch.from_numpy(np.ascontiguousarray(mask2)).to(torch.float)
    mask3 = torch.from_numpy(np.ascontiguousarray(mask3)).to(torch.float)
    mask4 = torch.from_numpy(np.ascontiguousarray(mask4)).to(torch.float)
    mask5 = torch.from_numpy(np.ascontiguousarray(mask5)).to(torch.float)
    allmask = torch.cat([mask1,mask2,mask3,mask4,mask5],dim=0)
    # vutils.save_image(allmask.cpu(),'/home/lyd16/PycharmProjects/wangbin_Project/FFA-Net-master/mask.png')
    # print(allmask.shape)
    # exit(-1)
    return allmask

def tran_mask(img,masknum):
    width,height = img.size

    img = img.resize((width//4, height//4),Image.ANTIALIAS)


    haze = np.asarray(img)
    I = np.asarray(img) / 255
    dark = dehaze.DarkChannel(I, 15);
    A = dehaze.AtmLight(I, dark);

    dc, a = dehaze.get_dc_A(I, 111, 0.001, 0.95, 0.80)

    A = A - A + a
    te = dehaze.TransmissionEstimate(I, A, 15);
    t = dehaze.TransmissionRefine(haze, te);

    tfile = t
    allmask = get_mask(masknum,tfile)

    return allmask

def get_mask(masknum,tfile):

    if masknum == 3:
        tag = 1/3
        masknums = [0.0,tag*1,tag*2]
        ht,wt = tfile.shape[0],tfile.shape[1]
        maxv = np.max(tfile)
        minv = np.min(tfile)
        tfile_norm = (tfile-minv)/(maxv-minv)
        mask1 = np.where(tfile_norm>= masknums[2], np.ones_like(tfile_norm), np.zeros_like(tfile_norm))
        mask2 = np.where((tfile_norm>= masknums[1]) & (tfile_norm<masknums[2]), np.ones_like(tfile_norm), np.zeros_like(tfile_norm))
        mask3 = np.where((tfile_norm>= 0.0) & (tfile_norm<masknums[1]), np.ones_like(tfile_norm), np.zeros_like(tfile_norm))
        # B x 3 x 1 x 25 x 25
        mask1 = np.resize(mask1,[1,1,ht,wt])
        mask2 = np.resize(mask2, [1,1, ht, wt])
        mask3 = np.resize(mask3, [1,1, ht, wt])

        mask1 = torch.from_numpy(np.ascontiguousarray(mask1)).to(torch.float)
        mask2 = torch.from_numpy(np.ascontiguousarray(mask2)).to(torch.float)
        mask3 = torch.from_numpy(np.ascontiguousarray(mask3)).to(torch.float)

        allmask = torch.cat([mask1,mask2,mask3],dim=0)


    if masknum == 5:
        # masknums = [0.0,0.2,0.4,0.6,0.8]
        ht,wt = tfile.shape[0],tfile.shape[1]
        maxv = np.max(tfile)
        minv = np.min(tfile)
        tfile_norm = (tfile-minv)/(maxv-minv)
        mask1 = np.where(tfile_norm>= 0.8, np.ones_like(tfile_norm), np.zeros_like(tfile_norm))
        mask2 = np.where((tfile_norm>= 0.6) & (tfile_norm<0.8), np.ones_like(tfile_norm), np.zeros_like(tfile_norm))
        mask3 = np.where((tfile_norm>= 0.4) & (tfile_norm<0.6), np.ones_like(tfile_norm), np.zeros_like(tfile_norm))
        mask4 = np.where((tfile_norm>= 0.2) & (tfile_norm<0.4), np.ones_like(tfile_norm), np.zeros_like(tfile_norm))
        mask5 = np.where((tfile_norm>= 0.0) & (tfile_norm<0.2), np.ones_like(tfile_norm), np.zeros_like(tfile_norm))
        # B x 3 x 1 x 25 x 25
        mask1 = np.resize(mask1,[1,1,ht,wt])
        mask2 = np.resize(mask2, [1,1, ht, wt])
        mask3 = np.resize(mask3, [1,1, ht, wt])
        mask4 = np.resize(mask4, [1,1, ht, wt])
        mask5 = np.resize(mask5, [1,1, ht, wt])
        mask1 = torch.from_numpy(np.ascontiguousarray(mask1)).to(torch.float)
        mask2 = torch.from_numpy(np.ascontiguousarray(mask2)).to(torch.float)
        mask3 = torch.from_numpy(np.ascontiguousarray(mask3)).to(torch.float)
        mask4 = torch.from_numpy(np.ascontiguousarray(mask4)).to(torch.float)
        mask5 = torch.from_numpy(np.ascontiguousarray(mask5)).to(torch.float)
        allmask = torch.cat([mask1,mask2,mask3,mask4,mask5],dim=0)
    if masknum ==7:
        tag = 1/7
        masknums = [tag*0,tag*1,tag*2,tag*3,tag*4,tag*5,tag*6]
        ht,wt = tfile.shape[0],tfile.shape[1]
        maxv = np.max(tfile)
        minv = np.min(tfile)
        tfile_norm = (tfile-minv)/(maxv-minv)
        mask1 = np.where(tfile_norm>= masknums[6], np.ones_like(tfile_norm), np.zeros_like(tfile_norm))
        mask2 = np.where((tfile_norm>= masknums[5]) & (tfile_norm<masknums[6]), np.ones_like(tfile_norm), np.zeros_like(tfile_norm))
        mask3 = np.where((tfile_norm>= masknums[4]) & (tfile_norm<masknums[5]), np.ones_like(tfile_norm), np.zeros_like(tfile_norm))
        mask4 = np.where((tfile_norm>= masknums[3]) & (tfile_norm<masknums[4]), np.ones_like(tfile_norm), np.zeros_like(tfile_norm))
        mask5 = np.where((tfile_norm>= masknums[2]) & (tfile_norm<masknums[3]), np.ones_like(tfile_norm), np.zeros_like(tfile_norm))
        mask6 = np.where((tfile_norm>= masknums[1]) & (tfile_norm<masknums[2]), np.ones_like(tfile_norm), np.zeros_like(tfile_norm))
        mask7 = np.where((tfile_norm>= 0.0) & (tfile_norm<masknums[1]), np.ones_like(tfile_norm), np.zeros_like(tfile_norm))
        # B x 3 x 1 x 25 x 25
        mask1 = np.resize(mask1,[1,1,ht,wt])
        mask2 = np.resize(mask2, [1,1, ht, wt])
        mask3 = np.resize(mask3, [1,1, ht, wt])
        mask4 = np.resize(mask4, [1,1, ht, wt])
        mask5 = np.resize(mask5, [1,1, ht, wt])
        mask6 = np.resize(mask6, [1,1, ht, wt])
        mask7 = np.resize(mask7, [1,1, ht, wt])

        mask1 = torch.from_numpy(np.ascontiguousarray(mask1)).to(torch.float)
        mask2 = torch.from_numpy(np.ascontiguousarray(mask2)).to(torch.float)
        mask3 = torch.from_numpy(np.ascontiguousarray(mask3)).to(torch.float)
        mask4 = torch.from_numpy(np.ascontiguousarray(mask4)).to(torch.float)
        mask5 = torch.from_numpy(np.ascontiguousarray(mask5)).to(torch.float)
        mask6 = torch.from_numpy(np.ascontiguousarray(mask6)).to(torch.float)
        mask7 = torch.from_numpy(np.ascontiguousarray(mask7)).to(torch.float)

        allmask = torch.cat([mask1,mask2,mask3,mask4,mask5,mask6,mask7],dim=0)
    return allmask



def depth_pre_mask(tfile):

    ht,wt = tfile.shape[0],tfile.shape[1]
    maxv = np.max(tfile)
    minv = np.min(tfile)
    tfile_norm = (tfile-minv)/(maxv-minv)
    mask1 = np.where(tfile_norm>= 0.8, np.ones_like(tfile_norm), np.zeros_like(tfile_norm))
    mask2 = np.where((tfile_norm>= 0.6) & (tfile_norm<0.8), np.ones_like(tfile_norm), np.zeros_like(tfile_norm))
    mask3 = np.where((tfile_norm>= 0.4) & (tfile_norm<0.6), np.ones_like(tfile_norm), np.zeros_like(tfile_norm))
    mask4 = np.where((tfile_norm>= 0.2) & (tfile_norm<0.4), np.ones_like(tfile_norm), np.zeros_like(tfile_norm))
    mask5 = np.where((tfile_norm>= 0.0) & (tfile_norm<0.2), np.ones_like(tfile_norm), np.zeros_like(tfile_norm))
    # B x 3 x 1 x 25 x 25
    mask1 = np.resize(mask1,[1,1,ht,wt])
    mask2 = np.resize(mask2, [1,1, ht, wt])
    mask3 = np.resize(mask3, [1,1, ht, wt])
    mask4 = np.resize(mask4, [1,1, ht, wt])
    mask5 = np.resize(mask5, [1,1, ht, wt])
    mask1 = torch.from_numpy(np.ascontiguousarray(mask1)).to(torch.float)
    mask2 = torch.from_numpy(np.ascontiguousarray(mask2)).to(torch.float)
    mask3 = torch.from_numpy(np.ascontiguousarray(mask3)).to(torch.float)
    mask4 = torch.from_numpy(np.ascontiguousarray(mask4)).to(torch.float)
    mask5 = torch.from_numpy(np.ascontiguousarray(mask5)).to(torch.float)
    allmask = torch.cat([mask1,mask2,mask3,mask4,mask5],dim=0)
    # vutils.save_image(allmask.cpu(),'/home/lyd16/PycharmProjects/wangbin_Project/FFA-Net-master/mask.png')
    # print(allmask.shape)
    # exit(-1)
    return allmask


def tensorShow(tensors,titles=None):
    '''
    t:BCWH
    '''
    fig=plt.figure()
    for tensor,tit,i in zip(tensors,titles,range(len(tensors))):
        img = make_grid(tensor)
        npimg = img.numpy()
        ax = fig.add_subplot(211+i)
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.set_title(tit)
    plt.show()


def depth_get_mask(plhaze,masknum):
    haze = tfs.ToTensor()(plhaze)

    shapes_h = haze.shape
    hazetemp = torch.unsqueeze(haze, dim=0)
    haze512 = F.interpolate(hazetemp, [256, 256])
    # print(haze512.shape)
    haze512 = haze512.to(opt.device)
    # haze512 = torch.unsqueeze(haze512,dim=0)
    normdepth = depthnet(haze512)
    normdepth = F.interpolate(normdepth, [shapes_h[1] // 4, shapes_h[2] // 4])
    normdepth = normdepth[0, 0, :, :]
    normdepth = normdepth.cpu().detach().numpy()
    allmask = get_mask(masknum=masknum,tfile=normdepth)
    return allmask

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


        haze,clear,pldata,normdata=self.augData(rain.convert("RGB") ,clear.convert("RGB"))

        if self.peddepthmask:

            shapes_h = haze.shape
            hazetemp = torch.unsqueeze(haze, dim=0)
            haze512 = F.interpolate(hazetemp, [256, 256])
            # print(haze512.shape)
            haze512 = haze512.to(opt.device)
            # haze512 = torch.unsqueeze(haze512,dim=0)
            normdepth = depthnet(haze512)
            normdepth = F.interpolate(normdepth, [shapes_h[1] // 4, shapes_h[2] // 4])
            normdepth = normdepth[0, 0, :, :]
            normdepth = normdepth.cpu().detach().numpy()
            allmask = get_mask(masknum=5,tfile=normdepth)
            # self.supermap[keyname]=normdepth
        else:
            typeq = clear_name.split('_')[0]
            if typeq == 'lowlight':
                self.islow =True

            if self.islow:
                allmask = low_tran_mask(pldata.convert("RGB"))
            else:
                allmask = tran_mask(pldata.convert("RGB"),masknum=5)

        return normdata,clear,haze,allmask,clear_name


    def augData(self,data,target):
        if self.train:
            rand_hor=random.randint(0,1)
            rand_rot=random.randint(0,3)
            data=tfs.RandomHorizontalFlip(rand_hor)(data)
            target=tfs.RandomHorizontalFlip(rand_hor)(target)

            if rand_rot:
                data=FF.rotate(data,90*rand_rot)
                target=FF.rotate(target,90*rand_rot)


        pldata = data
        data=tfs.ToTensor()(data)
        normdata=tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])(data)
        target=tfs.ToTensor()(target)

        return data,target,pldata,normdata

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
