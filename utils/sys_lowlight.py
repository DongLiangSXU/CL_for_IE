import sys
sys.path.append("..")
import os
import torch,sys,torchvision,argparse
import torchvision.transforms as tfs
from train_test.option import opt,model_name,log_dir
import time,math
import numpy as np
from torch.backends import cudnn
from torch import optim
import torch,warnings
from torch import nn
from torch.utils.data import DataLoader
#from tensorboardX import SummaryWriter
import torchvision.utils as vutils
warnings.filterwarnings('ignore')
from models import create_model
from torchvision.models import vgg16
from PIL import Image as pi
from torchvision.transforms import functional as FF


def torch_to_PIL(img_var):
    """
    Converts an image in torch.Tensor format to np.array.
    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    :param img_var:
    :return:
    """

    tensor = img_var.detach().cpu().numpy()[0]
    tensor = np.transpose(tensor,[1,2,0])
    tensor = tensor*255
    tensor = tensor.astype('uint8')
    tensor = pi.fromarray(tensor).convert('RGB')
    return tensor

def np_to_torch(img_np):
    """
    Converts image in numpy.array to torch.Tensor.
    From C x W x H [0..1] to  C x W x H [0..1]
    :param img_np:
    :return:
    """
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var):
    """
    Converts an image in torch.Tensor format to np.array.
    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    :param img_var:
    :return:
    """
    return img_var.detach().cpu().numpy()[0]

def get_model():
    import utils.lowlight_option as option
    import random
    rootpath = '/home/lyd16/PycharmProjects/wangbin_Project/Invertible-Image-Rescaling-master/codes/options/test'
    # darkmodelpath = rootpath+'/test_dark.yml'
    # lowlightmodelpath=rootpath+'/test_lowlight.yml'
    # midlightmodelpath=rootpath+'/test_midlight.yml'
    # fujimodelpath=rootpath+'/test_fuji.yml'
    lolmodelpath=rootpath+'/test_LOL.yml'
    pthpath = lolmodelpath
    optnew = option.parse(pthpath, is_train=False)
    optnew = option.dict_to_nonedict(optnew)
    model = create_model(optnew)
    model.netG.eval()
    return model
    
    
    # pthmap = {0:darkmodelpath,
    #           1:lowlightmodelpath,
    #           2:midlightmodelpath,
    #           3:fujimodelpath,
    #           4:lolmodelpath}
    # models_ = {}
    selectnumbers = [4]
    # for i in range(5):
    #     pthpath = pthmap[i]
    #     optnew = option.parse(pthpath, is_train=False)
    #     optnew = option.dict_to_nonedict(optnew)
    #     model = create_model(optnew)
    #     model.netG.eval()
    #     models_[i] = model
    # print(models_)

# 
# def syslowlight(img,model):
# 
# 
#     img = torch_to_PIL(img)
#     sizei = img.size
#     # print(sizei)
#     h, w = (sizei[0] // 128) * 128, (sizei[1] // 128) * 128
# 
#     # img = FF.crop(img,0,0,w,h)
#     img = img.resize((h*2,w*2),pi.ANTIALIAS)
#     input = tfs.ToTensor()(img)
#     input = torch.unsqueeze(input, dim=0)
#     clear = input.to(opt.device)
# 
# 
#     import utils.lowlight_option as option
#     import random
# 
#     rootpath = '/home/lyd16/PycharmProjects/wangbin_Project/Invertible-Image-Rescaling-master/codes/options/test'
#     darkmodelpath = rootpath+'/test_dark.yml'
#     lowlightmodelpath=rootpath+'/test_lowlight.yml'
#     midlightmodelpath=rootpath+'/test_midlight.yml'
#     fujimodelpath=rootpath+'/test_fuji.yml'
#     lolmodelpath=rootpath+'/test_LOL.yml'
# 
# 
#     pthmap = {0:darkmodelpath,
#               1:lowlightmodelpath,
#               2:midlightmodelpath,
#               3:fujimodelpath,
#               4:lolmodelpath}
#     models_ = {}
#     selectnumbers = [4]
#     for i in range(5):
#         pthpath = pthmap[i]
#         optnew = option.parse(pthpath, is_train=False)
#         optnew = option.dict_to_nonedict(optnew)
#         model = create_model(optnew)
#         model.netG.eval()
#         models_[i] = model
#     # print(models_)
# 
# 
# 
#     fakelowlights = []
#     for k in range(clear.shape[0]):
#         sclear = clear[k:k+1,:,:,:]
#         snum = random.choice(selectnumbers)
#         model =models_[snum]
#         output_img = model.downscale(sclear)
#         fakelowlights.append(output_img)
# 
#     fakelowlight = torch.cat(fakelowlights,dim=0)
#     fakelowlight = fakelowlight.to(opt.device)
#     return fakelowlight

        # vutils.save_image(clear.cpu(),cropclear+hname)
    #




import itertools
# 35.8
#
# if __name__ == "__main__":
#
#     syslowlight()


