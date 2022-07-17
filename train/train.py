import sys
sys.path.append("..")
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


import torch,sys,torchvision,argparse
import torchvision.transforms as tfs
from metrics import psnr,ssim
from models import *
import time,math
import numpy as np
from torch.backends import cudnn
from torch import optim
import torch,warnings
from torch import nn
#from tensorboardX import SummaryWriter
import torchvision.utils as vutils
warnings.filterwarnings('ignore')
from option import opt,model_name,log_dir
from data_utils import *
from torchvision.models import vgg16
print('log_dir :',log_dir)
print('model_name:',model_name)

# 324:LOL
# 320:rain100l
# 322:did
# 321:old_rain200h
# 327:msbdn_rain200h
#329:rain200h nodrconv
#3212:reside nodrconv
#3211:did nodrconv
#3213 lol nodrconv
#3214 rain100l nodrconv
#3215 sice part1 nodrconv
#3216 NHHaze nodrconv

#2022461：ffa2dcp2dcnv
#2022462:dcp2dcnv2ffa
#2022463:dcp2ffa2dcnv
#2022464:dcnv2dcp2ffa
models_={
    # 1062   net
    'ffa':DR_Net_phy(3,3)

}
loaders_={
    'its_train':Rain_train_loader,
    'its_test':ALL_test_loader,

}
start_time=time.time()
T=opt.steps
def lr_schedule_cosdecay(t,T,init_lr=opt.lr):
    lr=0.5*(1+math.cos(t*math.pi/T))*init_lr
    return lr

def train(net,loader_train,loader_test,optim,criterion,pre_parms):
    losses=[]
    start_step=0
    max_ssim=0
    max_psnr=0
    ssims=[]
    psnrs=[]
    if opt.resume and os.path.exists(opt.model_dir):
        print(f'resume from {opt.model_dir}')
        ckp=torch.load(opt.model_dir)
        losses=ckp['losses']
        net.load_state_dict(ckp['model'])
        start_step=ckp['step']
        max_ssim=ckp['max_ssim']
        max_psnr=ckp['max_psnr']
        psnrs=ckp['psnrs']
        ssims=ckp['ssims']
        print(f'start_step:{start_step} start training ---')
        print('maxpsnr:', max_psnr, 'max-ssim:', max_ssim)

    else :
        print('train from scratch *** ')
    # with torch.no_grad():
    #      ssim_eval, psnr_eval = test(net, loader_test, max_psnr, max_ssim, 0)
    #     # testnature(net)
    # #    # testnature(net,'/home/lyd16/PycharmProjects/wangbin_Project/Invertible-Image-Rescaling-master/darkface/')
    # #
    # exit(-1)

    step = 0
    for step in range(start_step+1,opt.steps+1):
        # for epoch in range(100):
        #     for i, (x, y, _, mask) in enumerate(loader_train):
        net.train()
        # step = step+1

        lr=opt.lr
        # print(lr)
        if not opt.no_lr_sche:
            lr=lr_schedule_cosdecay(step,T)
            for param_group in optim.param_groups:
                param_group["lr"] = lr

        x,y,_,mask,fname=next(iter(loader_train))
        x=x.to(opt.device);y=y.to(opt.device);mask = mask.to(opt.device);ohaze = _.to(opt.device);
        #        x = torch.pow(x,0.45)

        out=net(x,mask)

        loss=criterion[0](out,y)
        opt.perloss = True
        if opt.perloss:
            loss2=criterion[1](out,y)
            loss=loss+0.04*loss2

        loss.backward()

        optim.step()
        optim.zero_grad()

        #        toshow = torch.cat([x,out,y],dim=0)
        #        vutils.save_image(toshow.cpu(),'./'+str(step)+'show.png')
        #        vutils.save_image(mask[0].cpu(),'./'+str(step)+'mask.png')
        #       #
        #        exit(-1)
        losses.append(loss.item())
        print(f'\rtrain loss : {loss.item():.5f}| step :{step}/{opt.steps}|lr :{lr :.7f} |time_used :{(time.time()-start_time)/60 :.1f}',end='',flush=True)

        if step % opt.eval_step ==0 :
            with torch.no_grad():
                ssim_eval,psnr_eval=test(net,loader_test, max_psnr,max_ssim,step)

            print(f'\nepoch :{step//opt.eval_step} |ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}|max_ssim:{max_ssim:.4f}| max_psnr:{max_psnr:.4f}')


            ssims.append(ssim_eval)
            psnrs.append(psnr_eval)
            if ssim_eval > max_ssim and psnr_eval > max_psnr :
                # max_ssim=max(max_ssim,ssim_eval)
                # max_psnr=max(max_psnr,psnr_eval)
                max_ssim=ssim_eval
                max_psnr=psnr_eval

                torch.save({
                    'step':step,
                    'max_psnr':max_psnr,
                    'max_ssim':max_ssim,
                    'ssims':ssims,
                    'psnrs':psnrs,
                    'losses':losses,
                    'model':net.state_dict()
                },opt.model_dir)
                print(f'\n model saved at epoch :{step//opt.eval_step}| max_psnr:{max_psnr:.4f}|max_ssim:{max_ssim:.4f}')

    # np.save(f'./numpy_files/{model_name}_{opt.steps}_losses.npy',losses)
    # np.save(f'./numpy_files/{model_name}_{opt.steps}_ssims.npy',ssims)
    # np.save(f'./numpy_files/{model_name}_{opt.steps}_psnrs.npy',psnrs)

imgpath = '/home/lyd16/PycharmProjects/wangbin_Project/FFA-Net-master/testsot/136.png'

def test(net,loader_test,max_psnr,max_ssim,step):
    net.eval()
    torch.cuda.empty_cache()
    ssims=[]
    psnrs=[]
    num = 0
    #s=True
    for i ,(inputs,targets,_,mask,fname) in enumerate(loader_test):

        num = num+1
        filename = fname[0]

        inputs=inputs.to(opt.device);targets=targets.to(opt.device);mask = mask.to(opt.device);_ = _.to(opt.device)
        # print(inputs.shape)
        # vutils.save_image(_.cpu(), '/home/lyd16/PycharmProjects/wangbin_Project/FFA-Net-master/ohazere/'+str(num)+'haze.png')
        #        inputs = torch.pow(inputs,0.45)
        pred=net(inputs,mask)
        #        pred = torch.clamp(pred,0,1)
        ssim1=ssim(pred,targets).item()
        psnr1=psnr(pred,targets)
        #
        ##        tshow = torch.cat([inputs,pred,targets],dim=0)
        #        vutils.save_image(pred.cpu(), '/home/lyd16/PycharmProjects/wangbin_Project/FFA-Net-master/lHDN_re/DID_test/'+filename)
        #        vutils.save_image(pred.cpu(), '/home/lyd16/PycharmProjects/wangbin_Project/FFA-Net-master/nhre/nodrconv/'+filename)

        ssims.append(ssim1)
        psnrs.append(psnr1)
        # print(np.mean(ssims) ,np.mean(psnrs))


    return np.mean(ssims) ,np.mean(psnrs)

from torchvision.transforms import functional as FF
from PIL import Image as pi
def testnature(net):
    net.eval()
    torch.cuda.empty_cache()

    #    naturepath = '/home/lyd16/PycharmProjects/wangbin_Project/data/track1.2_test_sample/'
    #    naturepath = './realrain/'
    naturepath = '/home/lyd16/PycharmProjects/wangbin_Project/FFA-Net-master/net/rainre/makemask/'
    names=os.listdir(naturepath)

    # hazesavefilenames = os.listdir(imgsavepath)

    # num = 0
    # for i ,(haze, clear,ohaze,atov) in enumerate(loader_test):
    for hname in names:
        filename = naturepath+hname
        # filename = '/home/lyd16/PycharmProjects/wangbin_Project/data/track1.2_test_sample/0.png'
        #        imgsavepath = '/home/lyd16/PycharmProjects/wangbin_Project/FFA-Net-master/net/rainre/masksave/'

        img = pi.open(filename).convert('RGB')

        sizei = img.size
        print(sizei)
        h,w = (sizei[0]//4)*4,(sizei[1]//4)*4
        img = img.resize((h*4,w*4),pi.ANTIALIAS)
        # if h>3000 or w>3000:
        #     continue
        #        img = FF.crop(img,0,0,w,h)
        # img = img.resize((300,400),pi.ANTIALIAS)
        maskfortest = depth_get_mask(img,5)
        maskfortest = torch.unsqueeze(maskfortest, dim=0)
        maskfortest = maskfortest.to(opt.device)
        # print(img.size)
        # exit(-1)
        inputmap = tfs.ToTensor()(img)
        oinput = inputmap
        oinput = torch.unsqueeze(oinput, dim=0)
        oinput = oinput.to(opt.device)
        #        inputmap = tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])(inputmap)
        inputmap = torch.unsqueeze(inputmap, dim=0)
        # for:dehaze

        ohaze = inputmap

        #        ohaze = ohaze.to(opt.device)


        ohaze = ohaze.to(opt.device)
        #        ohaze = torch.pow(ohaze,0.45)
        dehazemap = net(ohaze,maskfortest)
        dehazemap = torch.clamp(dehazemap,0,1)
        tshow = torch.cat([oinput,dehazemap],dim=0)

        vutils.save_image(dehazemap.cpu(),
                          imgsavepath+hname)
        # exit(-1)

import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
# 35.8

if __name__ == "__main__":
    loader_train=loaders_[opt.trainset]
    loader_test=loaders_[opt.testset]
    net=models_[opt.net]
    net=net.to(opt.device)
    num_params = 0
    seednums = [20,30,40,50,60]
    seednum=seednums[0]
    setup_seed(seednum)
    # net.load_state_dict(torch.load('./trained_models/its_train_ffa_3_10216.pk')['model'])

    # for param in net.parameters():
    #     num_params += param.numel()
    # print(num_params / 1e6)

    # exit(-1)

    if opt.device=='cuda':
        net=torch.nn.DataParallel(net)
        cudnn.benchmark=True


    net.load_state_dict(torch.load('/home/lyd16/PycharmProjects/wangbin_Project/FFA-Net-master/net/trained_models/its_train_ffa_3_329.pk')['model'])



    parts = []
    for para in net.module.parameters():
        flatten = para.view(-1)
        parts.append(flatten)
    pre_parms = torch.cat(parts)

    criterion = []
    criterion.append(nn.L1Loss().to(opt.device))
    opt.perloss = True
    if opt.perloss:
        vgg_model = vgg16(pretrained=True).features[:16]
        vgg_model = vgg_model.to(opt.device)
        for param in vgg_model.parameters():
            param.requires_grad = False
        criterion.append(PerLoss(vgg_model).to(opt.device))
    optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()),lr=opt.lr, betas = (0.9, 0.999), eps=1e-08)
    optimizer.zero_grad()
    train(net,loader_train,loader_test,optimizer,criterion,pre_parms)


