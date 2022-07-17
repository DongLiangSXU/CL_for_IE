import sys
sys.path.append("..")
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


import torch,sys,torchvision,argparse
import torchvision.transforms as tfs
from metrics import psnr,ssim
from models import *
import time,math
from utils import add_rain
import numpy as np
from torch.backends import cudnn
from torch import optim
import torch,warnings
from torch import nn
from torch.utils.data import DataLoader
#from tensorboardX import SummaryWriter
import torchvision.utils as vutils
warnings.filterwarnings('ignore')
from option import opt,model_name,log_dir
from data_utils import *
from torchvision.models import vgg16
print('log_dir :',log_dir)
print('model_name:',model_name)
from skimage.color import rgb2hsv
# from nngeometry.metrics import FIM
# from nngeometry.object import PMatKFAC, PMatDiag, PVector
# from nngeometry.layercollection import LayerCollection
import torch
from torch.utils.data import DataLoader
from copy import deepcopy
from torch.nn import functional as F
from tqdm.notebook import tqdm


def gradient(y):

    gradient_h = torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])
    gradient_y = torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])

    return gradient_h, gradient_y


models_={
    # 'mynet':BSnet(32,3),
    'depthnet':NormDepth(iskt=False),
    # 'rnet':RNet(32,3),
    'HDM':DR_Net_phy(3,3),
}

loaders_={
    'its_train':Rain_train_loader,
    'its_test':ALL_test_loader,
}

def norm_depth(depth):
    maxv = torch.max(depth)
    minv = torch.min(depth)
    n_depth = (depth-minv)/(maxv-minv)
    return n_depth

def tran2depth(tranp):
    # tranp = F.sigmoid(tranp)
    t1 = -torch.log(tranp)
    tdepth = norm_depth(t1)
    return tdepth


def make_haze(dehazemap,rainmap,depthnet):
    # beta = random.uniform(0.5, 2)
    # a_map = 0.05
    fake_tran_list = []
    depth_re = norm_depth(depthnet(dehazemap))
    betav = [0.5,1,1,2,2,3,3,4]
    random.shuffle(betav)
    for k in range(depth_re.shape[0]):
        if k>=8:
            k = k-8
        beta = random.uniform(betav[k], betav[k]+1)
        depth_res = depth_re[k:k+1,:,:,:]
        fake_trans = torch.exp(-beta * depth_res)
        fake_tran_list.append(fake_trans)

    fake_tran = torch.cat(fake_tran_list,dim=0)
    # print(a_map)
    # fake_tran = torch.exp(-beta*depth_re)
    atolabel = torch.zeros_like(fake_tran)
    avalues = [0.5,0.6,0.7,0.8]
    random.shuffle(avalues)
    for i in range(atolabel.shape[0]):
        # if i>=8:
        #     i = i-8
        atov = random.choice(avalues)
        a_map = random.uniform(atov,atov+0.2)
        # print(a_map)
        atolabel[i:i+1,:,:,:] += a_map

    fake_haze = dehazemap*fake_tran+atolabel*(1-fake_tran)
    fake_haze_addrain = rainmap*fake_tran+atolabel*(1-fake_tran)
    atolabel = atolabel.to(opt.device)
    fake_haze = fake_haze.to(opt.device)
    fake_tran = fake_tran.to(opt.device)
    return fake_haze,fake_haze_addrain


start_time=time.time()
T=opt.steps
def lr_schedule_cosdecay(t,T,init_lr=opt.lr):
    lr=0.5*(1+math.cos(t*math.pi/T))*init_lr
    return lr



def train(net,loader_train, loader_test,haze_loader, low_loader,optim, optim_ewc,pre_parms,criterion):
    losses=[]
    start_step=0
    max_ssim=0
    max_psnr=0
    ssims=[]
    psnrs=[]
    # crloss = ContrastLoss()
    # syslowlightmodel = sys_lowlight.get_model()

    # loss_function = nn.CrossEntropyLoss()
    if opt.resume and os.path.exists(opt.model_dir):
        print(f'resume from {opt.model_dir}')
        ckp=torch.load(opt.model_dir)
        losses=ckp['losses']
        net.load_state_dict(ckp['model'])
        # rnet.load_state_dict(ckp['rnet'])
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
    #     ssim_eval,psnr_eval = test(net,loader_test, '')
    #     # testnature(net,'')
    # #
    # exit(-1)

    # step2:calcaluate grds
    pre_parms = pre_parms.to(opt.device)
    grads = torch.zeros_like(pre_parms)
    print(len(haze_loader))
    # for k in range(len(haze_loader)):
    #     x,y,_,mask,fname = next(iter(haze_loader))
    #     rclear=y.to(opt.device)
    #     rain = x.to(opt.device)
    #     mask = mask.to(opt.device)
    #     odata = _.to(opt.device)
    #     derain_re = net(rain,mask)
    #     loss_nf = criterion[0](derain_re,rclear)
    #     loss2=criterion[1](derain_re,rclear)
    #     loss=loss_nf+0.04*loss2
    #
    #     loss.backward()
    #     optim_ewc.step()
    #
    #
    #     tempgard = []
    #     for name, parms in net.module.named_parameters():
    #         flatten = parms.grad.view(-1)
    #         flatten = torch.abs(flatten)*1000
    #         # print(torch.mean(flatten))
    #         tempgard.append(flatten)
    #     sgard = torch.cat(tempgard)
    #     grads = grads+sgard
    #
    #     optim_ewc.zero_grad()
    grads = grads/len(haze_loader)
    grads = grads.to(opt.device)
    print(torch.max(grads))

    replay_ratio = [0,0,1,1]

    print('-----------------------ewc prepare done!--------------------------------')

    # cl_for_rain
    totrain_loader = loader_train
    for step in range(start_step+1,opt.steps+1):
        net.train()
        lr=opt.lr
        if not opt.no_lr_sche:
            lr=lr_schedule_cosdecay(step,T)
            for param_group in optim.param_groups:
                param_group["lr"] = lr

        isreplay = random.choice(replay_ratio)
        if isreplay or step % opt.eval_step ==0:
            x,y,_,mask,fname = next(iter(haze_loader))
        else:
            x,y,_,mask,fname = next(iter(totrain_loader))

        low_inputs = x.to(opt.device)
        clear = y.to(opt.device)
        mask = mask.to(opt.device)
        odata = _.to(opt.device)


        # vutils.save_image(low_inputs.cpu(),'./hazerainin.png')
        # exit(-1)

        derain_re = net(low_inputs,mask)

        pixelloss = criterion[0](derain_re,clear)
        loss2=criterion[1](derain_re,clear)
        pixelloss=pixelloss+0.04*loss2

        # a: dehaze
        # p: gt
        # n: haze
        # mycrloss = crloss(derain_re.contiguous(),clear.contiguous(),low_inputs.contiguous())

        final_loss = 1*pixelloss

        currentparm = []
        currentparms = torch.zeros_like(pre_parms)
        currentparms = currentparms.to(opt.device)
        for name, parms in net.module.named_parameters():
            flatten = parms.view(-1)
            # flatten = flatten*flatten
            # flatten = torch.abs(flatten)
            currentparm.append(flatten)
            currentparms = torch.cat(currentparm)


        noforgetloss = 10000*torch.mean(torch.mul(grads,torch.abs(currentparms-pre_parms)))
        # noforgetloss = 10000*(torch.sum(torch.mul(grads, torch.abs(currentparms - pre_parms)))
        #                       +torch.pow(torch.sum(torch.mul(grads, torch.abs(currentparms - pre_parms))),2))


        totalloss = final_loss

        totalloss.backward()

        optim.step()
        optim.zero_grad()

        # if step%250==0:
        #     toshow = torch.cat([low_inputs,derain_re,clear])
        #     vutils.save_image(toshow.cpu(),'/home/lyd16/PycharmProjects/wangbin_Project/AAAI2023works/CR_image_enhancement/tempshow/'+str(step)+'show.png')

        print(f'\rtrain loss : {totalloss.item():.5f}|{(pixelloss.item()):.5f}|{(noforgetloss.item()):.5f}|is_relay:{isreplay}|'
              f'step :{step}/{opt.steps}|lr :{lr :.7f} |time_used :{(time.time()-start_time)/60 :.1f}',
              end='',flush=True)


        if step % opt.eval_step ==0 :
            with torch.no_grad():
                # ssim_eval,psnr_eval,ssim_eval2,psnr_eval2,ssim_eval3,psnr_eval3 = test(net,loader_test, max_psnr,max_ssim,step)

                pthpath = './HDM_h2r/pth_save/'+str(step)+'/'
                imgsavepath = './HDM_h2r/imgsave/' + str(step) + '/'
                if not os.path.exists(pthpath):
                    os.makedirs(pthpath)
                if not os.path.exists(imgsavepath):
                    os.makedirs(imgsavepath)
                ssim_eval, psnr_eval = test(net,loader_test,imgsavepath)
                # testnature(net,imgsavepath)
                torch.save(net.state_dict(), '%s/hdm_super_h2r_%d.pth' % (pthpath, step))
                # torch.save(rnet.state_dict(),'%s/r_net_%d.pth' % (pthpath, step))

            # print(f'\nstep :{0} |ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}|ssim2:{ssim_eval2:.4f}| psnr2:{psnr_eval2:.4f}|ssim3:{ssim_eval3:.4f}| psnr3:{psnr_eval3:.4f}|max_ssim:{max_ssim:.4f}| max_psnr:{max_psnr:.4f}')
            print(
                f'\nstep :{0} |ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}|max_ssim:{max_ssim:.4f}| max_psnr:{max_psnr:.4f}')


            ssims.append(ssim_eval)
            psnrs.append(psnr_eval)
            if ssim_eval > max_ssim and psnr_eval > max_psnr :
                max_ssim=max(max_ssim,ssim_eval)
                max_psnr=max(max_psnr,psnr_eval)
                torch.save({
                    'step':step,
                    'max_psnr':max_psnr,
                    'max_ssim':max_ssim,
                    'ssims':ssims,
                    'psnrs':psnrs,
                    'losses':losses,
                    'model':net.state_dict(),
                },opt.model_dir)
                print(f'\n model saved at step :{step}| max_psnr:{max_psnr:.4f}|max_ssim:{max_ssim:.4f}')

#
def test(net,loader_test,imgsavepath):
    net.eval()
    # rnet.eval()

    torch.cuda.empty_cache()
    ssimshazy=[]
    psnrshazy=[]
    ssimsdid=[]
    psnrsdid=[]
    ssimslowlight=[]
    psnrslowlight=[]
    ssimsrain=[]
    psnrsrain=[]
    ssims = []
    psnrs = []

    for i ,(lowq,clear,_,mask,clear_name) in enumerate(loader_test):

        type = clear_name[0].split('_')[0]
        if type == 'did' or type == 'lowlight':
            continue

        if type == 'hazy':
            flagnum = clear_name[0].split('_')[1]
            flagnum = int(flagnum.split('.')[0])
            if flagnum>500:
                continue

        lowq = lowq.to(opt.device)
        odata = _.to(opt.device)
        # if type == 'lowlight':
        #     lowq = odata
        mask = mask.to(opt.device)
        derain_re = net(lowq,mask)
        y = clear.to(opt.device)
        derain_re = torch.clamp(derain_re, 0, 1)
        ssim1=ssim(derain_re,y).item()
        psnr1=psnr(derain_re,y)

        # scorestr = '==' + str(ssim1) + '=====' + str(psnr1)
        # if type == 'did':
        #     ssimsdid.append(ssim1)
        #     psnrsdid.append(psnr1)
        if type == 'rain':
            ssimsrain.append(ssim1)
            psnrsrain.append(psnr1)
        if type == 'hazy':
            ssimshazy.append(ssim1)
            psnrshazy.append(psnr1)
        # if type == 'lowlight':
        #     ssimslowlight.append(ssim1)
        #     psnrslowlight.append(psnr1)
        ssims.append(ssim1)
        psnrs.append(psnr1)
        # print(np.mean(ssims),np.mean(psnrs))
        # show = torch.cat([dehazemap, ohaze, y], dim=0)
        # vutils.save_image(derain_re.cpu(),imgsavepath+clear_name[0])

    print('did:--',np.mean(ssimsdid) ,'---',np.mean(psnrsdid),'---','hazy:--',np.mean(ssimshazy) ,'---',np.mean(psnrshazy), \
          '---','rain:--',np.mean(ssimsrain) ,np.mean(psnrsrain),'---','lol:--',np.mean(ssimslowlight) ,np.mean(psnrslowlight))
    return np.mean(ssims),np.mean(psnrs)

from collections import OrderedDict
def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True

import itertools
# 35.8

if __name__ == "__main__":
    loader_train=loaders_[opt.trainset]
    hazeloader = Haze_train_loader
    lowloader = LL_train_loader
    loader_test=loaders_[opt.testset]

    seednums = [20,30,40,50,60]
    seednum=seednums[0]
    setup_seed(seednum)

    net=models_['HDM']
    net=net.to(opt.device)

    depthnet = models_['depthnet']
    depthnet = depthnet.to(opt.device)

    if opt.device=='cuda':
        net=torch.nn.DataParallel(net)
        # rnet = torch.nn.DataParallel(rnet)
        cudnn.benchmark=True

    # net.load_state_dict(torch.load('/home/lyd16/PycharmProjects/wangbin_Project/FFA-Net-master/net/trained_models/its_train_ffa_3_3212.pk')['model'])



    parts = []
    # num_params = 0
    # for param in net.module.parameters():
    #     num_params += param.numel()
    #     print(num_params / 1e6)
    for para in net.module.parameters():
        flatten = para.view(-1)
        # flatten = flatten*flatten*100

        # flatten = torch.abs(flatten)
        # print(flatten)
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

    optimizer = optim.Adam(itertools.chain(net.parameters()),lr=opt.lr, betas = (0.9, 0.999), eps=1e-08)
    optimizer_ewc = optim.Adam(itertools.chain(net.parameters()),lr=0.000001, betas = (0.9, 0.999), eps=1e-08)

    optimizer.zero_grad()
    optimizer_ewc.zero_grad()
    train(net,loader_train,loader_test,hazeloader,lowloader,optimizer,optimizer_ewc,pre_parms,criterion)


