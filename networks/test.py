# Citation:
#     Gated Fusion Network for Joint Image Deblurring and Super-Resolution
#     The British Machine Vision Conference(BMVC2018 oral)
#     Xinyi Zhang, Hang Dong, Zhe Hu, Wei-Sheng Lai, Fei Wang and Ming-Hsuan Yang
# Contact:
#     cvxinyizhang@gmail.com
# Project Website:
#     http://xinyizhang.tech/bmvc2018
#     https://github.com/jacquelinelala/GFN

from __future__ import print_function
import argparse
import os
import time
from math import log10
from os.path import join
from torchvision import transforms
from torchvision import utils as utils
import torch
from torch.utils.data import DataLoader
from datasets.dataset_hf5 import DataValSet
import statistics
import re
import torch.nn as nn
import torch.nn.functional as F
import pytorch_ssim
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as sc
from networks import radialProfile

epsilon = 1e-8

parser = argparse.ArgumentParser(description="PyTorch LapSRN Test")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
parser.add_argument("--isTest", type=bool, default=True, help="Test or not")
parser.add_argument('--dataset', type=str, default='SOTS', help='Path of the validation dataset')
parser.add_argument("--checkpoint", default="/home/ubuntu/PycharmProjects/MSBDN-DFF-master/model.pkl", type=str,
                    help="Test on intermediate pkl (default: none)")
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--name', type=str, default='MSBDN', help='filename of the training models')
parser.add_argument("--start", type=int, default=2, help="Activated gate module")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def nyu_resize(img, w=512, h=512, padding=6):
    from skimage.transform import resize
    return resize(img, (w, h), preserve_range=True, mode='reflect', anti_aliasing=True)


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def is_pkl(filename):
    return any(filename.endswith(extension) for extension in [".pkl"])


import h5py


def RGB2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def get_1d_spectum(imgnumpy):
    img_numpy = imgnumpy
    img_gray = RGB2gray(img_numpy)
    fft = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(fft)
    fshift += epsilon
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    psd1D = radialProfile.azimuthalAverage(magnitude_spectrum)
    # psd1D = (psd1D-np.min(psd1D))/(np.max(psd1D)-np.min(psd1D))
    # showlist = list(psd1D
    # print(psd1D.shape)
    # print(psd1D)
    return psd1D


def draw(showlist):
    print(len(showlist[0]))
    # 360
    # 参与比较的方法：DCP,GCANET,DCPDN,MSBDN,AODNET,PDLDNET,haze,nohaze

    hazey = showlist[0]
    nohazey = showlist[1]
    gcay = showlist[2]
    # dcpy = showlist[3][188:288]
    # dcpy = showlist[4]
    # dcpdny = showlist[5]
    # msbdny = showlist[6]
    # pdldy = showlist[7]

    # encoding=utf-8
    from matplotlib import pyplot
    import matplotlib.pyplot as plt
    # xl = ['0-0.5','0.5-1','1-1.5','1.5-2','2-2.5','2.5-3','3-3.5','3.5-4','4-4.5','4.5-5','5-5.5','5.5-6',/
    #              '6-6.5','6.5-7','7-7.5','7.5-8','8-8.5','8.5-9','9-9.5','9.5-10']

    y_1 = hazey
    y_2 = nohazey
    y_3 = gcay
    # y_4 = dcpy
    # y_5 = gcay
    # y_6 = dcpdny
    # y_7 = msbdny
    # y_8 = pdldy
    names = [str(x) for x in range(len(y_1))]
    x = range(len(names))
    # plt.plot(x, y, 'ro-')
    # plt.plot(x, y1, 'bo-')
    # pl.xlim(-1, 11)  # 限定横轴的范围
    # pl.ylim(-1, 110)  # 限定纵轴的范围

    plt.plot(x, y_1, marker='+', ms=10, label='haze')
    plt.plot(x, y_2, marker='+', ms=10, label='haze-free')
    plt.plot(x, y_3, marker='+', ms=10, label='dehaze-free')
    # plt.plot(x, y_4, marker='+',  ms=10, label='dcp-free')
    # plt.plot(x, y_5, marker='+', ms=10, label='dcpdn')
    # plt.plot(x, y_6, marker='+',  ms=10, label='gca')
    # plt.plot(x, y_7, marker='+',  ms=10, label='msbdn')
    # plt.plot(x, y_8, marker='+', ms=10, label='pdld')

    plt.legend()  # 让图例生效
    # plt.xticks(x, names, rotation=1)

    plt.margins(0)
    plt.subplots_adjust(bottom=0.2)
    # plt.xlabel('')  # X轴标签
    plt.ylabel("spectum_1d")  # Y轴标签

    plt.show()


def reside_test(model, crit):
    savepath = '/home/ubuntu/PycharmProjects/dataset/data/itots/valin/'
    # savepath = '/home/ubuntu/PycharmProjects/dataset/data/sot/indoor/'
    realhazepath = '/home/ubuntu/PycharmProjects/dataset/data/REISDE/ULH/UnannotatedHazyImages/'

    files = os.listdir(savepath)
    sca = 0
    scp = 0
    c = 0
    kkk = 375
    l1f = np.zeros([kkk])
    l2f = np.zeros([kkk])
    l3f = np.zeros([kkk])
    for idx in files:
        # idx = '1400_5.png'
        print(idx)
        c = c + 1

        # 608-448

        f = h5py.File(savepath + idx, 'r')
        im = f['haze'][:]
        gt = f['gt'][:]

        # print(gt.shape)

        input = transforms.ToTensor()(im)
        input = input.float()
        input = torch.unsqueeze(input, 0)
        # input = input.cuda()
        imgIn = input.cuda()

        gttensor = transforms.ToTensor()(gt)
        gttensor = gttensor.float()
        gttensor = torch.unsqueeze(gttensor, 0)
        # input = input.cuda()
        gttensor = gttensor.cuda()

        # with torch.no_grad():
        #     for iteration, batch in enumerate(test_gen, 1):
        #         # print(iteration)
        #         Blur = batch[0]
        #         HR = batch[1]
        #         Blur = Blur.to(device)
        #         HR = HR.to(device)
        #
        #         name = batch[2][0][:-4]

        # start_time = time.perf_counter()#-------------------------begin to deal with an image's time

        sr = model(imgIn)

        # modify
        try:
            sr = torch.clamp(sr, min=0, max=1)
        except:
            sr = sr[0]
            sr = torch.clamp(sr, min=0, max=1)

        prediction = sr.data.cpu().numpy().squeeze().transpose((1, 2, 0))

        mse = crit(sr, gttensor)
        psnr = 10 * log10(1 / mse)

        ssim = pytorch_ssim.ssim(sr, gttensor)
        ssim = ssim.data.cpu().numpy()

        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.imshow(prediction)
        # plt.subplot(1, 2, 2)
        # plt.imshow(im)
        # #
        # plt.show()
        # l1 = get_1d_spectum(im)
        # l2 = get_1d_spectum(gt)
        # l3 = get_1d_spectum(prediction)
        #
        # l1f += l1
        # l2f += l2
        # l3f += l3

        import skimage.measure
        ssimk = skimage.measure.compare_ssim(gt, prediction, data_range=1, multichannel=True)
        sca += ssimk
        scp += psnr

        # print(ssim)
        print('ssim:%.4f || psnr:%.4f' % (sca / c, scp / c))
    # l1f = list(l1f / (c))
    # l2f = list(l2f / (c))
    # l3f = list(l3f / (c))
    #
    # showlist = []
    # showlist.append(l1f)
    # showlist.append(l2f)
    # showlist.append(l3f)
    # draw(showlist)


def sot(model):
    rootdir = '/media/wang/1b319375-860b-4c76-b38f-ad6d60cea6fe'
    # path = '/media/wang/1b319375-860b-4c76-b38f-ad6d60cea6fe/home/ubuntu/PycharmProjects/dataset/data/SOTS/HR_hazy/'
    # realhazepath = '/home/ubuntu/PycharmProjects/dataset/data/REISDE/ULH/UnannotatedHazyImages/'
    path = rootdir + '/home/ubuntu/PycharmProjects/Thermos_Master/sot_result/haze/'
    gtpath = rootdir + '/home/ubuntu/PycharmProjects/Thermos_Master/sot_result/gt/'
    # gtpath = '/media/wang/1b319375-860b-4c76-b38f-ad6d60cea6fe/home/ubuntu/PycharmProjects/dataset/data/SOTS/HR/'

    files = os.listdir(path)
    sca = 0
    scp = 0
    c = 0
    for idx in files:
        # idx = '1400_5.png'

        num = idx.split('.')[0]

        # num = int(idx.split('.')[0])
        # if num > 500:
        #     continue
        c = c + 1
        # name = '/home/ubuntu/PycharmProjects/dataset/data/SOTS/outdoor/hazy/' + idx
        #
        # gtname = '/home/ubuntu/PycharmProjects/dataset/data/SOTS/outdoor/gt/' + idx[0:4] + '.png'

        name = path + idx

        gtname = gtpath + idx
        im = Image.open(name).convert("RGB")
        gt = Image.open(gtname).convert("RGB")
        # im = im.crop((0, 0, 480, 480))
        # gt = gt.crop((0, 0, 480, 480))
        im = np.asarray(im) / 255
        gt = np.asarray(gt) / 255

        input = transforms.ToTensor()(im)
        input = input.float()
        input = torch.unsqueeze(input, 0)
        # input = input.cuda()
        imgIn = input.cuda()

        # with torch.no_grad():
        #     for iteration, batch in enumerate(test_gen, 1):
        #         # print(iteration)
        #         Blur = batch[0]
        #         HR = batch[1]
        #         Blur = Blur.to(device)
        #         HR = HR.to(device)
        #
        #         name = batch[2][0][:-4]

        # start_time = time.perf_counter()#-------------------------begin to deal with an image's time

        sr = model(imgIn)

        # modify
        try:
            sr = torch.clamp(sr, min=0, max=1)
        except:
            sr = sr[0]
            sr = torch.clamp(sr, min=0, max=1)

        prediction = sr.data.cpu().numpy().squeeze().transpose((1, 2, 0))
        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.imshow(prediction)
        # plt.subplot(1, 2, 2)
        # plt.imshow(im)
        # plt.show()

        import skimage.measure
        ssim = skimage.measure.compare_ssim(gt, prediction, data_range=1, multichannel=True)
        psnr = skimage.measure.compare_psnr(gt, prediction, data_range=1)

        import imageio
        savepath = '/media/wang/1b319375-860b-4c76-b38f-ad6d60cea6fe/home/ubuntu/PycharmProjects/Thermos_Master/sot_result'
        imageio.imsave(savepath + '/msbdn/' + num + '_' + str(psnr) + '_' + str(ssim) + 'dehaze.png', prediction)

        # print(ssim)
        sca += ssim
        scp += psnr
        # c = c+1
        # print(c)

        print(sca / c)
        print(scp / c)
        print('----------------------')


def showre(model):
    finalsavepath = '/home/ubuntu/PycharmProjects/DTTD_master/ALLRESULT/'
    # transavepath = finalsavepath + 'nature/tran/'
    dehazesavepath = finalsavepath + 'nature/dehaze/'
    # gtpath = finalsavepath + 'nature/gt/'
    # gttranpath = finalsavepath + 'nature/gttran/'

    hazepath = finalsavepath + 'nature/haze/'
    files = os.listdir(hazepath)
    for idx in files:
        tag = idx.split('pdld')[0]
        name = hazepath + idx
        im = Image.open(name).convert("RGB")

        # gtname = gtpath + tag + 'gt.png'
        #
        # gt = Image.open(gtname).convert("RGB")
        #
        # gt = np.asarray(gt)
        #
        # # gt = nyu_resize(gt, 384, 1280)
        # gt = gt / 255

        haze = np.asarray(im)

        input = transforms.ToTensor()(haze)
        input = input.float()
        input = torch.unsqueeze(input, 0)
        # input = input.cuda()
        imgIn = input.cuda()

        # with torch.no_grad():
        #     for iteration, batch in enumerate(test_gen, 1):
        #         # print(iteration)
        #         Blur = batch[0]
        #         HR = batch[1]
        #         Blur = Blur.to(device)
        #         HR = HR.to(device)
        #
        #         name = batch[2][0][:-4]

        # start_time = time.perf_counter()#-------------------------begin to deal with an image's time

        sr = model(imgIn)

        # modify
        try:
            sr = torch.clamp(sr, min=0, max=1)
        except:
            sr = sr[0]
            sr = torch.clamp(sr, min=0, max=1)

        prediction = sr.data.cpu().numpy().squeeze().transpose((1, 2, 0))

        # import skimage.measure
        #
        # score = skimage.measure.compare_ssim(gt, prediction, multichannel=True)
        #
        # print(tag + '--------' + str(score) + '-------' + str(0))
        sc.imsave(dehazesavepath + tag + 'msbdndehaze' + '.png', prediction)


def evekt2015(model):
    savepath = '/home/ubuntu/PycharmProjects/dataset/kt2015/'
    savepath = '/home/ubuntu/PycharmProjects/dataset/data/citytest/0.01/'
    # res_path = '/home/ubuntu/PycharmProjects/DTTD_master/depth_eve/kt/combine/'
    sca = 0
    pa = 0
    c = 0

    for i in range(1, 265):
        # 7,17,37,47,57,67,78,87
        # i = 87

        c = c + 1
        # i = 777

        num = str(i)
        name = savepath + num + '.h5'
        f = h5py.File(name, mode='r')
        gt = f['gt'][:]
        haze = f['haze'][:]
        # gtdepth = f['gtdepth'][:]
        # spdepth = f['sdepth'][:]
        # trans = f['tran'][:]

        input = transforms.ToTensor()(haze)
        input = input.float()
        input = torch.unsqueeze(input, 0)
        # input = input.cuda()
        imgIn = input.cuda()

        # with torch.no_grad():
        #     for iteration, batch in enumerate(test_gen, 1):
        #         # print(iteration)
        #         Blur = batch[0]
        #         HR = batch[1]
        #         Blur = Blur.to(device)
        #         HR = HR.to(device)
        #
        #         name = batch[2][0][:-4]

        # start_time = time.perf_counter()#-------------------------begin to deal with an image's time

        sr = model(imgIn)

        # modify
        try:
            sr = torch.clamp(sr, min=0, max=1)
        except:
            sr = sr[0]
            sr = torch.clamp(sr, min=0, max=1)

        prediction = sr.data.cpu().numpy().squeeze().transpose((1, 2, 0))

        import skimage.measure
        ssim = skimage.measure.compare_psnr(gt, prediction, data_range=1)
        sca += ssim

        print(sca / c)


def nature_test(model):
    all = 0
    count = 0
    avg_ssim = 0
    resavepath = '/home/ubuntu/PycharmProjects/dataset/data/fradre7mthod/'
    for i in range(21):
        count = count + 1
        i = i + 45

        # i = 47

        num = str(i)
        num = num.zfill(6)
        name = '/home/ubuntu/PycharmProjects/dataset/data/frida2/haze0/U080-' + num + '.png'
        # name = '/home/ubuntu/PycharmProjects/GANet-master/training/haze_image_2/'+num+'_10.png'
        # gtname = '/home/ubuntu/PycharmProjects/GANet-master/training/new_image_2/'+num+'_10.png'
        gtname = '/home/ubuntu/PycharmProjects/dataset/data/frida2/gt/LIma-' + num + '.png'

        im = Image.open(name).convert("RGB")

        im = np.asarray(im)
        haze = im

        # haze = nyu_resize(haze, 384, 1280)

        gt = Image.open(gtname).convert("RGB")

        gt = np.asarray(gt)

        # gt = nyu_resize(gt, 384, 1280)
        gt = gt / 255

        input = transforms.ToTensor()(haze)
        input = input.float()
        input = torch.unsqueeze(input, 0)
        # input = input.cuda()
        imgIn = input.cuda()

        # with torch.no_grad():
        #     for iteration, batch in enumerate(test_gen, 1):
        #         # print(iteration)
        #         Blur = batch[0]
        #         HR = batch[1]
        #         Blur = Blur.to(device)
        #         HR = HR.to(device)
        #
        #         name = batch[2][0][:-4]

        # start_time = time.perf_counter()#-------------------------begin to deal with an image's time

        sr = model(imgIn)

        # modify
        try:
            sr = torch.clamp(sr, min=0, max=1)
        except:
            sr = sr[0]
            sr = torch.clamp(sr, min=0, max=1)

        prediction = sr.data.cpu().numpy().squeeze().transpose((1, 2, 0))
        # sc.imsave(resavepath+'47msbdn.png',prediction)

        import skimage.measure
        ssim = skimage.measure.compare_psnr(gt, prediction, data_range=1)
        avg_ssim += ssim

        print(avg_ssim / count)


def dhazetest(model):
    avg_psnr = 0
    avg_ssim = 0
    med_time = []

    input_image = '/home/ubuntu/PycharmProjects/dataset/hazy/01_hazy.png'

    # ===== Load input image =====
    transform = transforms.Compose([
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    ]
    )

    img = Image.open(input_image).convert('RGB')

    imgIn = transform(img).unsqueeze_(0)
    imgIn = imgIn.cuda()

    sr = model(imgIn)

    # modify
    try:
        sr = torch.clamp(sr, min=0, max=1)
    except:
        sr = sr[0]
        sr = torch.clamp(sr, min=0, max=1)

    prediction = sr.data.cpu().numpy().squeeze().transpose((1, 2, 0))
    print('------------')
    plt.imshow(prediction)
    plt.show()


def test(model):
    avg_ssim = 0
    sap = '/home/ubuntu/PycharmProjects/DTTD_master/val512/val512/'
    sap = '/home/ubuntu/PycharmProjects/dataset/data/midd/'
    # sap = '/home/ubuntu/PycharmProjects/dataset/data/eigntest/'
    path = sap

    files = os.listdir(path)
    score_a = 0
    num = 0
    for idx in files:
        num = num + 1
        name = path + idx
        # name = '/home/ubuntu/PycharmProjects/dataset/data/eigntest/43.h5'
        f = h5py.File(name, 'r')
        haze = f['haze'][:]
        gt = f['gt'][:]
        if np.max(gt) > 1:
            gt = gt / 255
        # print(np.max(haze))
        # print(np.max(gt))

        input = transforms.ToTensor()(haze)
        input = input.float()
        input = torch.unsqueeze(input, 0)
        # input = input.cuda()
        imgIn = input.cuda()

        # with torch.no_grad():
        #     for iteration, batch in enumerate(test_gen, 1):
        #         # print(iteration)
        #         Blur = batch[0]
        #         HR = batch[1]
        #         Blur = Blur.to(device)
        #         HR = HR.to(device)
        #
        #         name = batch[2][0][:-4]

        # start_time = time.perf_counter()#-------------------------begin to deal with an image's time

        sr = model(imgIn)

        # modify
        try:
            sr = torch.clamp(sr, min=0, max=1)
        except:
            sr = sr[0]
            sr = torch.clamp(sr, min=0, max=1)

        prediction = sr.data.cpu().numpy().squeeze().transpose((1, 2, 0))

        import skimage.measure
        ssim = skimage.measure.compare_psnr(gt, prediction, data_range=1)
        avg_ssim += ssim

        print(avg_ssim / num)

from dense_crop import *
def test4k(model):
    hazypath = '/home/lyd16/PycharmProjects/wangbin_Project/data/4K/4ktest/'
    gtpath = '/home/lyd16/PycharmProjects/wangbin_Project/data/4K/groundtrues/'

    files = os.listdir(hazypath)
    sca = 0
    scp = 0
    c = 0
    for idx in files:

        c = c + 1

        name = hazypath + idx

        gtname = gtpath + idx
        im = Image.open(name).convert("RGB")
        gt = Image.open(gtname).convert("RGB")

        im = np.asarray(im) / 255
        gt = np.asarray(gt) / 255

        im2col = image2cols(image=im, patch_size=(240, 240), stride=120)
        im2col = np.transpose(im2col, [0, 3, 1, 2])
        imgpatchs = torch.from_numpy(im2col).float()
        imgpatchs = imgpatchs.cuda()

        col2img_map = model(imgpatchs)



        col2img_map = col2img_map.cpu().detach().numpy()
        col2img_map = np.transpose(col2img_map, [0, 2, 3, 1])
        fulldehazemap = col2image(coldata=col2img_map, imsize=(3840, 2160), stride=120)
        fulldehazemap = np.transpose(fulldehazemap, [2, 0, 1])
        fulldehazemap = torch.from_numpy(fulldehazemap).float()
        # print(fulldehazemap.shape)
        dehazemap = torch.unsqueeze(fulldehazemap, dim=0)
        sr = dehazemap.cuda()

        # modify
        try:
            sr = torch.clamp(sr, min=0, max=1)
        except:
            sr = sr[0]
            sr = torch.clamp(sr, min=0, max=1)

        prediction = sr.data.cpu().numpy().squeeze().transpose((1, 2, 0))

        import skimage.measure
        ssim = skimage.measure.compare_ssim(gt, prediction, data_range=1, multichannel=True)
        psnr = skimage.measure.compare_psnr(gt, prediction, data_range=1)

        # import imageio
        # savepath = '/media/wang/1b319375-860b-4c76-b38f-ad6d60cea6fe/home/ubuntu/PycharmProjects/Thermos_Master/sot_result'
        # imageio.imsave(savepath + '/msbdn/' + num + '_' + str(psnr) + '_' + str(ssim) + 'dehaze.png', prediction)

        # print(ssim)
        sca += ssim
        scp += psnr
        # c = c+1
        # print(c)

        print(sca / c)
        print(scp / c)
        print('----------------------')


def model_test(model):
    model = model.to(device)
    criterion = torch.nn.MSELoss(size_average=True)
    criterion = criterion.to(device)
    print(opt)
    test4k(model)
    # reside_test(model,criterion)
    # test(model)
    # nature_test(model)
    # showre(model)
    # dhazetest(model)
    # evekt2015(model)
    # return psnr


opt = parser.parse_args()
# device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if torch.cuda.is_available() else torch.device('cpu')
# str_ids = opt.gpu_ids.split(',')
# torch.cuda.set_device(int(str_ids[0]))
# root_val_dir = opt.dataset# #----------Validation path
# SR_dir = join(root_val_dir, 'Results')  #--------------------------SR results save path
# isexists = os.path.exists(SR_dir)
# if not isexists:
#     os.makedirs(SR_dir)
# print("The results of testing images sotre in {}.".format(SR_dir))
#
# testloader = DataLoader(DataValSet(root_val_dir), batch_size=1, shuffle=False, pin_memory=False)
# print("===> Loading model and criterion")


if __name__ == '__main__':

    if is_pkl(opt.checkpoint):
        test_pkl = '/home/lyd16/PycharmProjects/wangbin_Project/MSBDN-DFF-master/model.pkl'
        if is_pkl(test_pkl):
            print("Testing model {}----------------------------------".format(opt.checkpoint))
            model = torch.load(test_pkl, map_location=lambda storage, loc: storage)
            print(get_n_params(model))

            # model = model.eval()

            model_test(model)
            # sot(model)
        else:
            print(
                "It's not a pkl file. Please give a correct pkl folder on command line for example --opt.checkpoint /models/1/GFN_epoch_25.pkl)")
# else:
#     test_list = [x for x in sorted(os.listdir(opt.checkpoint)) if is_pkl(x)]
#     print("Testing on the given 3-step trained model which stores in /models, and ends with pkl.")
#     Results = []
#     Max = {'max_psnr':0, 'max_epoch':0}
#     for i in range(len(test_list)):
#         print("Testing model is {}----------------------------------".format(test_list[i]))
#         print(join(opt.checkpoint, test_list[i]))
#         model = torch.load(join(opt.checkpoint, test_list[i]), map_location=lambda storage, loc: storage)
#         print(get_n_params(model))
#         model = model.eval()
#         psnr = model_test(model)
#         Results.append({'epoch':"".join(re.findall(r"/d", test_list[i])[:]), 'psnr': psnr})
#         if psnr > Max['max_psnr']:
#             Max['max_psnr'] = psnr
#             Max['max_epoch'] = "".join(re.findall(r"/d", test_list[i])[:])
#     for Result in Results:
#         print(Result)
#     print('Best Results is : ===========================> ')
#     print(Max)



