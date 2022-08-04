import sys
sys.path.append("..")
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
from models import *
from torchvision.models import vgg16,resnet18
from option import opt,model_name,log_dir
import torch
import torchvision.transforms as tfs
import os
from torchvision.transforms import functional as FF
from PIL import Image as pi
from matplotlib import colors






def vgg_down_dim(imagepath,vggnet):
    fea_nps = []
    imgnames = os.listdir(imagepath)
    for img_name in imgnames:
        filename = imagepath+img_name
        img = pi.open(filename).convert('RGB')
        inputmap = tfs.ToTensor()(img)
        inputmap = tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])(inputmap)
        inputmap = torch.unsqueeze(inputmap, dim=0)

        inputmap = inputmap.to(opt.device)

        # feas = vggnet.output_features(inputmap)
        with torch.no_grad():
            feas = vggnet(inputmap)
            fea_nps.append(feas)

    fea_final_np = torch.cat(fea_nps,dim=0)
    fea_final_np = fea_final_np.cpu().detach().numpy()
    return fea_final_np


def get_2dim_feas(orgin_feas):
    dowm_dim_re_x = []
    dowm_dim_re_y = []

    for item in orgin_feas:
        # ts = TSNE(n_components=2, init='pca', random_state=0)
        ts = TSNE(n_components=2)

        # t-SNE降维
        result_ = ts.fit_transform(item)
        s_x = np.sum(result_[:,0])
        s_y = np.sum(result_[:,1])
        print(s_x,s_y)
        dowm_dim_re_x.append(s_x)
        dowm_dim_re_y.append(s_y)

    return dowm_dim_re_x,dowm_dim_re_y


# 主函数，执行t-SNE降维
def main(txtpath,imgspath):
    # vgg_model = vgg16(pretrained=True).features[:16]
    # vgg_model = vgg_model.to(opt.device)
    # for param in vgg_model.parameters():
    #     param.requires_grad = False
    # VGG_Pre = PerLoss(vgg_model).to(opt.device)
    reside18 = resnet18(pretrained=True)
    reside18 = reside18.to(opt.device)
    rainfeas = vgg_down_dim(imgspath,reside18)


    # rainfeas = reside18()

    ts = TSNE(n_components=2, init='pca', random_state=0)

    # t-SNE降维
    result_ = ts.fit_transform(rainfeas)

    # x_min, x_max = np.min(result_, 0), np.max(result_, 0)
    # result_ = (result_ - x_min) / (x_max - x_min)

    rainx = result_[:,0]
    rainx = rainx.tolist()
    rainy = result_[:,1]
    rainy = rainy.tolist()
    print(len(rainx),len(rainy))

    with open(txtpath,'w') as f:
        f.write(str(rainx)+'\n'+str(rainy))

    # sotx,soty = get_2dim_feas(sotfeas)






# 主函数
if __name__ == '__main__':
    rain200hpath = '/home/lyd16/PycharmProjects/wangbin_Project/data/Rain200H/crop/test/rain/'
    SOTpath='/home/lyd16/PycharmProjects/wangbin_Project/data/SOT/lowq/'
    # rainhaze = '/home/lyd16/PycharmProjects/wangbin_Project/data/Rain200H/crop/haze/'
    rain200h_hazy_path='/home/lyd16/PycharmProjects/wangbin_Project/data/Rain200H/crop/hazerain/'
    norain200_hazy_path='/home/lyd16/PycharmProjects/wangbin_Project/data/Rain200H/crop/haze/'
    lolpath = '/home/lyd16/PycharmProjects/wangbin_Project/data/LOL/our485/lowq/'
    rain200clearpath = '/home/lyd16/PycharmProjects/wangbin_Project/data/Rain200H/crop/test/norain/'

    main(txtpath='./clear_vgg_vis.txt',imgspath=rain200clearpath)
