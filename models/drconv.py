import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch.nn.functional as F
import torch.nn as nn
import torch
import torchvision.utils as vutils
from torch.autograd import Variable, Function


class asign_index(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kernel, guide_feature):
        ctx.save_for_backward(kernel, guide_feature)
        guide_mask = torch.zeros_like(guide_feature).scatter_(1, guide_feature.argmax(dim=1, keepdim=True),
                                                              1).unsqueeze(2)  # B x 3 x 1 x 25 x 25



#        mask1 = guide_mask[0][0:1]
#        mask2 = guide_mask[0][1:2]
#        mask3 = guide_mask[0][2:3]
#        mask4 = guide_mask[0][3:4]
#        mask5 = guide_mask[0][4:5]
#
#        vutils.save_image(mask1[0].cpu(),'/home/lyd16/PycharmProjects/wangbin_Project/FFA-Net-master/net/rainre/masksave/mask1.png')
#        vutils.save_image(mask2[0].cpu(),'/home/lyd16/PycharmProjects/wangbin_Project/FFA-Net-master/net/rainre/masksave/mask2.png')
#        vutils.save_image(mask3[0].cpu(),'/home/lyd16/PycharmProjects/wangbin_Project/FFA-Net-master/net/rainre/masksave/mask3.png')
#        vutils.save_image(mask4[0].cpu(),'/home/lyd16/PycharmProjects/wangbin_Project/FFA-Net-master/net/rainre/masksave/mask4.png')
#        vutils.save_image(mask5[0].cpu(),'/home/lyd16/PycharmProjects/wangbin_Project/FFA-Net-master/net/rainre/masksave/mask5.png')
#
#        print(mask1.shape)
#
#        zerosmap = torch.zeros_like(mask1)
#        # for mask1
#        newmask1 = torch.cat([mask1-0.2,zerosmap+0.2,zerosmap+0.2],dim=1)
#        # for mask2
#        newmask2 = torch.cat([zerosmap+0.2,mask2-0.2,zerosmap+0.2],dim=1)
#        # for mask3
#        newmask3 = torch.cat([zerosmap+0.2,zerosmap+0.2,mask3-0.2],dim=1)
#        # for mask4
#        newmask4 = torch.cat([mask4*0.7,mask4*0.7,zerosmap+0.2],dim=1)
#        # for mask5
#        newmask5 = torch.cat([mask5*0.7,zerosmap,mask5],dim=1)
#
#        finanalre = newmask1+newmask2+newmask3+newmask4+newmask5
#        from PIL import Image as pi
#        pi.ANTIALIAS
#
#        # allmask = torch.cat([mask1,mask2,mask3,mask4,mask5],dim=0)
#        vutils.save_image(finanalre.cpu(),'/home/lyd16/PycharmProjects/wangbin_Project/FFA-Net-master/net/rainre/masksave/dogsmask.png')
#        exit(-1)



        # vutils.save_image(guide_mask[0],'/home/lyd16/PycharmProjects/wangbin_Project/FFA-Net-master/net/rainre/sparsemask/convmask.png')
        # vutils.save_image(guide_mask[0][0],
        #                   '/home/lyd16/PycharmProjects/wangbin_Project/FFA-Net-master/net/rainre/sparsemask/Amask1.png')
        # vutils.save_image(guide_mask[0][1],
        #                   '/home/lyd16/PycharmProjects/wangbin_Project/FFA-Net-master/net/rainre/sparsemask/Amask2.png')
        # vutils.save_image(guide_mask[0][2],
        #                   '/home/lyd16/PycharmProjects/wangbin_Project/FFA-Net-master/net/rainre/sparsemask/Amask3.png')
        # vutils.save_image(guide_mask[0][3],
        #                   '/home/lyd16/PycharmProjects/wangbin_Project/FFA-Net-master/net/rainre/sparsemask/Amask4.png')
        # vutils.save_image(guide_mask[0][4],
        #                   '/home/lyd16/PycharmProjects/wangbin_Project/FFA-Net-master/net/rainre/sparsemask/Amask5.png')

        # exit(-1)

        return torch.sum(kernel * guide_mask, dim=1)

    @staticmethod
    def backward(ctx, grad_output):
        kernel, guide_feature = ctx.saved_tensors
        guide_mask = torch.zeros_like(guide_feature).scatter_(1, guide_feature.argmax(dim=1, keepdim=True),
                                                              1).unsqueeze(2)  # B x 3 x 1 x 25 x 25
        grad_kernel = grad_output.clone().unsqueeze(1) * guide_mask  # B x 3 x 256 x 25 x 25
        grad_guide = grad_output.clone().unsqueeze(1) * kernel  # B x 3 x 256 x 25 x 25
        grad_guide = grad_guide.sum(dim=2)  # B x 3 x 25 x 25
        softmax = F.softmax(guide_feature, 1)  # B x 3 x 25 x 25
        grad_guide = softmax * (grad_guide - (softmax * grad_guide).sum(dim=1, keepdim=True))  # B x 3 x 25 x 25
        return grad_kernel, grad_guide


def xcorr_slow(x, kernel,stride,padding, kwargs):
    """for loop to calculate cross correlation
    """
    batch = x.size()[0]
    out = []
    for i in range(batch):
        px = x[i]
        pk = kernel[i]
        px = px.view(1, px.size()[0], px.size()[1], px.size()[2])
        pk = pk.view(-1, px.size()[1], pk.size()[1], pk.size()[2])
        # print(px.shape)
        # print(pk.shape)
        po = F.conv2d(px, pk,stride=stride,padding=padding, **kwargs)
        out.append(po)
    out = torch.cat(out, 0)
    return out


def xcorr_fast(x, kernel,stride,padding, kwargs):
    """group conv2d to calculate cross correlation
    """
    batch = kernel.size()[0]
    pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])
    px = x.view(1, -1, x.size()[2], x.size()[3])
    # print(px.shape)
    # print(pk.shape)
    po = F.conv2d(px, pk, **kwargs, groups=batch,stride=stride,padding=padding)
    po = po.view(batch, -1, po.size()[2], po.size()[3])
    # print(po.shape)
    return po


class Corr(Function):
    @staticmethod
    def symbolic(g, x, kernel, groups):
        return g.op("Corr", x, kernel, groups_i=groups)

    @staticmethod
    def forward(self, x, kernel, stride,padding, groups, kwargs):
        """group conv2d to calculate cross correlation
        """
        batch = x.size(0)
        channel = x.size(1)
        x = x.view(1, -1, x.size(2), x.size(3))
        kernel = kernel.view(-1, channel // groups, kernel.size(2), kernel.size(3))
        # print(kernel.shape)
        out = F.conv2d(x, kernel, **kwargs,stride=stride,padding=padding, groups=groups * batch)
        out = out.view(batch, -1, out.size(2), out.size(3))
        return out


class Correlation(nn.Module):
    use_slow = True

    def __init__(self, use_slow=None):
        super(Correlation, self).__init__()
        if use_slow is not None:
            self.use_slow = use_slow
        else:
            self.use_slow = Correlation.use_slow

    def extra_repr(self):
        if self.use_slow: return "xcorr_slow"
        return "xcorr_fast"

    def forward(self, x, kernel,stride,padding, **kwargs):
        # print('------------------------------')
        if self.training:
            # print('------------------------------training')
            if self.use_slow:
                return xcorr_slow(x, kernel,stride,padding, kwargs)
            else:
                return xcorr_fast(x, kernel, stride,padding,kwargs)
        else:
            # print('-------------------------------no_train')
            return Corr.apply(x, kernel,stride,padding,1, kwargs)


class DRConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, region_num=8,stride=1,padding=1, **kwargs):
        super(DRConv2d, self).__init__()
        self.region_num = region_num
        self.stride = stride
        self.padding = padding

        self.conv_kernel = nn.Sequential(
            nn.AdaptiveAvgPool2d((kernel_size, kernel_size)),
            nn.Conv2d(in_channels, region_num * region_num, kernel_size=1),
            nn.Sigmoid(),
            nn.Conv2d(region_num * region_num, region_num * in_channels * out_channels, kernel_size=1,
                      groups=region_num)
        )
        self.conv_guide = nn.Conv2d(in_channels, region_num, kernel_size=kernel_size, stride=stride,padding=padding,**kwargs)

        self.corr = Correlation(use_slow=False)
        self.kwargs = kwargs
        self.asign_index = asign_index.apply

    def forward(self, input):
        kernel = self.conv_kernel(input)
        # print(kernel.shape)
        kernel = kernel.view(kernel.size(0), -1, kernel.size(2), kernel.size(3))  # B x (r*in*out) x W X H
        # print(kernel.shape)
        output = self.corr(input, kernel,self.stride,self.padding, **self.kwargs)  # B x (r*out) x W x H
        # print(output.shape)
        output = output.view(output.size(0), self.region_num, -1, output.size(2), output.size(3))  # B x r x out x W x H
        # print(output.shape)
        guide_feature = self.conv_guide(input)
        # print(guide_feature.shape)
        # print(output.shape)

        output = self.asign_index(output, guide_feature)
        # print(output.shape)

        return output



class DRConv2d_phy(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, region_num=8,stride=1,padding=1, **kwargs):
        super(DRConv2d_phy, self).__init__()
        self.region_num = region_num
        self.stride = stride
        self.padding = padding

        self.conv_kernel = nn.Sequential(
            nn.AdaptiveAvgPool2d((kernel_size, kernel_size)),
            nn.Conv2d(in_channels, region_num * region_num, kernel_size=1),
            nn.Sigmoid(),
            nn.Conv2d(region_num * region_num, region_num * in_channels * out_channels, kernel_size=1,
                      groups=region_num)
        )

        self.corr = Correlation(use_slow=False)
        self.kwargs = kwargs
        # self.asign_index = asign_index.apply

    def forward(self, input,phy_mask):


#        guide_mask = phy_mask
#        mask1 = guide_mask[0][0:1]
#        mask2 = guide_mask[0][1:2]
#        mask3 = guide_mask[0][2:3]
#        mask4 = guide_mask[0][3:4]
#        mask5 = guide_mask[0][4:5]
#
#        vutils.save_image(mask1[0].cpu(),'/home/lyd16/PycharmProjects/wangbin_Project/FFA-Net-master/net/rainre/masksave/maskd1.png')
#        vutils.save_image(mask2[0].cpu(),'/home/lyd16/PycharmProjects/wangbin_Project/FFA-Net-master/net/rainre/masksave/maskd2.png')
#        vutils.save_image(mask3[0].cpu(),'/home/lyd16/PycharmProjects/wangbin_Project/FFA-Net-master/net/rainre/masksave/maskd3.png')
#        vutils.save_image(mask4[0].cpu(),'/home/lyd16/PycharmProjects/wangbin_Project/FFA-Net-master/net/rainre/masksave/maskd4.png')
#        vutils.save_image(mask5[0].cpu(),'/home/lyd16/PycharmProjects/wangbin_Project/FFA-Net-master/net/rainre/masksave/maskd5.png')
#
#        print(mask1.shape)

#        zerosmap = torch.zeros_like(mask1)
#        # for mask1
#        newmask1 = torch.cat([mask1-0.2,zerosmap+0.2,zerosmap+0.2],dim=1)
#        # for mask2
#        newmask2 = torch.cat([zerosmap+0.2,mask2-0.2,zerosmap+0.2],dim=1)
#        # for mask3
#        newmask3 = torch.cat([zerosmap+0.2,zerosmap+0.2,mask3-0.2],dim=1)
#        # for mask4
#        newmask4 = torch.cat([mask4*0.7,mask4*0.7,zerosmap+0.2],dim=1)
#        # for mask5
#        newmask5 = torch.cat([mask5*0.7,zerosmap,mask5],dim=1)
#
#        finanalre = newmask1+newmask2+newmask3+newmask4+newmask5
#        from PIL import Image as pi
#        pi.ANTIALIAS
#
#        # allmask = torch.cat([mask1,mask2,mask3,mask4,mask5],dim=0)
#        vutils.save_image(finanalre.cpu(),'/home/lyd16/PycharmProjects/wangbin_Project/FFA-Net-master/net/rainre/masksave/dogdepthmask.png')
#        exit(-1)

        kernel = self.conv_kernel(input)
        # print(kernel.shape)
        kernel = kernel.view(kernel.size(0), -1, kernel.size(2), kernel.size(3))  # B x (r*in*out) x W X H
        # print(kernel.shape)
        output = self.corr(input, kernel,self.stride,self.padding, **self.kwargs)  # B x (r*out) x W x H
        # print(output.shape)
        output = output.view(output.size(0), self.region_num, -1, output.size(2), output.size(3))  # B x r x out x W x H
        # print(output.shape)
        output = torch.sum(output*phy_mask,dim=1)
        # print(output.shape)
        return output

# if __name__ == '__main__':
#     B = 1
#     in_channels = 3
#     out_channels = 3
#     size = 240
#     from PIL import Image as pi
#     import numpy as np
#
#
#     inmap = np.asarray(pi.open('/home/lyd16/PycharmProjects/wangbin_Project/FFA-Net-master/fig/1400_2.png').convert('RGB'))/255
#     inmap = np.transpose(inmap,[2,0,1])
#     intensor = torch.from_numpy(inmap).to(torch.float32)
#     intensor = torch.unsqueeze(intensor,dim=0)
#     intensor = intensor.cuda()
#
#
#
#     conv = DRConv2d(in_channels, out_channels, kernel_size=5, region_num=5,stride=1,padding=2).cuda()
#     conv.train()
#     # input = torch.ones(B, in_channels, size, size).cuda()
#     output,mask = conv(intensor)
#
#     # print(output)
#     mask = torch.transpose(mask,dim0=0,dim1=1)
#     print(mask.shape, output.shape)
#     print(mask[0])
#     vutils.save_image(mask.cpu(),'/home/lyd16/PycharmProjects/wangbin_Project/FFA-Net-master/fig/mask.png')
#     vutils.save_image(output.cpu(), '/home/lyd16/PycharmProjects/wangbin_Project/FFA-Net-master/fig/re.png')
#
#     # flops, params
#     from thop import profile
#     from thop import clever_format
#
#
#     class Conv2d(nn.Module):
#         def __init__(self):
#             super(Conv2d, self).__init__()
#             self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3)
#
#         def forward(self, input):
#             return self.conv(input)


    # conv2 = Conv2d().cuda()
    # conv2.train()
    # macs2, params2 = profile(conv2, inputs=(input,))
    # macs, params = profile(conv, inputs=(input,))
    # print(macs2, params2)
    # print(macs, params)