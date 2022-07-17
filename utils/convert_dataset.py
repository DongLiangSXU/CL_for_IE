from PIL import Image as pi
from torchvision.transforms import functional as FF
import os

# savepath = '/home/lyd16/PycharmProjects/wangbin_Project/data/unsup_test/'
# lowqsave = savepath+'lowq/'
# clearsave = savepath+'clear/'
# hazepath = '/home/lyd16/PycharmProjects/wangbin_Project/data/SOTS/'
# hazys = hazepath+'hazy/'
# haze2clear = hazepath+'clear/'
# hazymaps = os.listdir(hazys)
# num = 0
# for shaze in hazymaps:
#     num = num+1
#     hazename = hazys+shaze
#     clearname = haze2clear+shaze
#     hazy = pi.open(hazename).convert('RGB')
#     clear = pi.open(clearname).convert('RGB')
#     flagnum = int(shaze.split('.')[0])
#     if flagnum>500:
#         h,w = clear.size
#         newh,neww = (h//4)*4,(w//4)*4
#         hazy = FF.crop(hazy, 0, 0, neww, newh)
#         clear = FF.crop(clear, 0, 0, neww, newh)
#
#     savename = 'hazy_'+str(flagnum)+'.png'
#     hazy.save(lowqsave+savename)
#     clear.save(clearsave+savename)

#
rainpath = '/home/lyd16/PycharmProjects/wangbin_Project/data/Rain200L/test_L/rain/'
clear2rain = '/home/lyd16/PycharmProjects/wangbin_Project/data/Rain200L/test_L/norain/'
resavepath = '/home/lyd16/PycharmProjects/wangbin_Project/data/Rain200L/crop/test/'
rainmaps = os.listdir(rainpath)
num = 0
for srain in rainmaps:
    num = num+1
    rainname = rainpath+srain
    clearlast = srain.split('x2')[0]
    clearname = clear2rain+clearlast+'.png'

    rain = pi.open(rainname).convert('RGB')
    clear = pi.open(clearname).convert('RGB')
    h,w = rain.size[1],rain.size[0]
    # print(h,w)
    rain = FF.crop(rain, 0, 0, h-1, w-1)
    clear = FF.crop(clear, 0, 0, h-1, w-1)
    savename = 'rain_'+str(num)+'.png'
    rain.save(resavepath+'rain/'+savename)
    clear.save(resavepath+'norain/'+savename)
#
#
# llpath = '/home/lyd16/PycharmProjects/wangbin_Project/data/LOL/eval15/low/'
# clear2ll = '/home/lyd16/PycharmProjects/wangbin_Project/data/LOL/eval15/high/'
# llmaps = os.listdir(llpath)
# num = 0
# for ll in llmaps:
#     num = num+1
#     llname = llpath+ll
#     clearname = clear2ll+ll
#
#     ll = pi.open(llname).convert('RGB')
#     clear = pi.open(clearname).convert('RGB')
#
#     savename = 'lowlight_'+str(num)+'.png'
#     ll.save(lowqsave+savename)
#     clear.save(clearsave+savename)

# didrain = '/home/lyd16/PycharmProjects/wangbin_Project/data/DID_split_test/input/DID_in/'
# didlabel = '/home/lyd16/PycharmProjects/wangbin_Project/data/DID_split_test/label/DID_test_label/'
# didmaps = os.listdir(didrain)
# num = 0
# for dids in didmaps:
#     num = num+1
#     didname = didrain+dids
#     clearname = didlabel+dids
#
#     rain = pi.open(didname).convert('RGB')
#     clear = pi.open(clearname).convert('RGB')
#
#     savename = 'did_'+str(num)+'.png'
#     rain.save(lowqsave+savename)
#     clear.save(clearsave+savename)