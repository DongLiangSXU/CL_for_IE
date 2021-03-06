import cv2
import numpy as np
import torch
from PIL import Image as pi
import matplotlib.pyplot as plt
import torchvision.transforms as tfs

def get_noise(img, value=10):
    '''
    #生成噪声图像
    # >>> 输入： img图像
    #     value= 大小控制雨滴的多少
    # >>> 返回图像大小的模糊噪声图像
    '''

    noise = np.random.uniform(0, 256, img.shape[0:2])
    # 控制噪声水平，取浮点数，只保留最大的一部分作为噪声
    v = value * 0.01
    noise[np.where(noise < (256 - v))] = 0

    # 噪声做初次模糊
    k = np.array([[0, 0.1, 0],
                  [0.1, 8, 0.1],
                  [0, 0.1, 0]])

    noise = cv2.filter2D(noise, -1, k)

    # 可以输出噪声看看
    '''cv2.imshow('img',noise)
    cv2.waitKey()
    cv2.destroyWindow('img')'''
    return noise

def rain_blur(noise, length=10, angle=0,w=1):
    '''
    将噪声加上运动模糊,模仿雨滴

    # >>>输入
    noise：输入噪声图，shape = img.shape[0:2]
    length: 对角矩阵大小，表示雨滴的长度
    angle： 倾斜的角度，逆时针为正
    w:      雨滴大小

    # >>>输出带模糊的噪声

    '''


    #这里由于对角阵自带45度的倾斜，逆时针为正，所以加了-45度的误差，保证开始为正
    trans = cv2.getRotationMatrix2D((length/2, length/2), angle-45, 1-length/100.0)
    dig = np.diag(np.ones(length))   #生成对焦矩阵
    k = cv2.warpAffine(dig, trans, (length, length))  #生成模糊核
    k = cv2.GaussianBlur(k,(w,w),0)    #高斯模糊这个旋转后的对角核，使得雨有宽度

    #k = k / length                         #是否归一化

    blurred = cv2.filter2D(noise, -1, k)    #用刚刚得到的旋转后的核，进行滤波

    #转换到0-255区间
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    '''
    cv2.imshow('img',blurred)
    cv2.waitKey()
    cv2.destroyWindow('img')'''

    return blurred


def alpha_rain(rain,img,beta = 0.8):

    #输入雨滴噪声和图像
    #beta = 0.8   #results weight
    #显示下雨效果

    #expand dimensin
    #将二维雨噪声扩张为三维单通道
    #并与图像合成在一起形成带有alpha通道的4通道图像
    rain = np.expand_dims(rain,2)
    rain_effect = np.concatenate((img,rain),axis=2)  #add alpha channel

    rain_result = img.copy()    #拷贝一个掩膜
    rain = np.array(rain,dtype=np.float32)     #数据类型变为浮点数，后面要叠加，防止数组越界要用32位
    rain_result[:,:,0]= rain_result[:,:,0] * (255-rain[:,:,0])/255.0 + beta*rain[:,:,0]
    rain_result[:,:,1] = rain_result[:,:,1] * (255-rain[:,:,0])/255 + beta*rain[:,:,0]
    rain_result[:,:,2] = rain_result[:,:,2] * (255-rain[:,:,0])/255 + beta*rain[:,:,0]

    #对每个通道先保留雨滴噪声图对应的黑色（透明）部分，再叠加白色的雨滴噪声部分（有比例因子）
    return rain_result

    # cv2.imshow('rain_effct_result',rain_result)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

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

import random
def add_rain(tensor):
    # angle,density,length,width
    angle = random.randint(-15,15)
    density = random.randint(50,100)
    length = random.randint(50,70)
    widthnum = [1,3,5]
    width = random.choice(widthnum)

    # 角度在-15到15,密度从150-200之间，length从20到50之间，宽度从1和3选择
    # hazemappath = 'G:/workspace/Dark/darkface_test/acer/00001.png'
    # hazemappath = 'G:/workspace/Dark/darkface_test/tosys_haze_lowlight/haze/1.jpg'
    hazenp = np.asarray(torch_to_PIL(tensor))
    # img = cv2.imread('demo.jpeg')
    img = hazenp
    noise = get_noise(img,value=density)
    rain = rain_blur(noise,length=length,angle=angle,w=width)
    rain_map = alpha_rain(rain,img,beta=1)  #方法一，透明度赋值

    # rain_map = np.transpose(rain_map,[2,0,1])
    rain_tensor = pi.fromarray(rain_map.astype('uint8')).convert('RGB')
    rain_tensor = tfs.ToTensor()(rain_tensor)
    rain_tensor = torch.unsqueeze(rain_tensor,dim=0)
    return rain_tensor



if __name__ == '__main__':

    # angle,density,length,width
    angle = random.randint(-45,45)
    density = random.randint(500,900)
    length = random.randint(20,50)
    widthnum = [5]
    width = random.choice(widthnum)

    # 角度在-45到45,密度从50-500之间，length从20到50之间，宽度从1和3选择
    hazemappath = 'G:/workspace/Dark/darkface_test/acer/00077.png'
    # hazemappath = 'G:/workspace/Dark/darkface_test/tosys_haze_lowlight/haze/1.jpg'
    hazenp = np.asarray(pi.open(hazemappath).convert('RGB'))
    # img = cv2.imread('demo.jpeg')
    img = hazenp
    noise = get_noise(img,value=density)
    rain = rain_blur(noise,length=length,angle=angle,w=width)
    rain_map = alpha_rain(rain,img,beta=1)  #方法一，透明度赋值

    plt.imshow(rain_map)
    plt.show()
    #add_rain(rain,img)    #方法二,加权后有玻璃外的效果
