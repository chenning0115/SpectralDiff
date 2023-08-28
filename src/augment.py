import torch
from torch.functional import Tensor
from torchvision import transforms
import torch.nn.functional as F
import random
import math
'''
这个是对原patch进行缩小，参数size小于原patch的size
'''

class Augment:
    def __init__(self,params) -> None:
        self.name=params['type']

    def do(self,data):
        return self.real_do(data)
    
    def real_do(self,data)->Tensor:
        pass

class ShrinkAugment(Augment):
    def __init__(self,params) -> None:
        super(ShrinkAugment,self).__init__(params)
        self.size=params.get("size",3)

    def real_do(self,data):
        # data: batch,channel,patch_size,patch_size
        batch_size=data.size(0)
        channel_num=data.size(1)
        center=int(data.size(2)/2)
        margin=int((self.size-1)/2)
        newdata=torch.zeros(data.size())
        newdata[:,:,center-margin:center+margin+1,center-margin:center+margin+1]=data[:,:,center-margin:center+margin+1,center-margin:center+margin+1]
        
        return newdata

'''
使用高斯核对每个spectrum进行模糊，参数包括kernel_size和sigma_square
在json中：
"type":"Gauss"，
"kernel_size":5
"sigma_sq":2.25
'''
class GaussAugment(Augment):
    def __init__(self,params) -> None:
        super(GaussAugment,self).__init__(params)
        self.kernel_size=params.get("kernel_size",3)
        self.sigma_sq=params.get("sigma_sq",2.25)

    def real_do(self,data):
        # data: batch,channel,patch_size,patch_size
        t=transforms.GaussianBlur(self.kernel_size,self.sigma_sq)
        newdata=t(data)
        return newdata

'''
使用在spectrum维的gaussblur
"type":"SpectralFilter"，
"kernel_size":5
"sigma_sq":2.25
'''
class SpecFilterAugment(Augment):
    def __init__(self,params) -> None:
        super(SpecFilterAugment,self).__init__(params)
        self.kernel_size=params.get("kernel_size",3)
        self.sigma_sq=params.get("sigma_sq",2.25)
        self.margin=self.kernel_size/2
        self.filter=torch.Tensor(self.kernel_size)
        for i in range(self.margin+1):
            self.filter[i]=self.filter[self.kernel_size-1-i]=-1*torch.exp((self.margin-i)*(self.margin-i)/2/self.sigma_sq)/torch.sqrt(2*torch.PI*self.sigma_sq)

    def real_do(self,data):
        # data: batch,channel,patch_size,patch_size
        batch_size=data.size(0)
        channel_num=data.size(1)
        H=data.size(2)
        W=data.size(3)
        data=torch.transpose(data,(0,2,3,1))
        newdata=torch.zeros(data.shape())
        for i in range(batch_size):
            padding_data=torch.zeros(H,W,channel_num+2*self.margin)
            padding_data[:,:,self.margin:self.margin+channel_num+1]=data[i]
            for j in range(H):
                for k in range(W):
                    for l in range(channel_num):
                        newdata[i][j][k][l]=torch.dot(self.filter,padding_data[j][k][l:l+self.kernel_size])
        data=torch.transpose(data,(0,3,1,2))
        newdata=torch.transpose(newdata,(0,3,1,2))
        return newdata

class FlipAugment(Augment):
    def __init__(self, params) -> None:
        super().__init__(params)
        self.mirror=params.get('mirror','horizontal')
    
    def real_do(self,data):# b c h w
        if self.mirror=='horizontal':
            return transforms.functional.hflip(data)
        else:
            return transforms.functional.vflip(data)

class RotateAugment(Augment):
    def __init__(self, params) -> None:
        super().__init__(params)
        self.angle=params.get('angle',90) # 默认90，也可以是270，逆时针为正

    def real_do(self, data):
        newdata=torch.transpose(data,2,3)
        if self.angle==270:
            return transforms.functional.hflip(newdata)
        else:
            return transforms.functional.vflip(newdata)

class DownSampleAugment(Augment):
    # 降采样
    def __init__(self, params) -> None:
        super().__init__(params)
        self.scale=params.get("scale",2)

    def real_do(self, data):
        x=F.interpolate(data,scale_factor=(1./self.scale,1./self.scale))
        return F.interpolate(x,size=(data.size(2),data.size(3)))

class MaskAugment(Augment):# 3D随机mask,指的是mask大小随机再加left_top点随机
    def __init__(self, params) -> None:
        super().__init__(params)
        self.max_ratio=params['max_ratio']

    def rand_mask(self,data):
        b,s,h,w=data.size()
        s_len=math.floor((1-random.random()*self.max_ratio)*s)
        s_o=random.randint(0,s_len-1)
        h_len=math.floor((1-random.random()*self.max_ratio)*h)
        h_o=random.randint(0,h_len-1)
        w_len=math.floor((1-random.random()*self.max_ratio)*w)
        w_o=random.randint(0,w_len-1)
        return s_o,h_o,w_o,s-s_len,h-h_len,w-w_len # 返回mask起始原点，以及三个维度上的mask长度

    def real_do(self,data)->Tensor:
        b,s,h,w=data.size()
        s_o1,h_o1,w_o1,s_m1,h_m1,w_m1=self.rand_mask(data)
        s_o2,h_o2,w_o2,s_m2,h_m2,w_m2=self.rand_mask(data)
        left_mask=torch.ones_like(data)
        left_mask[:,s_o1:s_o1+s_m1,h_o1:h_o1+h_m1,w_o1:w_o1+w_m1]=0
        right_mask=torch.ones_like(data)
        right_mask[:,s_o2:s_o2+s_m2,h_o2:h_o2+h_m2,w_o2:w_o2+w_m2]=0
        return data*left_mask,data*right_mask
        

class SameAugment(Augment):
    def __init__(self, params) -> None:
        super().__init__(params)
        
    def real_do(self, data) -> Tensor:
        return data,data

class XMaskAugment(Augment):
    def __init__(self, params) -> None:
        super().__init__(params)

    def real_do(self, data) -> Tensor:
        '''
        data shape is [batch, spe, h, w]
        左边 奇数mask
        右边 偶数mask
        '''
        b, s, h, w = data.shape
        left_mask = torch.zeros_like(data)
        left_mask[:,list(range(0,s,2)),:,:] = 1
        right_mask = torch.ones_like(data) - left_mask
        left = data * left_mask
        right = data * right_mask
        return left, right
        

def do_augment(params,data):# 增强也有一系列参数呢，比如multiscale的尺寸、mask的大小、Gaussian噪声的参数等
    if params['type']=='shrink':
        return ShrinkAugment(params).do(data)
    if params['type']=='Gauss':
        return GaussAugment(params).do(data)
    if params['type']=='Flip':
        return FlipAugment(params).do(data)
    if params['type']=='Rotate':
        return RotateAugment(params).do(data)
    if params["type"]=='DownSample':
        return DownSampleAugment(params).do(data)
    if params['type'] == 'Same':
        return SameAugment(params).do(data)
    if params['type'] == 'Mask':
        return XMaskAugment(params).do(data)
    if params['type'] == '3DMask':
        return MaskAugment(params).do(data)