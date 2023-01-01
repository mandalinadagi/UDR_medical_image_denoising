##### originated from https://github.com/terryoo/AINDNet/blob/master/code/model.py 
##### it was written in tensorflow, converted to pytorch code and changed

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class FCN_Avg(nn.Module):
    def __init__(self, in_nc, nf=32):
        super(FCN_Avg, self).__init__()
        self.conv1 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.avg_pool4 = nn.AvgPool2d(4)

        self.conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.avg_pool2 = nn.AvgPool2d(2)
        self.conv5 = nn.Conv2d(nf, 3, 3, 1, 1, bias=True)
        
        self.conv6 = nn.Conv2d(3, 3, 3, 1, 1, bias=True)
        self.conv7 = nn.Conv2d(3, in_nc, 3, 1, 1, bias=True)
        self.relu = nn.ReLU()

    def forward(self, inp):
        x = self.relu(self.conv1(inp))
        x = self.relu(self.conv2(x))
        x = self.avg_pool4(x)

        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.avg_pool2(x)
        x = self.relu(self.conv5(x))

        scale = (inp.shape[2])/(x.shape[2])
        y = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=True)
        y = self.relu(self.conv6(y))
        y = self.relu(self.conv7(y))

        return x, y 

class ResBlock(nn.Module):
    def __init__(self, in_nc, out_nc):
        super(ResBlock, self).__init__()  
        
        self.conv1 = nn.Conv2d(in_nc, out_nc, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(out_nc, in_nc, 3, 1, 1, bias=True)
        self.bn    = nn.BatchNorm2d(out_nc)
        self.relu  = nn.ReLU()
    
    def forward(self, x):
        fea = self.conv2(self.relu(self.bn(self.conv1(x))))
        return fea + x


class ResUpsampleSum(nn.Module):
    def __init__(self, in_nc, out_nc):
        super(ResUpsampleSum, self).__init__() 
        self.resblock = ResBlock(in_nc, out_nc)
        self.transpose_conv = nn.ConvTranspose2d(out_nc, in_nc, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        rb = self.resblock(x2)
        deconv = self.transpose_conv(x1)
        deconv_out  = deconv + rb 
        return deconv_out


def param_free_norm(x, epsilon=1e-5):
    x_var, x_mean = torch.var_mean(x, [2,3], unbiased=False, keepdim = True)
    x_std = torch.sqrt(x_var + epsilon)
    return (x - x_mean) / x_std   


class ain(nn.Module):
    def __init__(self, nf):
        super(ain, self).__init__()
        self.conv1 = nn.Conv2d(3, nf, 5, 1, 2, bias=True)
        self.conv_gamma = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_beta = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.relu = nn.ReLU()

    def forward(self, noise_map, inp):
        x = param_free_norm(inp)
        scale = (inp.shape[2])/(noise_map.shape[2])
        noise_map_down = F.interpolate(noise_map, scale_factor=scale, mode='bilinear', align_corners=True)
        tmp = self.relu(self.conv1(noise_map_down))
        noisemap_gamma = self.conv_gamma(tmp)
        noisemap_beta = self.conv_beta(tmp)
       
        x = x * (1 + noisemap_gamma) + noisemap_beta
        return x    


class AIN_ResBlock(nn.Module):
    def __init__(self, nf):
        super(AIN_ResBlock, self).__init__()  
        self.ain1 = ain(nf)
        self.lrelu = nn.LeakyReLU(negative_slope=0.02, inplace=True)
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.ain2 = ain(nf)
        self.conv2= nn.Conv2d(nf, nf, 3, 1, 1, bias=True)


    def forward(self, noise_map, inp):
        x = self.lrelu(self.ain1(noise_map, inp))
        x = self.conv1(x)
        x = self.lrelu(self.ain2(noise_map, x))
        x = self.conv2(x)

        return x + inp


class AINDNet_recon(nn.Module):
    def __init__(self, in_nc, out_nc):
        super(AINDNet_recon, self).__init__()
        self.conv1 = nn.Conv2d(in_nc, 64, 3, 1, 1, bias=True)
        self.ain_rb1_1 = AIN_ResBlock(64)
        self.ain_rb1_2 = AIN_ResBlock(64)
        self.avg_pool2 = nn.AvgPool2d(2)

        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1, bias=True)
        self.ain_rb2_1 = AIN_ResBlock(128)
        self.ain_rb2_2 = AIN_ResBlock(128)

        self.conv3 = nn.Conv2d(128, 256, 3, 1, 1, bias=True)
        self.ain_rb3_1 = AIN_ResBlock(256)
        self.ain_rb3_2 = AIN_ResBlock(256)
        self.ain_rb3_3 = AIN_ResBlock(256)
        self.ain_rb3_4 = AIN_ResBlock(256)
        self.ain_rb3_5 = AIN_ResBlock(256)

        self.res_up1 = ResUpsampleSum(128, 256)
        self.ain_rb4_1 = AIN_ResBlock(128)
        self.ain_rb4_2 = AIN_ResBlock(128)
        self.ain_rb4_3 = AIN_ResBlock(128)

        self.res_up2 = ResUpsampleSum(64, 128)
        self.ain_rb5_1 = AIN_ResBlock(64)
        self.ain_rb5_2 = AIN_ResBlock(64)

        self.conv_out = nn.Conv2d(64, out_nc, 3, 1, 1, bias=True)
        self.relu = nn.ReLU()

    def forward(self, inp, noise_map):
        x = self.relu(self.conv1(inp))
        rb1 = self.ain_rb1_1(noise_map, x)
        rb2 = self.ain_rb1_2(noise_map, rb1)
        x_down = self.avg_pool2(rb2)

        x2 = self.relu(self.conv2(x_down))
        rb21 = self.ain_rb2_1(noise_map, x2)
        rb22 = self.ain_rb2_2(noise_map, rb21)
        x_down2 = self.avg_pool2(rb22)

        x3 = self.relu(self.conv3(x_down2))
        rb31 = self.ain_rb3_1(noise_map, x3)
        rb32 = self.ain_rb3_2(noise_map, rb31)
        rb33 = self.ain_rb3_3(noise_map, rb32)
        rb34 = self.ain_rb3_4(noise_map, rb33)
        rb35 = self.ain_rb3_5(noise_map, rb34)

        ups1 = self.res_up1(rb35, rb22)
        rb41 = self.ain_rb4_1(noise_map, ups1)
        rb42 = self.ain_rb4_2(noise_map, rb41)
        rb43 = self.ain_rb4_3(noise_map, rb42)

        ups2 = self.res_up2(rb43, rb2)
        rb51 = self.ain_rb5_1(noise_map, ups2)
        rb52 = self.ain_rb5_2(noise_map, rb51)

        out = self.conv_out(rb51)
        return out


class AINSRNet(nn.Module):
    def __init__(self, in_nc, out_nc):
        super(AINSRNet, self).__init__()
        self.fcn = FCN_Avg(in_nc)
        self.ainnet = AINSRNet_recon(in_nc, out_nc)

    def forward(self, inp_p, inp_o):
        down_noise_map, noise_map = self.fcn(inp)
        scale = (inp.shape[2]) /(down_noise_map.shape[2])
        upsample_noise_map = F.interpolate(down_noise_map, scale_factor=scale, mode='bilinear', align_corners=True)
        noise_map = 0.8 *upsample_noise_map + 0.2*noise_map
        out = self.ainnet(inp, noise_map) + inp
        return noise_map, out


