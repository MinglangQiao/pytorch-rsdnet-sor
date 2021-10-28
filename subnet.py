import torch
import torch.nn as nn
import torch.nn.functional as F

from net_module import ConvBlock, ResidualBlock_with_Conv, ResidualBlock_wo_Conv


class ResidualBlock2(nn.Module):
    def __init__(self):
        super(ResidualBlock2, self).__init__()

        # in_channel, out_channel, kernel_size, stride, padding, bias_term
        branch1 = (64, 256, 1, 1, 0, False, 1)
        branch2a = (64, 64, 1, 1, 0, False, 1)
        branch2b = (64, 64, 3, 1, 1, False, 1)
        branch2c = (64, 256, 1, 1, 0, False, 1)
        self.res2a = ResidualBlock_with_Conv(branch1, branch2a, branch2b, branch2c, True)

        branch2a_2b1 = (256, 64, 1, 1, 0, False, 1)
        self.res2b = ResidualBlock_wo_Conv(branch2a_2b1, branch2b, branch2c, True)
        self.res2c = ResidualBlock_wo_Conv(branch2a_2b1, branch2b, branch2c, True)

    def forward(self, x):
        x, x_debug = self.res2a(x)

        # x = self.res2a(x)
        x, x_debug2b = self.res2b(x)
        x, x_debug2c = self.res2c(x)

        # return x, x_debug
        # return x, x_debug2b
        return x, x_debug2c
        # return x


class ResidualBlock3(nn.Module):
    def __init__(self):
        super(ResidualBlock3, self).__init__()

        # in_channel, out_channel, kernel_size, stride, padding, bias_term, dilation
        branch1_3 = (256, 512, 1, 2, 0, False, 1)
        branch2a_3 = (256, 128, 1, 2, 0, False, 1)
        branch2b_3 = (128, 128, 3, 1, 1, False, 1)
        branch2c_3 = (128, 512, 1, 1, 0, False, 1)
        self.res3a = ResidualBlock_with_Conv(branch1_3, branch2a_3, branch2b_3, branch2c_3, True)

        branch2a_3b1 = (512, 128, 1, 1, 0, False, 1)
        self.res3b1 = ResidualBlock_wo_Conv(branch2a_3b1, branch2b_3, branch2c_3, True)
        self.res3b2 = ResidualBlock_wo_Conv(branch2a_3b1, branch2b_3, branch2c_3, True)
        self.res3b3 = ResidualBlock_wo_Conv(branch2a_3b1, branch2b_3, branch2c_3, True)

    def forward(self, x):
        x, x_debug3a = self.res3a(x)
        x, x_debug3b1 = self.res3b1(x)
        x, x_debug3b2 = self.res3b2(x)
        x, x_debug3b3 = self.res3b3(x)

        # return x, x_debug3a
        # return x, x_debug3b1
        # return x, x_debug3b2
        return x, x_debug3b3
        # return x


class ResidualBlock4(nn.Module):
    def __init__(self):
        super(ResidualBlock4, self).__init__()
       
        # branch1_4a means: branch 1 of block 4a
        res4a_branch1 = (512, 1024, 1, 1, 0, False, 1)
        res4a_branch2a = (512, 256, 1, 1, 0, False, 1)
        res4a_branch2b = (256, 256, 3, 1, 2, False, 2) # only this one is 2
        res4a_branch2c = (256, 1024, 1, 1, 0, False, 1)
        self.res4a = ResidualBlock_with_Conv(res4a_branch1, res4a_branch2a, res4a_branch2b, res4a_branch2c)

        res4b1_branch2a = (1024, 256, 1, 1, 0, False, 1)
        self.res4b1 = ResidualBlock_wo_Conv(res4b1_branch2a, res4a_branch2b, res4a_branch2c)
        self.res4b2 = ResidualBlock_wo_Conv(res4b1_branch2a, res4a_branch2b, res4a_branch2c)
        self.res4b3 = ResidualBlock_wo_Conv(res4b1_branch2a, res4a_branch2b, res4a_branch2c)
        self.res4b4 = ResidualBlock_wo_Conv(res4b1_branch2a, res4a_branch2b, res4a_branch2c)
        self.res4b5 = ResidualBlock_wo_Conv(res4b1_branch2a, res4a_branch2b, res4a_branch2c)
        self.res4b6 = ResidualBlock_wo_Conv(res4b1_branch2a, res4a_branch2b, res4a_branch2c)
        self.res4b7 = ResidualBlock_wo_Conv(res4b1_branch2a, res4a_branch2b, res4a_branch2c)
        self.res4b8 = ResidualBlock_wo_Conv(res4b1_branch2a, res4a_branch2b, res4a_branch2c)
        self.res4b9 = ResidualBlock_wo_Conv(res4b1_branch2a, res4a_branch2b, res4a_branch2c)
        self.res4b10 = ResidualBlock_wo_Conv(res4b1_branch2a, res4a_branch2b, res4a_branch2c)
        self.res4b11 = ResidualBlock_wo_Conv(res4b1_branch2a, res4a_branch2b, res4a_branch2c)
        self.res4b12 = ResidualBlock_wo_Conv(res4b1_branch2a, res4a_branch2b, res4a_branch2c)
        self.res4b13 = ResidualBlock_wo_Conv(res4b1_branch2a, res4a_branch2b, res4a_branch2c)
        self.res4b14 = ResidualBlock_wo_Conv(res4b1_branch2a, res4a_branch2b, res4a_branch2c)
        self.res4b15 = ResidualBlock_wo_Conv(res4b1_branch2a, res4a_branch2b, res4a_branch2c)
        self.res4b16 = ResidualBlock_wo_Conv(res4b1_branch2a, res4a_branch2b, res4a_branch2c)
        self.res4b17 = ResidualBlock_wo_Conv(res4b1_branch2a, res4a_branch2b, res4a_branch2c)
        self.res4b18 = ResidualBlock_wo_Conv(res4b1_branch2a, res4a_branch2b, res4a_branch2c)
        self.res4b19 = ResidualBlock_wo_Conv(res4b1_branch2a, res4a_branch2b, res4a_branch2c)
        self.res4b20 = ResidualBlock_wo_Conv(res4b1_branch2a, res4a_branch2b, res4a_branch2c)
        self.res4b21 = ResidualBlock_wo_Conv(res4b1_branch2a, res4a_branch2b, res4a_branch2c)
        self.res4b22 = ResidualBlock_wo_Conv(res4b1_branch2a, res4a_branch2b, res4a_branch2c)

    def forward(self, x):
        x = self.res4a(x)
        x = self.res4b1(x)
        x = self.res4b2(x)
        x = self.res4b3(x)
        x = self.res4b4(x)
        x = self.res4b5(x)
        x = self.res4b6(x)
        x = self.res4b7(x)
        x = self.res4b8(x)
        x = self.res4b9(x)
        x = self.res4b10(x)
        x = self.res4b11(x)
        x = self.res4b12(x)
        x = self.res4b13(x)
        x = self.res4b14(x)
        x = self.res4b15(x)
        x = self.res4b16(x)
        x = self.res4b17(x)
        x = self.res4b18(x)
        x = self.res4b19(x)
        x = self.res4b20(x)
        x = self.res4b21(x)
        x = self.res4b22(x)

        return x


class ResidualBlock5(nn.Module):
    def __init__(self):
        super(ResidualBlock5, self).__init__()

        branch1 = (1024, 2048, 1, 1, 0, False, 1)
        branch2a = (1024, 512, 1, 1, 0, False, 1)
        branch2b = (512, 512, 3, 1, 4, False, 4)
        branch2c = (512, 2048, 1, 1, 0, False, 1)
        self.res5a = ResidualBlock_with_Conv(branch1, branch2a, branch2b, branch2c)

        res5b_branch2a = (2048, 512, 1, 1, 0, False, 1)
        self.res5b = ResidualBlock_wo_Conv(res5b_branch2a, branch2b, branch2c)
        self.res5c = ResidualBlock_wo_Conv(res5b_branch2a, branch2b, branch2c)
    
    def forward(self, x):
        x = self.res5a(x)
        x = self.res5b(x)
        x = self.res5c(x)

        return x

class Saliency_Block(nn.Module):
    def __init__(self):
        super(Saliency_Block, self).__init__()
        
        self.predicted_saliency_map_0 = nn.Conv2d(2048, 12, 3, stride=1, padding=1) ## need to specify inital value
        self.predicted_saliency_map_1 = nn.Conv2d(12, 12, 3, stride=1, padding=1) ## need to specify inital value
        
        self.predicted_saliency_mask_s0 = nn.Conv2d(12, 6, 3, stride=1, padding=1)
        self.predicted_saliency_mask_s1 = nn.Conv2d(6, 3, 3, stride=1, padding=1)
        self.relu = nn.ReLU()

        self.predicted_saliency_mask = nn.Conv2d(3, 1, 1, stride=1, padding=0)
        
        
    def forward(self, x):
        x = self.predicted_saliency_map_0(x)
        x = self.predicted_saliency_map_1(x)

        x = self.relu(self.predicted_saliency_mask_s0(x))
        x = self.relu(self.predicted_saliency_mask_s1(x))

        x = self.predicted_saliency_mask(x)
        
        # self.predicted_saliency_mask_interp
        # x = F.interpolate(x, scale_factor=(8, 8), mode="bilinear", align_corners=True)
        
        return x