import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock_wo_Relu(nn.Module):
    def __init__(self, setting):
        super(ConvBlock_wo_Relu, self).__init__()

        in_channel, out_channel, kernel_size, stride, padding, bias_term, dilation = setting

        # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, bias=bias_term, dilation=dilation)
        # BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn = nn.BatchNorm2d(out_channel) # scale param

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        return x


class ConvBlock(nn.Module):
    def __init__(self, setting):
        super(ConvBlock, self).__init__()

        in_channel, out_channel, kernel_size, stride, padding, bias_term, dilation = setting

        # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, bias=bias_term, dilation=dilation)
        # BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn = nn.BatchNorm2d(out_channel) # scale param
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class ResidualBlock_with_Conv(nn.Module):
    def __init__(self, branch1, branch2a, branch2b, branch2c, debug=False):
        super(ResidualBlock_with_Conv, self).__init__()
        self.debug = debug

        self.resx_branch1 = ConvBlock_wo_Relu(branch1)
 
        self.resx_branch2a = ConvBlock(branch2a)
        self.resx_branch2b = ConvBlock(branch2b)
        self.resx_branch2c = ConvBlock_wo_Relu(branch2c)

        self.relu = nn.ReLU()


    def forward(self, x):
        ## for debug
        x1 = self.resx_branch1(x)
        x2 = self.resx_branch2a(x)
        x3 = self.resx_branch2b(x2)
        x4 = self.resx_branch2c(x3)
        x5 = self.relu(x1 + x4)

        if self.debug:
            x_debug = x2.clone()
            return x5, x_debug
        else:
            return x5

        ## raw
        # x1 = self.resx_branch1(x)
        # x = self.resx_branch2a(x)
        # x = self.resx_branch2b(x)
        # x = self.resx_branch2c(x)
        # x += x1

        # if self.debug:
        #     return self.relu(x), x1
        # else:
        #     return self.relu(x)

class ResidualBlock_wo_Conv(nn.Module):
    def __init__(self, branch2a, branch2b, branch2c, debug=False):
        super(ResidualBlock_wo_Conv, self).__init__()
        self.debug = debug

        self.resx_branch2a = ConvBlock(branch2a)
        self.resx_branch2b = ConvBlock(branch2b)
        self.resx_branch2c = ConvBlock_wo_Relu(branch2c)

        self.relu = nn.ReLU()

    def forward(self, x):
        ## debug
        x1 = self.resx_branch2a(x)
        x2 = self.resx_branch2b(x1)
        x3 = self.resx_branch2c(x2)
        x += x3

        if self.debug:
            x_debug = x1.clone()
            return self.relu(x), self.relu(x_debug)
            # return self.relu(x), self.relu(x_debug)
        else:
            return self.relu(x)

        ## raw
        # x1 = self.resx_branch2a(x)
        # x1 = self.resx_branch2b(x1)
        # x1 = self.resx_branch2c(x1)
        # x1 += x

        # return self.relu(x1)