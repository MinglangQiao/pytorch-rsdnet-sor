import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2
import time

from subnet import ResidualBlock2, ResidualBlock3, ResidualBlock4, ResidualBlock5, Saliency_Block
from net_module import ConvBlock
from utils import crop_img, UpsamplingBilinear2d, load_caffe_param, normalize_input


class RSDNET(nn.Module):
    def __init__(self):
        super(RSDNET, self).__init__()
        
        # in_channel, out_channel, kernel_size, stride, padding, bias_term, dilation
        conv1_setting = (3, 64, 7, 2, 3, False, 1)
        self.conv1 = ConvBlock(conv1_setting)
        self.pool1 = nn.MaxPool2d(3, 2, 1) # kernel_size, stride=None, padding

        self.res2 = ResidualBlock2()
        self.res3 = ResidualBlock3()
        self.res4 = ResidualBlock4()
        self.res5 = ResidualBlock5()

        self.output = Saliency_Block()

    def forward(self, x):
        x0 = self.pool1(self.conv1(x))
        
        x1, x_debug2 = self.res2(x0)
        x2, x_debug3 = self.res3(x1)
        x3 = self.res4(x2)
        x4 = self.res5(x3)
        x5 = self.output(x4) 
        
        # return x0
        # return x1
        # return x2
        
        # x5 = F.interpolate(x5, scale_factor=8, mode='bilinear')
        return x5
        
        # return x_debug2
        # return x_debug3
        



if __name__ == "__main__":

    model_path = "/temp_disk2/leise/ml/tmm_ref/rank_related/rsdnet/scripts/inference/weights.pkl"

    crop_size = 513
    print(">>>>> hello !")

    img_path = "data/imgs/266434.jpg" # (640, 480)
    input_img = cv2.imread(img_path) # (480, 640, 1) > caffe: (3, 513, 513) - BGR
    H, W, _ = np.shape(input_img)

    input_img = crop_img(input_img, crop_size)
    cv2.imwrite("pytorch_crop.jpg", input_img)

    input_img = normalize_input(input_img)
    input_img = torch.from_numpy(np.expand_dims(input_img, 0)).permute(0, 3, 1, 2).float()  # (1, h, w, c) to (1, c, h, 3)
    
    rsdnet = RSDNET().float()
    rsdnet.eval()

    st = time.time()
    print(">>>> start load model ... ")
    state_dict = load_caffe_param(rsdnet, model_path)
    print(">>>> done load model, cost:  ", time.time() - st)

    rsdnet.load_state_dict(state_dict)

    with torch.no_grad():
        output = rsdnet(input_img) # (1, 1, 376, 504) > caffe: (513, 513)

        a1 = output[0, 0, :, :].numpy()

        output = UpsamplingBilinear2d(output)
        # output = output.numpy()

        output_img = output[0, 0:H, 0:W]
        a = 1

        output_img = output_img / np.max(output_img) * 255
        output_img = output_img.astype(np.uint8)

    ## compare with caffe
    import matplotlib.pyplot as plt
    caffe_result_path = "/home/ml/tmm_ref/rank_related/rsdnet/saliency_maps_salsod_rsdnet_1/266434.png"
    caffe_img = cv2.imread(caffe_result_path, 0) # gray img

    cv2.imwrite('pytorch_out.png', output_img)

    diff = caffe_img - output_img
    plt.imshow(diff)
    plt.colorbar()
    plt.savefig('diff.jpg')