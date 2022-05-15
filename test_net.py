import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2
import time
import os
from tqdm import tqdm

from subnet import ResidualBlock2, ResidualBlock3, ResidualBlock4, ResidualBlock5, Saliency_Block
from net_module import ConvBlock
from utils import crop_img, UpsamplingBilinear2d, load_caffe_param, normalize_input
from model import RSDNET
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def inference_on_pascals():
    img_list = os.listdir(caffe_out_path)
    img_list.sort()
    print(">>> img_list； ", len(img_list))
    # tt

    rsdnet = RSDNET().float()
    rsdnet.eval()
    rsdnet.cuda()

    st = time.time()
    print(">>>> start load model ... ")
    state_dict = load_caffe_param(rsdnet, model_path)
    print(">>>> done load model, cost:  ", time.time() - st)

    rsdnet.load_state_dict(state_dict)

    all_img_time = []
    pbar = tqdm(total=len(img_list))
    with torch.no_grad():
        for i_img, name in enumerate(img_list):
            pbar.update(1)

            img_path = pascals_img_path + name[:-4] + ".jpg" # "data/imgs/266434.jpg" # (640, 480)
            # print(">>>> img_path: ", img_path)

            input_img = cv2.imread(img_path) # (480, 640, 1) > caffe: (3, 513, 513) - BGR
            H, W, _ = np.shape(input_img)

            # input_img = crop_img(input_img, crop_size)
            # cv2.imwrite("pytorch_crop.jpg", input_img)

            input_img = normalize_input(input_img)
            input_img = torch.from_numpy(np.expand_dims(input_img, 0)).permute(0, 3, 1, 2).float()  # (1, h, w, c) to (1, c, h, 3)
            

            input_img = input_img.cuda()

            st = time.time()
            output = rsdnet(input_img) # (1, 1, 376, 504) > caffe: (513, 513)
            ct = time.time() - st
            all_img_time.append(ct)

            a1 = output[0, 0, :, :].cpu().numpy()
            # print(">>>> a1: ", np.shape(a1))
            # tt

            output = UpsamplingBilinear2d(output.cpu(), H, W)
            # output = output.numpy()

            output_img = output[0] # output[0, 0:W, 0:H]
            a = 1

            output_img = output_img / np.max(output_img) * 255
            output_img = output_img.astype(np.uint8)

            ## compare with caffe
            # caffe_result_path = caffe_out_path + name
            # caffe_img = cv2.imread(caffe_result_path, 0) # gray img

            # diff = caffe_img - output_img
            # plt.imshow(diff)
            # plt.colorbar()
            # plt.savefig('diff.jpg')

            # print(">>> output_img: ", H, W, np.shape(output_img))
            # tt

            pytorch_result_path = save_out_path + "test_on_rsdnet/" + name
            cv2.imwrite(pytorch_result_path, output_img)
            # tt
        print(">>>> time: ", np.mean(all_img_time))
        pbar.close()

if __name__ == "__main__":

    model_path = "/mnt/disk10T/minglang/tmm_data/train_test_data/pretrain_models/sor_related/rsdnet_pytorch/weights.pkl"

    pascals_root = "/mnt/disk10T/minglang/tmm_data/train_test_data/database_sor/PASCAL-S/"
    pascals_img_path = pascals_root + "images/"
    
    caffe_out_path = "/mnt/disk10T/minglang/tmm_data/model_results/rsdnet_related/rsdnet_caffe/predictions_rsdnet/"
    save_out_path = "/mnt/disk10T/minglang/tmm_data/model_results/rsdnet_related/rsdnet_pytorch/"

    crop_size = 513

    inference_on_pascals()
    