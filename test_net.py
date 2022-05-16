from tkinter import ttk
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
from scipy import misc

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
crop_size = 513

def inference_on_pascals():
    pytorch_result_dir = save_out_path + "test_on_rsdnet/"
    if not os.path.exists(pytorch_result_dir):
        os.makedirs(pytorch_result_dir)
        
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
            
            # if not name[:-4] in ['10']:
            #     continue

            img_path = pascals_img_path + name[:-4] + ".jpg" # "data/imgs/266434.jpg" # (640, 480)
            # print(">>>> img_path: ", img_path)

            input_img = cv2.imread(img_path) # (480, 640, 1) > caffe: (3, 513, 513) - BGR
            H, W, _ = np.shape(input_img)
            # if H > 513 or W > 513:
            #     tt

            st = time.time()
            input_img = normalize_input(input_img)
            input_img = crop_img(input_img, crop_size)
            
            # caff_input = np.load("/mnt/disk10T/minglang/tmm_ref/rank_related/rsdnet/caffe_crop.npy")
            # diff = input_img - caff_input
            # diff_value = np.sum(diff)

            # show_diff = np.concatenate((diff[:, :, 0], diff[:, :, 1], diff[:, :, 2]), axis=1)
            # plt.imshow(show_diff)
            # plt.colorbar()
            # plt.savefig("diff_value_of_input.png")
            # cv2.imwrite("pytorch_crop.png", input_img.astype(np.uint8))
            # tt

            input_img = torch.from_numpy(np.expand_dims(input_img, 0)).permute(0, 3, 1, 2).float()  # (1, h, w, c) to (1, c, h, 3)
            input_img = input_img.cuda()

            output = rsdnet(input_img) # (1, 1, 376, 504) > caffe: (513, 513)
            # a1 = output[0, 0, :, :].cpu().numpy()
            # cv2.imwrite("./out_10_pytorch.png", a1.astype(np.uint8))

            # caff_output = np.load("/mnt/disk10T/minglang/tmm_ref/rank_related/rsdnet/out_10_caffe.npy")
            # diff = a1 - caff_output
            # diff_value = np.sum(np.abs(diff))
            # show_diff = diff
            # plt.imshow(show_diff)
            # plt.colorbar()
            # plt.savefig("diff_value_of_output.png")

            # print(">>>> a1: ", np.shape(a1))
            # tt

            # caff_output = np.load("/mnt/disk10T/minglang/tmm_ref/rank_related/rsdnet/out_10_caffe.npy")
            # output = UpsamplingBilinear2d(torch.from_numpy(np.array([[caff_output]])))

            output = UpsamplingBilinear2d(output.cpu())
            ## using caffe output
            
            ct = time.time() - st
            all_img_time.append(ct)

            # output = output.numpy()

            output_img = output[0, 0:H, 0:W] # output[0]
            # output_img = output[0, :, :] # output[0]
            
            # a1 = output_img
            # caff_output = np.load("/mnt/disk10T/minglang/tmm_ref/rank_related/rsdnet/out_inter_caffe.npy")
            # diff = a1 - caff_output
            # diff_value = np.sum(np.abs(diff))
            # show_diff = diff
            # cv2.imwrite("./out_inter_pytorch.png", output_img.astype(np.uint8))
            # plt.imshow(show_diff)
            # plt.colorbar()
            # plt.savefig("diff_value_of_output.png")
            
            ## compare with caffe
            # caffe_result_path = caffe_out_path + name
            # caffe_img = cv2.imread(caffe_result_path, 0) # gray img

            # diff = caffe_img - output_img
            # plt.imshow(diff)
            # plt.colorbar()
            # plt.savefig('diff.jpg')

            # print(">>> output_img: ", H, W, np.shape(output_img))
            # tt

            
            pytorch_result_path = pytorch_result_dir + name
            # cv2.imwrite(pytorch_result_path, output_img)
            misc.toimage(output_img, cmin = 0.0, cmax = 255).save(pytorch_result_path)
            # tt
        print(">>>> time: ", np.mean(all_img_time), len(all_img_time))
        pbar.close()

if __name__ == "__main__":

    model_path = "/mnt/disk10T/minglang/tmm_data/train_test_data/pretrain_models/sor_related/rsdnet_pytorch/weights.pkl"

    pascals_root = "/mnt/disk10T/minglang/tmm_data/train_test_data/database_sor/PASCAL-S/"
    pascals_img_path = pascals_root + "images/"
    
    caffe_out_path = "/mnt/disk10T/minglang/tmm_data/model_results/rsdnet_related/rsdnet_caffe/predictions_rsdnet/"
    save_out_path = "/mnt/disk10T/minglang/tmm_data/model_results/rsdnet_related/rsdnet_pytorch_v1_interp_by_pytorch/"

    inference_on_pascals()
    