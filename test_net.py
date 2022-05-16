
import torch

import numpy as np
import cv2
import time
import os
from tqdm import tqdm


from utils import crop_img, UpsamplingBilinear2d, load_caffe_param, normalize_input
from model import RSDNET
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

            input_img = cv2.imread(img_path) # (480, 640, 1) > caffe: (3, 513, 513) - BGR
            H, W, _ = np.shape(input_img)
            # if H > 513 or W > 513:
            #     tt

            st = time.time()
            input_img = normalize_input(input_img)
            input_img = crop_img(input_img, crop_size)
            
            input_img = torch.from_numpy(np.expand_dims(input_img, 0)).permute(0, 3, 1, 2).float()  # (1, h, w, c) to (1, c, h, 3)
            input_img = input_img.cuda()

            output = rsdnet(input_img) # (1, 1, 376, 504) > caffe: (513, 513)

            output = UpsamplingBilinear2d(output.cpu())
            ## using caffe output
            
            ct = time.time() - st
            all_img_time.append(ct)

            output_img = output[0, 0:H, 0:W] # output[0]
            
            pytorch_result_path = pytorch_result_dir + name
            misc.toimage(output_img, cmin = 0.0, cmax = 255).save(pytorch_result_path)
            # tt
        print(">>>> time: ", np.mean(all_img_time), len(all_img_time))
        pbar.close()

if __name__ == "__main__":

    model_path = "xx/weights.pkl"

    pascals_root = "xx/PASCAL-S/"
    pascals_img_path = pascals_root + "images/"
    
    caffe_out_path = "xx/" # path to images for testing
    save_out_path = "xx/" # path for output

    inference_on_pascals()
    