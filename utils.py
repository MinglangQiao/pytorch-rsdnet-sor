import numpy as np
import cv2
import pickle as pkl

import torch
from torch import from_numpy

def normalize_input(input_img, MEAN_VAL=[104.00, 116.669, 122.675]):
    """
    From: https://github.com/islamamirul/rsdnet/blob/84626d66f33d862888029aed61b19ff0f327f3dc/models/test_rsdnet.prototxt#L17
    mean_value: 104.00 - B
    mean_value: 116.669 - G
    mean_value: 122.675 - R
    """
    input_img = input_img.astype(np.float32) - MEAN_VAL

    return input_img

def crop_img(input_img, crop_size):
    """
    reference: https://github.com/xiamenwcy/extended-caffe/blob/master/src/caffe/data_transformer.cpp
    """
    height, width, _ = np.shape(input_img)

    min_size = np.min([height, width])
    
    if min_size >= crop_size:
        h_off = int((height - crop_size) / 2)
        w_off = int((width - crop_size) / 2)
        input_img = input_img[h_off:(h_off+crop_size), w_off:(w_off+crop_size), :]

    else:
        pad_height = np.max([crop_size - height, 0])
        pad_width = np.max([crop_size - width, 0])
        if pad_height > 0 or pad_width > 0:
            input_img = np.pad(input_img, ((0, pad_height), (0, pad_width), (0, 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))

        # Then crop
        height1, width1, _ = np.shape(input_img)

        h_off = int((height1 - crop_size) / 2)
        w_off = int((width1 - crop_size) / 2)
        input_img = input_img[h_off:(h_off+crop_size), w_off:(w_off+crop_size), :]

    return input_img


def UpsamplingBilinear2d(input_tensor, zoom_factor=8):
    """
    reference: https://github.com/gjs94/YOLOv3-caffe/blob/master/interp.py
    """
    # from scipy import interpolate
    # import scipy.ndimage.interpolation.zoom as scipy_interp
    
    input_np = input_tensor.numpy()
    b, c, h_out, w_out = np.shape(input_np)
    w_out = w_out + (w_out - 1) * (zoom_factor - 1)
    h_out = h_out + (h_out - 1) * (zoom_factor - 1)

    out_imgs = []
    for i_img, img in enumerate(input_np):
        img = np.transpose(img, (1, 2, 0))
        img = cv2.resize(img, (w_out, h_out), interpolation=cv2.INTER_LINEAR)
        out_imgs.append(img)

    return np.array(out_imgs)


def load_caffe_param(net, model_path):
    """
    reference: https://github.com/Mannix1994/SfSNet-Pytorch/
    """
    index = 0
    all_layer_name = []
    for name, param in list(net.named_parameters()):
        # print(str(index) + '\t', name, '\t', param.size())
        index += 1
        all_layer_name.append(name)

    state_dict = {}
    with open(model_path, 'rb') as wp:
        name_weights = pkl.load(wp, encoding='latin1')

    def _set_with_bias(layer, key):
        state_dict[layer + '.weight'] = from_numpy(name_weights[key]['weight'])
        state_dict[layer + '.bias'] = from_numpy(name_weights[key]['bias'])

    def _set_wo_bias(layer, key):
        state_dict[layer + '.weight'] = from_numpy(name_weights[key]['weight'])

    def _set_bn(layer, key_mean_var, key_scale):
        state_dict[layer + '.running_var'] = from_numpy(name_weights[key_mean_var]['running_var'])
        state_dict[layer + '.running_mean'] = from_numpy(name_weights[key_mean_var]['running_mean'])
        state_dict[layer + '.weight'] = from_numpy(name_weights[key_scale]['weight'])
        state_dict[layer + '.bias'] = from_numpy(name_weights[key_scale]['bias'])

    def _set_res_with_conv(layer, key_conv, key_mean_var, key_scale):
        # branch1
        _set_wo_bias(layer + '.resx_branch1.conv', key_conv + "_branch1")
        _set_bn(layer+'.resx_branch1.bn', key_mean_var + "_branch1", key_scale + "_branch1")

        # branch2a
        _set_wo_bias(layer + '.resx_branch2a.conv', key_conv + "_branch2a")
        _set_bn(layer+'.resx_branch2a.bn', key_mean_var + "_branch2a", key_scale + "_branch2a")
        # branch2b
        _set_wo_bias(layer + '.resx_branch2b.conv', key_conv + "_branch2b")
        _set_bn(layer+'.resx_branch2b.bn', key_mean_var + "_branch2b", key_scale + "_branch2b")
        # branch2c
        _set_wo_bias(layer + '.resx_branch2c.conv', key_conv + "_branch2c")
        _set_bn(layer+'.resx_branch2c.bn', key_mean_var + "_branch2c", key_scale + "_branch2c")

    def _set_res_wo_conv(layer, key_conv, key_mean_var, key_scale):
        # branch2a
        _set_wo_bias(layer + '.resx_branch2a.conv', key_conv + "_branch2a")
        _set_bn(layer+'.resx_branch2a.bn', key_mean_var + "_branch2a", key_scale + "_branch2a")
        # branch2b
        _set_wo_bias(layer + '.resx_branch2b.conv', key_conv + "_branch2b")
        _set_bn(layer+'.resx_branch2b.bn', key_mean_var + "_branch2b", key_scale + "_branch2b")
        # branch2c
        _set_wo_bias(layer + '.resx_branch2c.conv', key_conv + "_branch2c")
        _set_bn(layer+'.resx_branch2c.bn', key_mean_var + "_branch2c", key_scale + "_branch2c")

    ## conv1
    _set_wo_bias('conv1.conv', 'conv1')
    _set_bn('conv1.bn', 'bn_conv1', 'scale_conv1')
    
    ## res2
    _set_res_with_conv("res2.res2a", "res2a", "bn2a", "scale2a")
    _set_res_wo_conv("res2.res2b", "res2b", "bn2b", "scale2b")
    _set_res_wo_conv("res2.res2c", "res2c", "bn2c", "scale2c")
    
    ## res3
    res3, layer0, layer1, layer2, layer3 = '3', '3a', '3b1', '3b2', '3b3'
    _set_res_with_conv("res{}.res{}".format(res3, layer0), "res{}".format(layer0), "bn{}".format(layer0), "scale{}".format(layer0))
    _set_res_wo_conv("res{}.res{}".format(res3, layer1), "res{}".format(layer1), "bn{}".format(layer1), "scale{}".format(layer1))
    _set_res_wo_conv("res{}.res{}".format(res3, layer2), "res{}".format(layer2), "bn{}".format(layer2), "scale{}".format(layer2))
    _set_res_wo_conv("res{}.res{}".format(res3, layer3), "res{}".format(layer3), "bn{}".format(layer3), "scale{}".format(layer3))

    ## res4
    res4, layer4_0, layer4_1, layer4_2, layer4_3, layer4_4, layer4_5, layer4_6, layer4_7, layer4_8, layer4_9, layer4_10  = '4', '4a', '4b1', '4b2', '4b3', '4b4', '4b5', '4b6', '4b7', '4b8', '4b9', '4b10'
    layer4_11, layer4_12, layer4_13, layer4_14, layer4_15, layer4_16, layer4_17, layer4_18, layer4_19, layer4_20, layer4_21, layer4_22 = '4b11', '4b12', '4b13', '4b14', '4b15', '4b16', '4b17', '4b18', '4b19', '4b20', '4b21', '4b22'
    _set_res_with_conv("res{}.res{}".format(res4, layer4_0), "res{}".format(layer4_0), "bn{}".format(layer4_0), "scale{}".format(layer4_0))
    _set_res_wo_conv("res{}.res{}".format(res4, layer4_1), "res{}".format(layer4_1), "bn{}".format(layer4_1), "scale{}".format(layer4_1))
    _set_res_wo_conv("res{}.res{}".format(res4, layer4_2), "res{}".format(layer4_2), "bn{}".format(layer4_2), "scale{}".format(layer4_2))
    _set_res_wo_conv("res{}.res{}".format(res4, layer4_3), "res{}".format(layer4_3), "bn{}".format(layer4_3), "scale{}".format(layer4_3))
    _set_res_wo_conv("res{}.res{}".format(res4, layer4_4), "res{}".format(layer4_4), "bn{}".format(layer4_4), "scale{}".format(layer4_4))
    _set_res_wo_conv("res{}.res{}".format(res4, layer4_5), "res{}".format(layer4_5), "bn{}".format(layer4_5), "scale{}".format(layer4_5))
    _set_res_wo_conv("res{}.res{}".format(res4, layer4_6), "res{}".format(layer4_6), "bn{}".format(layer4_6), "scale{}".format(layer4_6))
    _set_res_wo_conv("res{}.res{}".format(res4, layer4_7), "res{}".format(layer4_7), "bn{}".format(layer4_7), "scale{}".format(layer4_7))
    _set_res_wo_conv("res{}.res{}".format(res4, layer4_8), "res{}".format(layer4_8), "bn{}".format(layer4_8), "scale{}".format(layer4_8))
    _set_res_wo_conv("res{}.res{}".format(res4, layer4_9), "res{}".format(layer4_9), "bn{}".format(layer4_9), "scale{}".format(layer4_9))
    _set_res_wo_conv("res{}.res{}".format(res4, layer4_10), "res{}".format(layer4_10), "bn{}".format(layer4_10), "scale{}".format(layer4_10))
    _set_res_wo_conv("res{}.res{}".format(res4, layer4_11), "res{}".format(layer4_11), "bn{}".format(layer4_11), "scale{}".format(layer4_11))
    _set_res_wo_conv("res{}.res{}".format(res4, layer4_12), "res{}".format(layer4_12), "bn{}".format(layer4_12), "scale{}".format(layer4_12))
    _set_res_wo_conv("res{}.res{}".format(res4, layer4_13), "res{}".format(layer4_13), "bn{}".format(layer4_13), "scale{}".format(layer4_13))
    _set_res_wo_conv("res{}.res{}".format(res4, layer4_14), "res{}".format(layer4_14), "bn{}".format(layer4_14), "scale{}".format(layer4_14))
    _set_res_wo_conv("res{}.res{}".format(res4, layer4_15), "res{}".format(layer4_15), "bn{}".format(layer4_15), "scale{}".format(layer4_15))
    _set_res_wo_conv("res{}.res{}".format(res4, layer4_16), "res{}".format(layer4_16), "bn{}".format(layer4_16), "scale{}".format(layer4_16))
    _set_res_wo_conv("res{}.res{}".format(res4, layer4_17), "res{}".format(layer4_17), "bn{}".format(layer4_17), "scale{}".format(layer4_17))
    _set_res_wo_conv("res{}.res{}".format(res4, layer4_18), "res{}".format(layer4_18), "bn{}".format(layer4_18), "scale{}".format(layer4_18))
    _set_res_wo_conv("res{}.res{}".format(res4, layer4_19), "res{}".format(layer4_19), "bn{}".format(layer4_19), "scale{}".format(layer4_19))
    _set_res_wo_conv("res{}.res{}".format(res4, layer4_20), "res{}".format(layer4_20), "bn{}".format(layer4_20), "scale{}".format(layer4_20))
    _set_res_wo_conv("res{}.res{}".format(res4, layer4_21), "res{}".format(layer4_21), "bn{}".format(layer4_21), "scale{}".format(layer4_21))
    _set_res_wo_conv("res{}.res{}".format(res4, layer4_22), "res{}".format(layer4_22), "bn{}".format(layer4_22), "scale{}".format(layer4_22))

    ## res5
    res5, layer5_0, layer5_1 , layer5_2  = '5', '5a', '5b', '5c'
    _set_res_with_conv("res{}.res{}".format(res5, layer5_0), "res{}".format(layer5_0), "bn{}".format(layer5_0), "scale{}".format(layer5_0))
    _set_res_wo_conv("res{}.res{}".format(res5, layer5_1), "res{}".format(layer5_1), "bn{}".format(layer5_1), "scale{}".format(layer5_1))
    _set_res_wo_conv("res{}.res{}".format(res5, layer5_2), "res{}".format(layer5_2), "bn{}".format(layer5_2), "scale{}".format(layer5_2))

    # saliency
    _set_with_bias("output.predicted_saliency_map_0", "predicted_saliency_map_0")
    _set_with_bias("output.predicted_saliency_map_1", "predicted_saliency_map_1")
    _set_with_bias("output.predicted_saliency_mask_s0", "predicted_saliency_mask_s0")
    _set_with_bias("output.predicted_saliency_mask_s1", "predicted_saliency_mask_s1")
    _set_with_bias("output.predicted_saliency_mask", "predicted_saliency_mask")

    return state_dict


def analysis_diff():
    import matplotlib.pyplot as plt

    caffe_path = "/temp_disk2/ml/tmm_ref/rank_related/rsdnet/saliency_maps_salsod_rsdnet_1/266434_1028.png"
    pytorch_path = "/temp_disk2/leise/ml/rsdnet-pytorch/pytorch_out.png"
    
    caffe_img = cv2.imread(caffe_path, 0).astype(np.int)
    pytorch_img = cv2.imread(pytorch_path, 0).astype(np.int)
    diff = (caffe_img - pytorch_img)
    print(np.max(diff), np.min(diff))
    diff = np.absolute(diff)

    plt.imshow(diff)
    plt.colorbar()
    # plt.show()
    plt.savefig('diff1.png')

if __name__ == "__main__":
    analysis_diff()