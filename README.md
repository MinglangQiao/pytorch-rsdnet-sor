# rsdnet-pytorch
A pytorch implementation of  rsdnet for the paper:  

[Revisiting Salient Object Detection: Simultaneous Detection, Ranking, and Subitizing of Multiple Salient Objects, Presented at CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Islam_Revisiting_Salient_Object_CVPR_2018_paper.pdf)

![image](https://github.com/MinglangQiao/pytorch-rsdnet-sor/blob/master/data/rsdnet_framework.PNG)

This  repository is based on the caffe version of [rsdnet](https://github.com/islamamirul/rsdnet) by [Islam](https://github.com/islamamirul). Since building of caffe is not easy, I built this repository for more convenient usage of rsdnet.


## Dependency
```
python 3.6 
pytorch 1.4
opencv-python
```

Install them by the [requirements.txt](https://github.com/MinglangQiao/rsdnet-pytorch/blob/master/requirements.txt)
```bash
pip install -r requirements.txt
``` 

## Model
Download the pytorch model from [Dropbox](https://www.dropbox.com/s/7du5mgo8a0k5rcn/weights.pkl?dl=0)(key: sorrank) or [BaiduPan](https://pan.baidu.com/s/1dPGJPp-g-m8yWHAc4hmDWA )(key: 652f), which is converted from the original [caffe model](https://www.dropbox.com/sh/we3vk0z9nln0jao/AABVOTQ2N9kcBN_gnN2rJ11Wa?dl=0) by scripts [caffe2pytorch.py](https://github.com/MinglangQiao/rsdnet-pytorch/blob/cd8ae1d98b66ea29ecf10f202f54a4f27641859d/scripts/caffe2pytorch.py#L55) and [utils.py](https://github.com/MinglangQiao/rsdnet-pytorch/blob/cd8ae1d98b66ea29ecf10f202f54a4f27641859d/utils.py#L69).

## Test
Put the model in proper dictionary and set the model, input and output path in model.py, then run
```py
python model.py
```

## Comparison
### Result comparison

| Model  | SOR | MAE | AUC | max-Fm | med-Fm | avg-Fm |
| :---  | :---:  | :---:  | :---:  | :---:  | :---:  |:---:  |
| rsdnet-caffe  | xx  | xx  |  xx  |  xx  |  xx  |   xx  | 
| rsdnet-pytorch  | xx  | xx  |  xx  |  xx  |  xx  |   xx  | 

### Inference time comparison
| Model  | Time (ms) | 
| :---  | :---:  |
| rsdnet-caffe  | xx  | 
| rsdnet-pytorch  | xx  |


### Map
Comparison with the caffe version. Currently the difference is introduced by different implementations of caffe Interp layer and
opencv bilinear interpolation. I will solve that soon.
![image](https://github.com/MinglangQiao/rsdnet-pytorch/blob/master/data/compare.jpg)


## Check list

- [x] Test code in pytorch
- [ ] Recover the original image size as input
- [ ] performance comparison of pytorch and caffe version  
- [ ] Training code in pytorch

## Reference
[1] [islamamirul/rsdnet](https://github.com/islamamirul/rsdnet) \
[2] [Caffe转Pytorch模型系列教程 概述](https://blog.csdn.net/DumpDoctorWang/article/details/88716962)

