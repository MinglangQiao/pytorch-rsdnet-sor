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
python test_net.py
```

## Comparison
### [Quantitative result]()

<!-- | Model  | SOR &#8593; | MAE &#8595; | AUC &#8593; | max-Fm &#8593;| med-Fm &#8593; | avg-Fm &#8593;|
| :---  | :---:  | :---:  | :---:  | :---:  | :---:  |:---:  |
| rsdnet-caffe  | 0.8250  | 0.0910  |  xx  |  xx  |  xx  |   xx  | 
| rsdnet-pytorch  | 0.8280  | 0.0910 |  xx  |  xx  |  xx  |   xx  | -->

| Model  | SOR &#8593; | MAE &#8595; | Inference time (s) &#8595; | 
| :---  | :---:  | :---:  | :---:  |
| rsdnet-caffe  | 0.8250  | 0.0910  |  0.063  |
| rsdnet-pytorch  | 0.8280  | 0.0910 |  0.302  | 

* Results are evaluated on a V100 GPU for all 425 test images of PASCAL-S. As can be seen, the pytorch implementation has slightly higher SOR and same MAE, through slower inference speed.

* The difference is introduced by different implementations of caffe Interp layer of DeepLab and
Pytorch bilinear interpolation. I tried Opencv, PIL and Pytorch bilinear method, and find that the pytorch version is the best one.


### [Saliency map]()
Comparison with the caffe version.
![image](https://github.com/MinglangQiao/rsdnet-pytorch/blob/master/large_file/compare.jpg)


## Check list

- [x] Test code in pytorch
- [x] Recover the original image size as input
- [x] performance comparison of pytorch and caffe version  
- [ ] Training code in pytorch

## Results
The results of the pytorch-rsdnet on PASCAL-S could be download from \[[Baidu pan, key:fnpr](https://pan.baidu.com/s/109wVcp3yF4BKqgtynwDfOg)\].


## Reference
[1] [islamamirul/rsdnet](https://github.com/islamamirul/rsdnet) \
[2] [Caffe转Pytorch模型系列教程 概述](https://blog.csdn.net/DumpDoctorWang/article/details/88716962) \
[3] [kazuto1011/deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch)

