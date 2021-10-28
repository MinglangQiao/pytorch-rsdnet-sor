# rsdnet-pytorch
A pytorch implementation of  rsdnet for the paper:  

[Revisiting Salient Object Detection: Simultaneous Detection, Ranking, and Subitizing of Multiple Salient Objects, Presented at CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Islam_Revisiting_Salient_Object_CVPR_2018_paper.pdf)


This  repository is based on the caffe version of [rsdnet](https://github.com/islamamirul/rsdnet) by [Islam](https://github.com/islamamirul). Since the environment building of caffe is not easy, I build this repository for more convenient usage of rsdnet.

## Dependency
```
python 3.6 
pytorch 1.4
opencv-python
```

Install them by the [requirements.txt]()
```bash
pip install -r requirements.txt
``` 

## Model
Download the pytorch model from [Dropbox](https://www.dropbox.com/s/7du5mgo8a0k5rcn/weights.pkl?dl=0)(key: sorrank) or [BaiduPan](https://pan.baidu.com/s/1dPGJPp-g-m8yWHAc4hmDWA )(key: 652f), which is converted from the original [caffe model]() by scripts [caffe2pytorch.py]() and [utils.py]().

## Test
Put the model in proper dictionary and set the model, input and output path in test.py, then run
```py
python model.py
```

## Result comparison
### Table

* results obtained by xx

### Map
Comparation with caffe version.
![image](https://github.com/MinglangQiao/rsdnet-pytorch/blob/master/data/compare.jpg)


## Check list

- [x] Test code in pytorch
- [ ] Recover the original image size as input
- [ ] performance comparison of pytorch and caffe version  
- [ ] Training code in pytorch

## Reference
[1] [islamamirul/rsdnet](https://github.com/islamamirul/rsdnet) \
[2] [Caffe转Pytorch模型系列教程 概述](https://blog.csdn.net/DumpDoctorWang/article/details/88716962)

