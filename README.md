# Introduction

본 프로젝트는 [RetinaFace](https://github.com/biubug6/Pytorch_Retinaface) 코드에 기초를 두고, 위 모델의 비효율적인 정보 흐름과 과도한 메모리 낭비 문제를 해결하는 데에 초점을 맞춥니다. 기존 RetinaFace는 경량 Backbone (MobileNet v1)을 사용 시 성능이 약 13% 급락하며, Multi-Head Loss를 사용하는 과정에서 정보의 효율성이 떨어져 Detection Head에 과도한 메모리가 할당되는 문제가 있습니다.

이러한 문제를 해결하기 위하여 저희는 다음 네 가지 구조를 개편합니다.
1) ECA-CBAM: 중요한 채널/공간 정보를 강조하여 Head 입력을 정제합니다. CBAM의 MLP를 1D Conv(ECA)로 대체해 **연산 부담을 줄이면서 표현력을 유지**합니다.

2) WFPN (Weighted Feature Pyramid Network): 기존 FPN의 단순 sum 대신 학습 가능한 가중치 결합을 통해, 스케일별 중요도를 학습하며 **정보 손실을 줄이고 정보 흐름**을 원활하게 합니다.

3) SCM (Shuffle Context Module): 채널 수를 줄여 연산을 억제한 뒤, Shuffle을 통해 Context를 섞어 **효율적으로 표현을 강화**합니다.

4) Deformable Convolution: offset 기반 소형/변형 얼굴에서도 안정적으로 특징을 포착해 **탐지 성능을 보완**합니다.

--- 

# RetinaFace vs ACWFace

## RetinaFace Structure

* Backbone ➞ FPN  ➞ Detection Head

![image](./assets/retinaface_structure.png)

## Our model Structure

* Backbone ➞ ECA-CBAM ➞ WFPN ➞ ECA-CBAM ➞ SCM ➞ ECA-CBAM ➞ SCM ➞ ECA-CBAM ➞ Detection Head

![image](./assets/our_model_structure.png)

---

# Result

## Ablation Study

![image](./assets/ablation_study.png)

* WiderFace Dataset의 Hard Case에서 약 2.7% AP 향상.
* Hard Case는 가려짐과 매우 소형 객체가 많으므로 이는 상당한 성능 개선으로 해석됨.

## Visualization

![image](./assets/visualization.png)

---

# Installation
## Clone and install
1. git clone https://github.com/kyeongha-git/Face_Detection_Project

2. Pytorch version 1.1.0+ and torchvision 0.3.0+ are needed.

3. Codes are based on Python 3

##### Data
1. Download Link: from [google cloud](https://drive.google.com/open?id=11UGV3nbVv1x9IC--_tK3Uxf7hA6rlbsS) Password: ruck

2. Organise the dataset directory as follows:

```Shell
  ./data/widerface/
    train/
      images/
      label.txt
    val/
      images/
      wider_val.txt
```
ps: wider_val.txt only include val file names but not label information.

## Training
We provide restnet50 and mobilenet0.25 as backbone network to train model.
We trained Mobilenet0.25 on imagenet dataset and get 46.58%  in top 1. If you do not wish to train the model, we also provide trained model. Pretrain model  and trained model are put in [google cloud](https://drive.google.com/open?id=1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1) Password: fstq . The model could be put as follows:
```Shell
  ./weights/
      mobilenet0.25_Final.pth
      mobilenetV1X0.25_pretrain.tar
      Resnet50_Final.pth
```
1. Before training, you can check network configuration (e.g. batch_size, min_sizes and steps etc..) in ``src/utils/config.py``.

2. Train the model using WIDER FACE:
  ```Shell
  CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train.py --network resnet50
  # or
  CUDA_VISIBLE_DEVICES=0 python tools/train.py --network mobile0.25
  ```


## Evaluation
### Evaluation widerface val
1. Generate txt file
```Shell
python tools/test_widerface.py --trained_model weight_file --network mobile0.25 or resnet50
```
2. Evaluate txt results. Demo come from [Here](https://github.com/wondervictor/WiderFace-Evaluation)
```Shell
cd ./widerface_evaluate
python setup.py build_ext --inplace
python evaluation.py
```
3. You can also use widerface official Matlab evaluate demo in [Here](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/WiderFace_Results.html)
### Evaluation FDDB

1. Download the images [FDDB](https://drive.google.com/open?id=17t4WULUDgZgiSy5kpCax4aooyPaz3GQH) to:
```Shell
./data/FDDB/images/
```

2. Evaluate the trained model using:
```Shell
python tools/test_fddb.py --trained_model weight_file --network mobile0.25 or resnet50
```


# Reference

* RetinaFace Paper: [https://arxiv.org/abs/1905.00641](https://arxiv.org/abs/1905.00641)
* RetinaFace Code: [https://github.com/biubug6/Pytorch\_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)
* FDLite Paper: [https://arxiv.org/abs/2406.19107](https://arxiv.org/abs/2406.19107)
* Light-Weight RetinaNet for Object Detection Paper: Private.
* ACWFace Paper: [https://www.spiedigitallibrary.org/journals/journal-of-electronic-imaging/volume-31/issue-1/013012/ACWFace-efficient-and-lightweight-face-detector-based-on-RetinaFace/10.1117/1.JEI.31.1.013012.short](https://www.spiedigitallibrary.org/journals/journal-of-electronic-imaging/volume-31/issue-1/013012/ACWFace-efficient-and-lightweight-face-detector-based-on-RetinaFace/10.1117/1.JEI.31.1.013012.short)
