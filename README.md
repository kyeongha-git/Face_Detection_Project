# Abstract

This project optimizes the [RetinaFace](https://github.com/biubug6/Pytorch_Retinaface) model for lightweight deployment and enhanced detection performance. By refactoring and decoupling core modules with a MobileNet 0.25 backbone, our proposed model **reduces FLOPs by ~2.6%** while **improving AP by 3.07%** on the WiderFace Hard validation set compared to the baseline.

### Customized Modules
* **ECA-CBAM:** Integrates ECA into the CBAM structure by replacing the MLP in the Channel Attention module with a 1D Conv. It improves AP by **~0.38%** on WiderFace Hard with only a marginal **~0.16% increase in FLOPs** compared to standard ECA.
* **WFPN (Weighted Feature Pyramid Network):** Replaces simple summation in FPN with learnable weighted combinations to prevent information loss. Integrated with Deformable Convolution Networks (DCN) to better detect distorted objects.
* **SCM (Shuffle Context Module):** Reduces channels to suppress computational cost and uses channel shuffling to efficiently enhance feature representations. Also integrated with DCN.
* **Deformable Convolution:** Captures features robustly using offsets, significantly compensating for detection performance on small and deformed faces.

--- 

# RetinaFace vs Our Model

## RetinaFace Structure
* Backbone ➞ FPN ➞ Detection Head

![image](./assets/retinaface_structure.png)

## Our Model Structure
* Backbone ➞ ECA-CBAM ➞ WFPN ➞ ECA-CBAM ➞ SCM ➞ ECA-CBAM ➞ SCM ➞ ECA-CBAM ➞ Detection Head

![image](./assets/our_model_structure.png)

---

# Results

## Ablation Study

![image](./assets/ablation_study.png)

* **FLOPs reduced by 2.6%** compared to the baseline RetinaFace (based on 640 x 640 image input).
* **AP improved by 1.19% (Easy), 1.59% (Medium), and 3.07% (Hard)** on the WiderFace dataset.

## Attention Ablation Study

![image](./assets/attention_ablation_study.png)

* ECA-CBAM outperforms ECA and CBAM by **0.38% and 0.39% AP**, respectively, on WiderFace Hard.
* While ECA is sufficient for relatively easy cases, the spatial attention in ECA-CBAM proves highly effective for detecting small or occluded faces.
* Achieves performance gains while maintaining FLOPs and parameter counts nearly identical to ECA.

## Visualization

![image](./assets/visualization.png)

---

# Quick Start with Docker (Recommended)

The easiest way to run evaluation is via the pre-built Docker image. No local Python environment setup is required.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed
- WiderFace validation dataset (see [Data](#data) section below)
- *(Optional, for GPU)* [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed

## 1. Pull the Docker Image

```shell
docker pull kyeonghah/our-model-eval:latest
```

## 2. Prepare the Dataset

Download the WiderFace validation images from [Google Drive](https://drive.google.com/open?id=11UGV3nbVv1x9IC--_tK3Uxf7hA6rlbsS) (Password: `ruck`) and organize as follows:

```
/your/local/path/widerface/val/
  images/
    0--Parade/
    1--Handshaking/
    ...
  wider_val.txt
```

> `wider_val.txt` lists val image filenames but does not contain label information.

## 3. Run Evaluation

**GPU mode** (recommended):
```shell
docker run --rm --gpus all \
  -v /your/local/path/widerface/val:/app/data/widerface/val:ro \
  kyeonghah/our-model-eval:latest evaluate
```

**CPU mode** (no GPU required):
```shell
docker run --rm \
  -e USE_CPU=1 \
  -v /your/local/path/widerface/val:/app/data/widerface/val:ro \
  kyeonghah/our-model-eval:latest evaluate
```

The container runs a two-step pipeline automatically:
1. **Inference** — runs `test_widerface.py` over all validation images and saves per-image txt results
2. **AP Evaluation** — runs `evaluation.py` to compute Easy / Medium / Hard AP scores

Expected output:
```
==================== Results ====================
Easy   Val AP: 0.9039
Medium Val AP: 0.8810
Hard   Val AP: 0.7808
=================================================
```

## Other Commands

```shell
# Open an interactive shell inside the container
docker run --rm -it \
  -v /your/local/path/widerface/val:/app/data/widerface/val:ro \
  kyeonghah/our-model-eval:latest shell
```

---

# Local Installation

If you prefer to run without Docker, follow the steps below.

## Clone and Install

```shell
git clone https://github.com/kyeongha-git/Face_Detection_Project
cd Face_Detection_Project/our_model
pip install -r requirements.txt
```

> Requires Python 3, PyTorch 2.x, and torchvision.

## Data

<a name="data"></a>

Download the WiderFace dataset from [Google Drive](https://drive.google.com/open?id=11UGV3nbVv1x9IC--_tK3Uxf7hA6rlbsS) (Password: `ruck`) and organize as follows:

```shell
./data/widerface/
  train/
    images/
    label.txt
  val/
    images/
    wider_val.txt
```

> `wider_val.txt` lists val image filenames only and does not contain label information.

## Training

Pretrained MobileNetV1X0.25 (ImageNet top-1: 46.58%) and fully trained models are available on [Google Drive](https://drive.google.com/drive/folders/12iCRdAreBJeNPZSqD0C-dXfNZ5R4m6RD?usp=sharing). Place weights as follows:

```shell
./weights/
    mobilenet0.25_Final.pth
    mobilenetV1X0.25_pretrain.tar
    Resnet50_Final.pth
```

1. Configure training settings in `src/utils/config.py` (e.g., `batch_size`, `min_sizes`, `steps`).

2. Train the model on WiderFace:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train.py --network mobile0.25
# or single GPU
CUDA_VISIBLE_DEVICES=0 python tools/train.py --network mobile0.25
```

## Evaluation

### WiderFace Validation

**Step 1** — Generate per-image result txt files:
```shell
python tools/test_widerface.py \
  --trained_model ./weights/mobilenet0.25_Final.pth \
  --network mobile0.25
```

**Step 2** — Build the Cython bbox module and compute AP:
```shell
cd ./widerface_evaluate
python setup.py build_ext --inplace
python evaluation.py
```

You can also use the official WiderFace Matlab evaluation tool: [WiderFace Results](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/WiderFace_Results.html)

---

# Reference

* RetinaFace Paper: [https://arxiv.org/abs/1905.00641](https://arxiv.org/abs/1905.00641)
* RetinaFace Code: [https://github.com/biubug6/Pytorch\_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)
* FDLite Paper: [https://arxiv.org/abs/2406.19107](https://arxiv.org/abs/2406.19107)
* Light-Weight RetinaNet for Object Detection Paper: [https://arxiv.org/abs/1905.10011](https://arxiv.org/abs/1905.10011)
* ACWFace Paper: [https://www.spiedigitallibrary.org/journals/journal-of-electronic-imaging/volume-31/issue-1/013012/ACWFace-efficient-and-lightweight-face-detector-based-on-RetinaFace/10.1117/1.JEI.31.1.013012.short](https://www.spiedigitallibrary.org/journals/journal-of-electronic-imaging/volume-31/issue-1/013012/ACWFace-efficient-and-lightweight-face-detector-based-on-RetinaFace/10.1117/1.JEI.31.1.013012.short)
