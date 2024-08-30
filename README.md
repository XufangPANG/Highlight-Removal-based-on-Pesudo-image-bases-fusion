# Highlight-Removal-based-on-Pesudo-image-bases-fusion

## Introduction
> [Single Image Highlight Removal via Innovative Pseudo Image Bases Fusion with a Dual-Network]\
 Xufang PANG, Xiansheng CHEN, Chao YANG, Zhenliang ZHENG, Shengbo LIU, Hongjun ZHOU, Bo JIANG, Ning DING, Xiaoping ZHONG
>
> Our approach is inspired by the observation that specular highlights increase brightness and decrease saturation. We generate pseudo-SV (saturation-value) modulated image bases, effectively creating a discrete color space that approximates the brightness, saturation, and hues of highlight-free pixels. Utilizing a dual-network architecture, we jointly train a highlight detection sub-network and a highlight removal sub-network. By leveraging generated image bases and estimated highlight positional priors, our method learns texture nuances across highlight levels, producing high-quality highlight-free images via a weighted fusion process.

Experimental results show that our method significantly restores texture and color details, outperforming existing methods in PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index) scores.

<p align="left">
  <img width=720" src="fig\framework.png">
</p>

<p align="left">
  <img width=720" src="fig\image_bases.png">
</p>


## Prerequisites
```
conda env create -f environment.yaml
```
## Dataset
```
|--ShiQ
    |--train
        |--train_A
            |--00001.png
        |--train_B
            |--00001.png
        |--train_C
            |--00001.png
    |--test
        |--train_A
        |--train_B
        |--train_C
```

## train and test
Modify the path to your dataset by entering it into the `opt.dataroot` parameter in `joint_train.py` and `test.py`.
```
python joint_train.py
python test.py
```

## Highlight Removal Results
| Methods\Metrics  | PSNR↑ | SSIM↑ | 
| :----: | :-----: | :------: |
|Without highlight detection|35.18|	0.97|
|Input Ground Truth masks|	36.17|	0.99|
|Separately trained|        34.91|	0.98|
|With pixel-wise fusion|	35.81|	0.98|
|Ours|                      35.85|	0.99|

## visual 

<p align="left">
  <img width=720" src="fig\visual.png">
</p>
