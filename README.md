# HCFNet

HCFNet: HCFNet: A Hierarchical Calibration and Fusion Network for RGB-D Salient Object Detection

## Environments

```bash
conda create -n magnet python=3.10.8
conda activate hcfnet
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.14.1 cudatoolkit=11.6 -c pytorch -c conda-forge
conda install -c conda-forge opencv-python
pip install timm==0.6.5
conda install -c conda-forge tqdm
conda install yacs

```

## Data Preparation

* Download the RGB-D raw data from [Google drive](https://www.google.com/search?q=%23%E5%9C%A8%E8%BF%99%E9%87%8C%E6%9B%BF%E6%8D%A2%E4%B8%BA%E6%82%A8%E7%9A%84%E6%95%B0%E6%8D%AE%E9%9B%86%E9%93%BE%E6%8E%A5)

Note that the depth maps of the raw data above are foreground is white.

## Training & Testing

* Train the HCFNet:
i. download the pretrained Swin Transformer V2-B pth from [Google drive](https://www.google.com/search?q=%23%E5%9C%A8%E8%BF%99%E9%87%8C%E6%9B%BF%E6%8D%A2%E4%B8%BA%E6%82%A8%E7%9A%84%E9%A2%84%E8%AE%AD%E7%BB%83%E6%9D%83%E9%87%8D%E9%93%BE%E6%8E%A5).
ii. modify the `rgb_root` `depth_root` `gt_root` `pdepth_root` in `train.py` according to your own data path.
iii. run `python train.py`
* Test the HCFNet:
i. modify the `test_path` `pth_path` in `test.py` according to your own data path.
ii. run `python test_Net.py`

## Evaluate tools

* You can select one of toolboxes to get the metrics [CODToolbox](https://www.google.com/search?q=%23%E6%B5%8B%E8%AF%95%E5%B7%A5%E5%85%B7%E9%93%BE%E6%8E%A51) / [SOD_Evaluation_Metrics](https://www.google.com/search?q=%23%E6%B5%8B%E8%AF%95%E5%B7%A5%E5%85%B7%E9%93%BE%E6%8E%A52)

## Saliency Maps

We provide the saliency maps of DUT, LFSD, NJU2K, NLPR, SIP, STERE datasets.

* RGB-D [Google drive].([#在这里替换为您的显著性图下载链接])

## Trained Models

* RGB-D [Google drive].([#在这里替换为您的训练模型下载链接])

## Contact

If you have any questions about the code or the paper, please feel free to contact:

[您的名字/拼音]
Email: [您的邮箱地址]
