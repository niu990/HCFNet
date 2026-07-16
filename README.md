# HCFNet

HCFNet: HCFNet: A Hierarchical Calibration and Fusion Network for RGB-D Salient Object Detection

## Environments

```bash
conda create -n hcfnet python=3.10.8
conda activate hcfnet
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.14.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install opencv-python-headless
pip install timm==0.6.5
pip install tqdm timm
pip install "numpy<2"
pip install tensorboardX
```

## Data Preparation

* Download the RGB-D raw data from [Google drive](https://drive.google.com/drive/folders/1tZydny9By986G0NtUTd6k7V36Fn78wcI?usp=drive_link)

Note that the depth maps of the raw data above are foreground is white.

## Training & Testing

* Train the HCFNet:  
i. download the pretrained Swin Transformer V2-B pth from [Google drive](https://drive.google.com/file/d/1IqEOQmVaft0J3SxUPcjFuCS3nPSRu2mL/view?usp=drive_link) and place it into the `./pre/` folder.  
ii. modify the `rgb_root` `depth_root` `gt_root` `pdepth_root` in `train.py` according to your own data path.  
iii. run `python train.py`  
* Test the HCFNet:
    Run the following command to generate saliency maps. Please configure your own dataset and weights paths:
    ```bash
    python test.py --test_path ./Dataset/val/ --pth_path ./Checkpoint/HCFNet_Final/HCFNet_best.pth
    ```

## Evaluation

We use the [PySODMetrics](https://github.com/lartpang/PySODMetrics) library for quantitative evaluation. 

i. Install the required evaluation metric library:
```bash
pip install pysodmetrics
```
ii. Run the evaluation script:

```bash
python eval.py
```

## Saliency Maps
We provide the saliency maps of DUT-RGBD, LFSD, NJU2K, NLPR, SSD, DES datasets.

RGB-D [Google drive](https://drive.google.com/drive/folders/1OB-lFteJw3mCMcgEL--rdQar-KOGFgYv?usp=drive_link)

## Trained Models
RGB-D [Google drive](https://drive.google.com/file/d/1-h1vz1DwIKbTqZ-BGt3jQQjs181uU3Kw/view?usp=drive_link)

## Contact
If you have any questions about the code or the paper, please feel free to contact:

Hanqing Niu  
Email: niuhanqing@stu.hebtu.edu.cn
