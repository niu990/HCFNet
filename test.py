import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2

from lib.model_HCFNet import HCFNet
from utils.data import test_dataset


parser = argparse.ArgumentParser(description='HCFNet Test')

parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--testsize', type=int, default=256)
parser.add_argument('--load', type=str, required=True)
parser.add_argument('--save_path', type=str, default='./test_maps/')


opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id


def run_inference():
    model = HCFNet(
        channel=48,
        deep_supervision=True,
        swin_pretrained_path="./pre/swinv2_base_patch4_window16_256.pth"
    ).cuda()

    print(f"==> Loading weights from {opt.load}")
    state_dict = torch.load(opt.load, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    exp_name = os.path.basename(os.path.dirname(opt.load))
    save_root = os.path.join(opt.save_path, exp_name)
    os.makedirs(save_root, exist_ok=True)

    test_datasets = [
        'NJU2K',
        'NLPR',
        'DUT-RGBD',
        'LFSD',
        'SSD',
        'DES'
    ]

    for dataset in test_datasets:
        image_root = f'./Dataset/val/{dataset}/RGB/'
        gt_root = f'./Dataset/val/{dataset}/GT/'
        depth_root = f'./Dataset/val/{dataset}/depth/'
        pdepth_root = f'./Dataset/val/{dataset}/Depth_pseudo/'

        if not os.path.exists(image_root):
            print(f'[Skip] {dataset} not found: {image_root}')
            continue

        save_path = os.path.join(save_root, dataset)
        os.makedirs(save_path, exist_ok=True)

        test_loader = test_dataset(
            image_root,
            gt_root,
            depth_root,
            pdepth_root,
            opt.testsize
        )

        print(f'---> Dataset: {dataset} | Total: {test_loader.size}')

        for _ in range(test_loader.size):
            image, gt, depth, pdepth, name, _ = test_loader.load_data()

            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)

            image = image.cuda(non_blocking=True)
            depth = depth.cuda(non_blocking=True)
            pdepth = pdepth.cuda(non_blocking=True)

            with torch.no_grad():
                res = model(image, depth, pseudo_depth=pdepth, return_neck=False)
                if isinstance(res, (tuple, list)):
                    out = res[0]
                else:
                    out = res

            out = F.interpolate(out, size=gt.shape, mode='bilinear', align_corners=False)
            out = torch.sigmoid(out).data.cpu().numpy().squeeze()
            out = (out - out.min()) / (out.max() - out.min() + 1e-8)

            if name.endswith('.jpg'):
                name = name.replace('.jpg', '.png')
            elif not name.endswith('.png'):
                name = os.path.splitext(name)[0] + '.png'

            cv2.imwrite(os.path.join(save_path, name), out * 255)

    print(f'==> Test Done! Results saved to: {save_root}')


if __name__ == '__main__':
    run_inference()