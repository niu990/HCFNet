import os
import cv2
import tqdm
import argparse
import warnings

warnings.filterwarnings('ignore')

os.environ["OPENCV_LOG_LEVEL"] = "FATAL" 

from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure

def eval_all(method_path, gt_root, test_sets):
    print("-" * 88)
    print(f"{'Dataset':<12} | {'Sm':<8} | {'Fmax':<8} | {'wFm':<8} | {'Emax':<8} | {'MAE':<8}")
    print("-" * 88)

    for dataset in test_sets:
        pred_dir = os.path.join(method_path, dataset)
        gt_dir = os.path.join(gt_root, dataset, 'GT')

        # 如果路径不存在，直接静默跳过，不打印警告
        if not os.path.exists(pred_dir) or not os.path.exists(gt_dir):
            continue

        S = Smeasure(); E = Emeasure(); F = Fmeasure(); W = WeightedFmeasure(); M = MAE()

        img_list = [f for f in os.listdir(gt_dir) if f.endswith(('.png', '.jpg'))]
        valid_count = 0

        for img_name in tqdm.tqdm(img_list, desc=f"Evaluating {dataset}", leave=False):
            gt_path = os.path.join(gt_dir, img_name)
            pred_path = os.path.join(pred_dir, img_name)
            
            # 如果原始后缀不存在，尝试替换为 .png
            if not os.path.exists(pred_path):
                pred_path = os.path.join(pred_dir, os.path.splitext(img_name)[0] + '.png')

            # 在交给 cv2.imread 之前，严格检查文件是否存在，从源头避免底层警告
            if not os.path.exists(pred_path):
                continue

            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

            if pred is None or gt is None: 
                continue
            
            if pred.shape != gt.shape:
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

            S.step(pred, gt); E.step(pred, gt); F.step(pred, gt); W.step(pred, gt); M.step(pred, gt)
            valid_count += 1

        # 如果找不到可评估的有效图片，直接静默跳过
        if valid_count == 0:
            continue

        sm = S.get_results()["sm"]
        f_max = F.get_results()["fm"]["curve"].max()
        wfm = W.get_results()["wfm"]
        e_max = E.get_results()["em"]["curve"].max()
        mae = M.get_results()["mae"]

        print(f"{dataset:<12} | {sm:.4f}   | {f_max:.4f}   | {wfm:.4f}   | {e_max:.4f}   | {mae:.4f}")

    print("-" * 88)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method_path', type=str, default='./test_maps/HCFNet_Final')
    parser.add_argument('--gt_root', type=str, default='./Dataset/val/')
    args = parser.parse_args()

    datasets = ['STEREO','NJU2K', 'NLPR', 'SIP', 'DUT-RGBD', 'LFSD', 'SSD', 'RGBD135']
    eval_all(args.method_path, args.gt_root, datasets)