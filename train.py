import os
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from tensorboardX import SummaryWriter
import argparse

from lib.model_HCFNet import HCFNet, AlignmentLoss
from utils.data import get_loader, test_dataset
from utils.utils import adjust_lr


parser = argparse.ArgumentParser(description='HCFNet Final Training')

parser.add_argument('--epoch', type=int, default=150)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--batchsize', type=int, default=6)
parser.add_argument('--accumulation_steps', type=int, default=1)
parser.add_argument('--trainsize', type=int, default=256)
parser.add_argument('--clip', type=float, default=0.5)
parser.add_argument('--decay_rate', type=float, default=0.1)
parser.add_argument('--decay_epoch', type=int, default=60)
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--exp_name', type=str, default='HCFNet_Final')

parser.add_argument('--lambda_mi', type=float, default=0.1,
                    help='MI loss global weight (default: 0.1, set 0 to disable)')
parser.add_argument('--mi_tau', type=float, default=0.15,
                    help='MI loss temperature (default: 0.15)')
parser.add_argument('--mi_stage2_weight', type=float, default=0.5,
                    help='Stage2 MI loss relative weight (default: 0.5)')
parser.add_argument('--shift_small', type=int, default=2,
                    help='small shift for MI loss negative sample')
parser.add_argument('--shift_large', type=int, default=8,
                    help='large shift for MI loss negative sample')

# Data path
parser.add_argument('--rgb_root', type=str, default='./Dataset/train/RGB/')
parser.add_argument('--gt_root', type=str, default='./Dataset/train/GT/')
parser.add_argument('--depth_root', type=str, default='./Dataset/train/depth/')
parser.add_argument('--pdepth_root', type=str, default='./Dataset/train/Depth_pseudo/')

opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / (weit.sum(dim=(2, 3)) + 1e-8)

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1 + 1e-8)
    return (wbce + wiou).mean()


def train(train_loader, model, mi_loss_fn, optimizer, scaler, epoch, total_step, writer):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_record = 0.0

    for i, (images, gts, depths, pdepths) in enumerate(train_loader, start=1):
        images = images.cuda(non_blocking=True)
        gts = gts.cuda(non_blocking=True)
        depths = depths.cuda(non_blocking=True)
        pdepths = pdepths.cuda(non_blocking=True)

        with torch.amp.autocast('cuda'):
            pred, sides, neck_info = model(images, depths, pseudo_depth=pdepths, return_neck=True)

            loss_main = structure_loss(pred, gts)

            loss_side = torch.tensor(0.0, device=images.device)
            if sides is not None:
                for k in ['aux2', 'aux3', 'aux4']:
                    if k in sides and sides[k] is not None:
                        loss_side = loss_side + structure_loss(sides[k], gts) * 0.5

            loss_mi = torch.tensor(0.0, device=images.device)
            if opt.lambda_mi > 0:
                mi1 = mi_loss_fn(neck_info.get("dcm_info_1", None))
                mi2 = mi_loss_fn(neck_info.get("dcm_info_2", None))
                loss_mi = mi1 + opt.mi_stage2_weight * mi2

            total_loss = (loss_main + loss_side + loss_mi) / opt.accumulation_steps

        scaler.scale(total_loss).backward()

        if i % opt.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        loss_record += total_loss.item()

        if i % 20 == 0 or i == total_step:
            print(f'{datetime.now()} [Epoch {epoch:03d}] [Step {i:04d}/{total_step:04d}] '
                  f'Loss: {total_loss.item() * opt.accumulation_steps:.4f} | '
                  f'Main: {loss_main.item():.4f} | Side: {loss_side.item():.4f} | '
                  f'MI: {loss_mi.item():.4f}')

    epoch_loss = loss_record / total_step
    writer.add_scalar('Train/TotalLoss', epoch_loss, epoch)
    return epoch_loss


def validate(model, epoch, writer, save_path):
    test_loader = test_dataset(
        './Dataset/val/NJU2K/RGB/',
        './Dataset/val/NJU2K/GT/',
        './Dataset/val/NJU2K/depth/',
        './Dataset/val/NJU2K/Depth_pseudo/',
        opt.trainsize
    )

    model.eval()
    mae_sum = 0.0

    with torch.no_grad():
        for _ in range(test_loader.size):
            image, gt, depth, pdepth, _, _ = test_loader.load_data()
            gt_np = np.asarray(gt, np.float32)
            gt_np /= (gt_np.max() + 1e-8)

            res = model(image.cuda(), depth.cuda(), pseudo_depth=pdepth.cuda(), return_neck=False)
            if isinstance(res, (tuple, list)):
                res = res[0]

            res = F.interpolate(res, size=gt_np.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.mean(np.abs(res - gt_np))

    mae = mae_sum / test_loader.size
    writer.add_scalar('Val/MAE', mae, epoch)
    print(f'==> Epoch {epoch:03d} | Val MAE: {mae:.4f}')

    global best_mae, best_epoch
    if epoch == 1 or mae < best_mae:
        best_mae = mae
        best_epoch = epoch
        torch.save(model.state_dict(), os.path.join(save_path, 'HCFNet_best.pth'))
        print(f'>>> Saved Best Model at Epoch {best_epoch} | MAE: {best_mae:.4f}')


if __name__ == '__main__':
    save_path = f'./Checkpoint/{opt.exp_name}/'
    os.makedirs(save_path, exist_ok=True)
    writer = SummaryWriter(os.path.join(save_path, 'summary'))

    print(f"==> MI Loss Config: lambda_mi={opt.lambda_mi}, tau={opt.mi_tau}, "
      f"stage2_weight={opt.mi_stage2_weight}, "
      f"shift_small={opt.shift_small}, shift_large={opt.shift_large}")

    model = HCFNet(
        channel=48,
        deep_supervision=True,
        swin_pretrained_path="./pre/swinv2_base_patch4_window16_256.pth"
    ).cuda()

    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    scaler = torch.cuda.amp.GradScaler()

    mi_loss_fn = AlignmentLoss(
        lambda_mi=opt.lambda_mi,
        tau=opt.mi_tau,
        shift_small=opt.shift_small,
        shift_large=opt.shift_large
    ).cuda()

    train_loader = get_loader(
        opt.rgb_root, opt.gt_root, opt.depth_root, opt.pdepth_root,
        batchsize=opt.batchsize, trainsize=opt.trainsize
    )

    print(f"==> Start Training | Exp: {opt.exp_name}")
    best_mae = 1.0
    best_epoch = 0

    for epoch in range(1, opt.epoch + 1):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, mi_loss_fn, optimizer, scaler, epoch, len(train_loader), writer)
        validate(model, epoch, writer, save_path)