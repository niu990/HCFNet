import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.res2net_v1b_base import Res2Net_model
from lib.Swin_V2 import SwinTransformerV2


# =========================================================
# Basic Blocks
# =========================================================

def safe_gn(c, max_groups=8):
    g = min(max_groups, c)
    while g > 1 and (c % g != 0):
        g -= 1
    return nn.GroupNorm(g, c)


class ConvGNAct(nn.Module):
    def __init__(self, c_in, c_out, k=1, s=1, p=None, groups=1, act=nn.GELU):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, groups=groups, bias=False)
        self.gn = safe_gn(c_out)
        self.act = act() if act is not None else nn.Identity()

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


class ResidualRefine(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block = nn.Sequential(
            ConvGNAct(c, c, 3, 1),
            ConvGNAct(c, c, 3, 1, act=None)
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class SpatialGate(nn.Module):
    def __init__(self, c):
        super().__init__()
        mid = max(c // 4, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(c, mid, 1, bias=False),
            safe_gn(mid),
            nn.GELU(),
            nn.Conv2d(mid, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)


# =========================================================
# Depth Calibration Module (DCM): stage1 / stage2
# =========================================================

class DCM(nn.Module):
    def __init__(self, channels, proj_ratio=4):
        super().__init__()

        proj_c = max(channels // proj_ratio, 8)

        self.proj_rgb = nn.Sequential(
            nn.Conv2d(channels, proj_c, 3, padding=1, bias=False),
            safe_gn(proj_c)
        )
        self.proj_dep = nn.Sequential(
            nn.Conv2d(channels, proj_c, 3, padding=1, bias=False),
            safe_gn(proj_c)
        )
        self.proj_pse = nn.Sequential(
            nn.Conv2d(1, proj_c, 3, padding=1, bias=False),
            safe_gn(proj_c)
        )

        self.cons_encoder = nn.Sequential(
            ConvGNAct(3, 16, 3, 1),
            ConvGNAct(16, 16, 3, 1)
        )
        
        rel_in = 16 + 2 

        self.rel_head = nn.Sequential(
            ConvGNAct(rel_in, 16, 3, 1),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

        self.dep_calib = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.rgb_calib = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)

        self.alpha = nn.Parameter(torch.tensor(0.2))
        self.beta = nn.Parameter(torch.tensor(0.05))

    def forward(self, f_rgb, f_dep, d_raw, d_pseudo=None):
        if d_pseudo is None:
            d_pseudo = d_raw

        _, _, H, W = f_rgb.shape
        d_raw_up = F.interpolate(d_raw, size=(H, W), mode='bilinear', align_corners=False)
        d_pse_up = F.interpolate(d_pseudo, size=(H, W), mode='bilinear', align_corners=False)

        z_rgb = self.proj_rgb(f_rgb)
        z_dep = self.proj_dep(f_dep)
        z_pse = self.proj_pse(d_pse_up)

        z_rgb_n = F.normalize(z_rgb, dim=1)
        z_dep_n = F.normalize(z_dep, dim=1)
        z_pse_n = F.normalize(z_pse, dim=1)

        mi_raw = torch.sum(z_rgb_n * z_dep_n, dim=1, keepdim=True)
        mi_pse = torch.sum(z_rgb_n * z_pse_n, dim=1, keepdim=True)

        d_diff = torch.abs(d_raw_up - d_pse_up)
        d_cons = self.cons_encoder(torch.cat([d_raw_up, d_pse_up, d_diff], dim=1))
        
        rel_inputs = [mi_raw, mi_pse, d_cons]

        r_map = self.rel_head(torch.cat(rel_inputs, dim=1))

        f_dep_c = f_dep + self.alpha * r_map * self.dep_calib(f_dep)
        f_rgb_c = f_rgb + self.beta * r_map * self.rgb_calib(f_rgb)

        aux = {
            "z_rgb": z_rgb,
            "z_dep": z_dep,
            "r_map": r_map
        }
        return f_rgb_c, f_dep_c, r_map, aux


class AlignmentLoss(nn.Module):
    def __init__(self, lambda_mi=0.1, tau=0.15, shift_small=2, shift_large=8):
        super().__init__()
        self.lambda_mi = lambda_mi
        self.tau = tau
        self.shift_small = shift_small
        self.shift_large = shift_large

    def _local_info_nce_loss(self, z_rgb, z_dep):
        z_rgb_n = F.normalize(z_rgb, dim=1)
        z_dep_n = F.normalize(z_dep, dim=1)

        pos_score = torch.sum(z_rgb_n * z_dep_n, dim=1)

        z_dep_neg_small = torch.roll(
            z_dep_n,
            shifts=(self.shift_small, self.shift_small),
            dims=(2, 3)
        )
        neg_score_small = torch.sum(z_rgb_n * z_dep_neg_small, dim=1)

        z_dep_neg_large = torch.roll(
            z_dep_n,
            shifts=(self.shift_large, self.shift_large),
            dims=(2, 3)
        )
        neg_score_large = torch.sum(z_rgb_n * z_dep_neg_large, dim=1)

        logits = torch.stack(
            [pos_score, neg_score_small, neg_score_large], dim=-1
        ) / self.tau

        labels = torch.zeros(
            logits.shape[:-1], dtype=torch.long, device=z_rgb.device
        )

        return F.cross_entropy(logits.reshape(-1, 3), labels.reshape(-1))

    def forward(self, mi_info):
        if mi_info is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            return torch.tensor(0.0, device=device)

        z_rgb = mi_info.get("z_rgb", None)
        z_dep = mi_info.get("z_dep", None)

        if z_rgb is None or z_dep is None:
            device = z_rgb.device if z_rgb is not None else (
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
            return torch.tensor(0.0, device=device)

        return self.lambda_mi * self._local_info_nce_loss(z_rgb, z_dep)


# =========================================================
# Conflict-Aware Fusion Module (CAFM): stage1 / stage2
# =========================================================

class CAFM(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.rgb_proj = ConvGNAct(channels, channels, 1, 1)
        self.dep_proj = ConvGNAct(channels, channels, 1, 1)
        self.rel_proj = ConvGNAct(1, channels, 3, 1)

        self.rgb_sa = SpatialGate(channels)
        self.dep_sa = SpatialGate(channels)

        in_c = channels * 4

        self.local_fuse = nn.Sequential(
            ConvGNAct(in_c, channels, 1, 1),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            safe_gn(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            safe_gn(channels),
            nn.GELU()
        )

        self.gamma = nn.Parameter(torch.tensor(0.05))
        self.dep_gain = nn.Parameter(torch.tensor(0.2))

    def forward(self, f_rgb, f_dep, r_map):
        rgb = self.rgb_proj(f_rgb)
        dep = self.dep_proj(f_dep)

        if r_map.shape[2:] != dep.shape[2:]:
            r_map = F.interpolate(r_map, size=dep.shape[2:], mode='bilinear', align_corners=False)

        dep = dep + self.dep_gain * self.rel_proj(r_map) * dep

        rgb = rgb + self.rgb_sa(rgb) * rgb
        dep = dep + self.dep_sa(dep) * dep

        cons = rgb * dep
        diff = torch.abs(rgb - dep) * (1.0 - r_map)
        
        feats = [rgb, dep, cons, diff]

        fusion = self.local_fuse(torch.cat(feats, dim=1))
        out = f_rgb + self.gamma * fusion
        return out


# =========================================================
# High-level Semantic Fusion (HSF): stage3 / stage4
# =========================================================

class HSF(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        self.fuse = nn.Sequential(
            ConvGNAct(channels * 2, channels, 1, 1),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            safe_gn(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            safe_gn(channels),
            nn.GELU()
        )
            
        self.refine = ResidualRefine(channels)

    def forward(self, f_rgb, f_dep):
        if f_rgb.shape[2:] != f_dep.shape[2:]:
            f_dep = F.interpolate(f_dep, size=f_rgb.shape[2:], mode='bilinear', align_corners=False)
        out = self.fuse(torch.cat([f_rgb, f_dep], dim=1))
        out = self.refine(out)
        return out


# =========================================================
# Progressive Residual Decoder (PRD)
# =========================================================

class PRDStage(nn.Module):
    def __init__(self, c, use_rel_guidance=True):
        super().__init__()
        self.use_rel_guidance = use_rel_guidance

        self.high_proj = ConvGNAct(c, c, 1, 1)
        self.low_proj = ConvGNAct(c, c, 1, 1)

        if self.use_rel_guidance:
            self.rel_gate = nn.Sequential(
                ConvGNAct(1, c, 3, 1),
                nn.Conv2d(c, c, 1, bias=False),
                nn.Sigmoid()
            )

        self.fuse = ConvGNAct(c * 2, c, 3, 1)

        self.refine = nn.Sequential(
            ResidualRefine(c),
            ResidualRefine(c)
        )

        self.eta = nn.Parameter(torch.tensor(0.2))

    def forward(self, high, low, rel=None):
        high = F.interpolate(high, size=low.shape[2:], mode='bilinear', align_corners=False)
        high = self.high_proj(high)
        low = self.low_proj(low)

        if self.use_rel_guidance and rel is not None:
            rel = F.interpolate(rel, size=low.shape[2:], mode='bilinear', align_corners=False)
            low = low + self.eta * self.rel_gate(rel) * low

        out = self.fuse(torch.cat([high, low], dim=1))
        out = self.refine(out)

        return out


class PRD(nn.Module):
    def __init__(self, in_channels, dec_dim=48, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision

        self.proj = nn.ModuleList([ConvGNAct(c, dec_dim, 1, 1) for c in in_channels])

     
        self.stage3 = PRDStage(dec_dim, use_rel_guidance=False) 
        self.stage2 = PRDStage(dec_dim, use_rel_guidance=True)
        self.stage1 = PRDStage(dec_dim, use_rel_guidance=True)

        self.final_refine = ResidualRefine(dec_dim)

        self.pred4 = nn.Conv2d(dec_dim, 1, 1)
        self.pred3 = nn.Conv2d(dec_dim, 1, 1)
        self.pred2 = nn.Conv2d(dec_dim, 1, 1)
        self.pred1 = nn.Conv2d(dec_dim, 1, 1)

    def forward(self, feats, rels=None, image_size=None):
        f1, f2, f3, f4 = [proj(f) for proj, f in zip(self.proj, feats)]

        r1 = rels[0] if rels is not None and len(rels) > 0 else None
        r2 = rels[1] if rels is not None and len(rels) > 1 else None
        
        d3 = self.stage3(f4, f3, rel=None)
        d2 = self.stage2(d3, f2, rel=r2)
        d1 = self.stage1(d2, f1, rel=r1)
        d1 = self.final_refine(d1)

        p1 = self.pred1(d1)
        p2 = self.pred2(d2)
        p3 = self.pred3(d3)
        p4 = self.pred4(f4)

        if image_size is not None:
            p1 = F.interpolate(p1, size=image_size, mode='bilinear', align_corners=False)
            p2 = F.interpolate(p2, size=image_size, mode='bilinear', align_corners=False)
            p3 = F.interpolate(p3, size=image_size, mode='bilinear', align_corners=False)
            p4 = F.interpolate(p4, size=image_size, mode='bilinear', align_corners=False)

        res = {
            "sal": p1,
            "aux2": p2,
            "aux3": p3,
            "aux4": p4
        }
        return res if self.deep_supervision else {"sal": p1}


# =========================================================
# Full Model HCFNet
# =========================================================

class HCFNet(nn.Module):
    def __init__(self,
                 channel=48,
                 ind=50,
                 deep_supervision=True,
                 swin_pretrained_path="./pre/swinv2_base_patch4_window16_256.pth"):
        super().__init__()
        self.deep_supervision = deep_supervision

        self.rgb_backbone = SwinTransformerV2()
        self.depth_backbone = Res2Net_model(ind)
        self.layer_dep0 = nn.Conv2d(1, 3, kernel_size=1)

        self.dep_p1 = nn.Conv2d(256, 128, 1)
        self.dep_p2 = nn.Conv2d(512, 256, 1)
        self.dep_p3 = nn.Conv2d(1024, 512, 1)
        self.dep_p4 = nn.Conv2d(2048, 1024, 1)

        # Depth Calibration Modules for stage1 / stage2
        self.dcm1 = DCM(128)
        self.dcm2 = DCM(256)

        # Conflict-Aware Fusion Modules for stage1 / stage2
        self.cafm1 = CAFM(128)
        self.cafm2 = CAFM(256)

        # High-level Semantic Fusion for stage3 / stage4
        self.hsf3 = HSF(512)
        self.hsf4 = HSF(1024)

        # Progressive Residual Decoder
        self.decoder = PRD(
            in_channels=[128, 256, 512, 1024],
            dec_dim=channel,
            deep_supervision=deep_supervision
        )

        if swin_pretrained_path:
            self._load_swin_pretrained(swin_pretrained_path)

    def _load_swin_pretrained(self, path):
        try:
            ckpt = torch.load(path, map_location="cpu")
            state = ckpt["model"] if "model" in ckpt else ckpt
            self.rgb_backbone.load_state_dict(state, strict=False)
            print(f"==> Loaded Swin pretrained from: {path}")
        except Exception as e:
            print(f"==> Warning: failed to load Swin pretrained from {path}, reason: {e}")

    def forward(self, imgs, depths, pseudo_depth=None, return_neck=False):
        if pseudo_depth is None:
            pseudo_depth = depths

        stage_rgb = self.rgb_backbone(imgs)
        img_feats = [stage_rgb[0], stage_rgb[1], stage_rgb[2], stage_rgb[4]]

        _, d1, d2, d3, d4 = self.depth_backbone(self.layer_dep0(depths))
        dep_feats = [self.dep_p1(d1), self.dep_p2(d2), self.dep_p3(d3), self.dep_p4(d4)]

        # Shallow calibration: stage1 / stage2
        rgb1_c, dep1_c, r1, aux1 = self.dcm1(img_feats[0], dep_feats[0], depths, pseudo_depth)
        rgb2_c, dep2_c, r2, aux2 = self.dcm2(img_feats[1], dep_feats[1], depths, pseudo_depth)

        img_feats[0], dep_feats[0] = rgb1_c, dep1_c
        img_feats[1], dep_feats[1] = rgb2_c, dep2_c

        rels = [r1, r2]

        # Stage1 / Stage2 fusion
        f1 = self.cafm1(img_feats[0], dep_feats[0], r1)
        f2 = self.cafm2(img_feats[1], dep_feats[1], r2)

        # Stage3 / Stage4 high-level fusion
        f3 = self.hsf3(img_feats[2], dep_feats[2])
        f4 = self.hsf4(img_feats[3], dep_feats[3])

        fused = [f1, f2, f3, f4]

        # Decoding
        dec_out = self.decoder(fused, rels=rels, image_size=imgs.shape[2:])
        final = dec_out["sal"]

        if return_neck:
            neck_info = {
                "dcm_info_1": aux1,
                "dcm_info_2": aux2
            }
            return final, (dec_out if self.deep_supervision else None), neck_info

        return (final, dec_out) if self.deep_supervision else final