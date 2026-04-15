import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, List
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

from model import MultimodalReverseDistillation

class BaseEvaluator:
    def __init__(self, config=None, device=None):
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
    def save_model(self, path):
        raise NotImplementedError
        
    def load_model(self, path):
        raise NotImplementedError
        
    def build_feature_bank(self, train_loader):
        raise NotImplementedError
        
    def compute_anomaly_scores(self, test_loader):
        raise NotImplementedError
        
    def visualize_results(self, test_loader, output_dir, num_samples=5):
        raise NotImplementedError

class ASTEvaluator(BaseEvaluator):
    """
    AST (Asymmetric Student-Teacher) 基线评估器。
    基于 Reverse Distillation (反向蒸馏) 网络模型。
    适合用来统一检出相对明显且结构规则的物体（如 dowel），或者进行对比 Baseline。
    """
    def __init__(self, model=None, config=None, device=None):
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = MultimodalReverseDistillation().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.student.parameters(), lr=0.0002, weight_decay=1e-5)
        self.epochs = 50
        
    def compute_normals(self, xyz):
        """
        通过 2D Sobel 滤波器对 XYZ 坐标的深度维度进行差分计算，快速逼近生成伪 3D 表面法向量。
        此法向量可过滤绝对坐标的干扰（例如不受物体在平面中摆放位置改变的影响），
        能够凸显出表面纹理中高频的凹凸不平或边缘毛刺缺陷。
        """
        B, C, H, W = xyz.shape
        kernel_u = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3).to(xyz.device)
        kernel_v = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3).to(xyz.device)
        
        xyz_unfold = xyz.view(B*3, 1, H, W)
        grad_u = F.conv2d(xyz_unfold, kernel_u, padding=1).view(B, 3, H, W)
        grad_v = F.conv2d(xyz_unfold, kernel_v, padding=1).view(B, 3, H, W)
        
        normals = torch.cross(grad_u, grad_v, dim=1)
        normals = F.normalize(normals, p=2, dim=1)
        
        mask = (xyz[:, 2:3, :, :] != 0).float()
        normals = normals * mask
        return normals

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"AST model saved to {path}")
        
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.model.eval()
        print(f"AST model loaded from {path}")
        
    def build_feature_bank(self, train_loader):
        # We repurpose this function to train the Reverse Distillation model
        self.model.train()
        
        print("Training Student Network for Reverse Distillation...")
        for epoch in range(self.epochs):
            total_loss = 0
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.epochs}')
            for batch in pbar:
                rgb = batch['rgb'].to(self.device)
                xyz = batch['xyz'].to(self.device)
                
                self.optimizer.zero_grad()
                batch_loss = 0
                
                # Apply 4-angle rotation augmentation for robust texture learning
                for k in range(4):
                    if k > 0:
                        rgb_rot = torch.rot90(rgb, k=k, dims=[2, 3])
                        xyz_rot = torch.rot90(xyz, k=k, dims=[2, 3])
                    else:
                        rgb_rot = rgb
                        xyz_rot = xyz
                        
                    normals = self.compute_normals(xyz_rot)
                    teachers, students = self.model(rgb_rot, normals)
                    
                    # Compute a valid mask for the batch (where Z != 0)
                    fg_mask = (xyz_rot[:, 2:3, :, :] != 0).float()
                    
                    for t, s in zip(teachers, students):
                        t_norm = F.normalize(t, p=2, dim=1)
                        s_norm = F.normalize(s, p=2, dim=1)
                        dist = 1 - torch.sum(t_norm * s_norm, dim=1, keepdim=True)
                        
                        # Downsample fg_mask to feature size
                        mask_resized = F.interpolate(fg_mask, size=dist.shape[2:], mode='nearest')
                        
                        # Apply mask
                        masked_dist = dist * mask_resized
                        
                        # Mean over foreground pixels only
                        fg_pixel_count = mask_resized.sum() + 1e-5
                        batch_loss += (masked_dist.sum() / fg_pixel_count) / 4.0
                        
                batch_loss.backward()
                self.optimizer.step()
                
                total_loss += batch_loss.item()
                pbar.set_postfix({'loss': f"{batch_loss.item():.4f}"})

    def pad_score_map(self, dist_map, B):
        return dist_map

    def compute_dist_maps(self, teachers, students):
        dist_maps = []
        for t, s in zip(teachers, students):
            t_norm = F.normalize(t, p=2, dim=1)
            s_norm = F.normalize(s, p=2, dim=1)
            dist = 1 - torch.sum(t_norm * s_norm, dim=1, keepdim=True)
            dist_maps.append(F.interpolate(dist, size=(256, 256), mode='bilinear', align_corners=False))
        anomaly_map = torch.mean(torch.cat(dist_maps, dim=1), dim=1).cpu().numpy()
        return anomaly_map

    def compute_anomaly_scores(self, test_loader) -> Dict:
        self.model.eval()
        
        all_sample_scores = []
        all_sample_labels = []
        all_point_scores = []
        all_point_labels = []
        
        for batch in tqdm(test_loader, desc='Evaluating'):
            rgb = batch['rgb'].to(self.device)
            xyz = batch['xyz'].to(self.device)
            labels = batch['label'].numpy()
            gts = batch['gt'] if 'gt' in batch else None
            B = rgb.shape[0]
            
            normals = self.compute_normals(xyz)
            
            with torch.no_grad():
                teachers, students = self.model(rgb, normals)
            
            anomaly_map = self.compute_dist_maps(teachers, students)
            dist_map_smooth = np.array([cv2.GaussianBlur(m, (21, 21), 4) for m in anomaly_map])
            
            for i in range(B):
                raw_mask = (xyz[i] != 0).any(dim=0).cpu().numpy().astype(np.uint8) * 255
                kernel = np.ones((11, 11), np.uint8)
                valid_mask = cv2.erode(raw_mask, kernel, iterations=1).astype(bool)
                
                if valid_mask.sum() == 0:
                    valid_mask = (xyz[i] != 0).any(dim=0).cpu().numpy().astype(bool)
                    
                pts = dist_map_smooth[i][valid_mask]
                
                if len(pts) > 0:
                    sample_scores = np.sort(pts)[-100:].mean()
                else:
                    sample_scores = dist_map_smooth[i].max()
                    
                all_sample_scores.append(sample_scores)
                all_sample_labels.append(labels[i])
                
                if gts is not None and gts[i].sum() >= 0:
                    g_pts = gts[i].squeeze(0).numpy()[valid_mask]
                    all_point_scores.extend(pts)
                    all_point_labels.extend(g_pts)
            
        sample_auroc = roc_auc_score(all_sample_labels, all_sample_scores)
        sample_ap = average_precision_score(all_sample_labels, all_sample_scores)
        
        res = {
            'sample_level': {'auroc': sample_auroc, 'ap': sample_ap}
        }
        
        if len(all_point_labels) > 0 and sum(all_point_labels) > 0:
            res['point_level'] = {
                'auroc': roc_auc_score(all_point_labels, all_point_scores),
                'ap': average_precision_score(all_point_labels, all_point_scores)
            }
            
        return res

    def visualize_results(self, test_loader, output_dir, num_samples=5):
        os.makedirs(output_dir, exist_ok=True)
        self.model.eval()
        
        saved_good = 0
        saved_bad = 0
        
        for batch in test_loader:
            rgb = batch['rgb'].to(self.device)
            xyz = batch['xyz'].to(self.device)
            labels = batch['label'].numpy()
            gts = batch['gt'] if 'gt' in batch else None
            categories = batch.get('category', [''] * rgb.shape[0])
            defect_types = batch.get('defect', [''] * rgb.shape[0])
            B = rgb.shape[0]
            
            normals = self.compute_normals(xyz)
            with torch.no_grad():
                teachers, students = self.model(rgb, normals)
                
            anomaly_map = self.compute_dist_maps(teachers, students)
            dist_map_smooth = np.array([cv2.GaussianBlur(m, (21, 21), 4) for m in anomaly_map])
            
            for i in range(B):
                is_defect = labels[i] == 1
                if is_defect and saved_bad >= num_samples:
                    continue
                if not is_defect and saved_good >= num_samples:
                    continue
                    
                mask = (xyz[i, 2] != 0).cpu().numpy().astype(bool)
                rgb_img = rgb[i].permute(1, 2, 0).cpu().numpy()
                rgb_img = (rgb_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
                rgb_img = np.clip(rgb_img, 0, 1)
                
                z_img = xyz[i, 2].cpu().numpy().copy()
                z_img[~mask] = np.nan
                
                heat = dist_map_smooth[i].copy()
                heat[~mask] = np.nan
                pts = heat[mask]
                sample_s = np.sort(pts)[-100:].mean() if len(pts)>0 else 0
                
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                axes[0].imshow(rgb_img)
                axes[0].set_title(f"RGB ({defect_types[i]})")
                axes[0].axis('off')
                
                im_z = axes[1].imshow(z_img, cmap='viridis')
                axes[1].set_title("Depth (Z)")
                axes[1].axis('off')
                
                im_h = axes[2].imshow(heat, cmap='jet')
                axes[2].set_title(f"Heatmap (Score: {sample_s:.2f})")
                axes[2].axis('off')
                plt.colorbar(im_h, ax=axes[2], fraction=0.046, pad=0.04)
                
                if gts is not None and len(gts[i]) > 0:
                    gt_img = gts[i].squeeze(0).numpy()
                    axes[3].imshow(gt_img, cmap='gray')
                    axes[3].set_title("Ground Truth")
                axes[3].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{categories[i]}_{defect_types[i]}_{saved_good+saved_bad}.png"))
                plt.close(fig)
                
                if is_defect:
                    saved_bad += 1
                else:
                    saved_good += 1
                    
            if saved_good >= num_samples and saved_bad >= num_samples:
                break


import torchvision.models as models
from sklearn.metrics import roc_auc_score, average_precision_score
import cv2
import numpy as np

class SpatialPatchCoreEvaluator(BaseEvaluator):
    """
    Spatial PatchCore (三维空间扩展的 PatchCore) 评估器。
    通过 ResNet 提取局部 2D 特征图，再融合 3D 空间特征（XYZ 坐标以及表面法向量），
    将其平铺构建 KNN (K-Nearest Neighbors) 或 CDist 特征内存提取库。
    对于形状随机、纹理高频周期变化等极其难以建立平滑流形的问题（如泡沫 foam、轮胎 tire 等），
    此方法具有非常好的鲁棒性和精度提取表现。
    """
    def __init__(self, config=None, device=None, xyz_weight=1.0, subsample_ratio=0.01, blur_radius=4, top_k=100):
        super().__init__(config, device)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = models.resnet18(pretrained=True).to(self.device).eval()
        self.features = []
        self.xyz_weight = xyz_weight
        self.subsample_ratio = subsample_ratio
        self.blur_radius = blur_radius
        self.top_k = top_k
        
        def h(m, i, o): self.features.append(o)
        self.model.layer2.register_forward_hook(h)
        self.model.layer3.register_forward_hook(h)

    def compute_normals(self, xyz):
        """
        通过 2D Sobel 滤波器对 XYZ 坐标的深度维度进行差分计算，快速逼近生成伪 3D 表面法向量。
        此法向量可过滤绝对坐标的干扰（例如不受物体在平面中摆放位置改变的影响），
        能够凸显出表面纹理中高频的凹凸不平或边缘毛刺缺陷。
        """
        B, C, H, W = xyz.shape
        kernel_u = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3).to(xyz.device)
        kernel_v = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3).to(xyz.device)
        
        xyz_unfold = xyz.view(B*3, 1, H, W)
        grad_u = F.conv2d(xyz_unfold, kernel_u, padding=1).view(B, 3, H, W)
        grad_v = F.conv2d(xyz_unfold, kernel_v, padding=1).view(B, 3, H, W)
        
        normals = torch.cross(grad_u, grad_v, dim=1)
        normals = F.normalize(normals, p=2, dim=1)
        
        mask = (xyz[:, 2:3, :, :] != 0).float()
        normals = normals * mask
        return normals

    def extract_features(self, rgb, xyz):
        self.features = []
        with torch.no_grad():
            self.model(rgb)
            
        f1 = F.interpolate(self.features[0], size=(64, 64), mode='bilinear', align_corners=False)
        f2 = F.interpolate(self.features[1], size=(64, 64), mode='bilinear', align_corners=False)
        
        normals = self.compute_normals(xyz)
        normals_resized = F.interpolate(normals, size=(64, 64), mode='bilinear', align_corners=False) * 5.0
        
        # Spatial constraint: rescale XYZ coordinates
        z = F.interpolate(xyz[:, 2:3], size=(64, 64), mode='bilinear', align_corners=False) * self.xyz_weight
        return torch.cat([f1, f2, normals_resized, z], dim=1)

    def save_model(self, path):
        if self.memory_bank is not None:
            torch.save({'memory_bank': self.memory_bank}, path)
            print(f"SpatialPatchCore memory bank saved to {path} with size {self.memory_bank.shape}")
        else:
            print("No memory bank to save!")
            
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.memory_bank = checkpoint['memory_bank'].to(self.device)
        print(f"SpatialPatchCore memory bank loaded from {path} with size {self.memory_bank.shape}")

    def build_feature_bank(self, train_loader):
        print(f"Building Spatial PatchCore Bank... (xyz_weight={self.xyz_weight}, subsample={self.subsample_ratio})")
        lst = []
        import tqdm
        for b in tqdm.tqdm(train_loader):
            rgb = b['rgb'].to(self.device)
            xyz = b['xyz'].to(self.device)
            
            c = self.extract_features(rgb, xyz)
            
            # Mask based on native xyz, NOT the weighted one (in case weight is 0)
            z_mask = F.interpolate(xyz[:, 2:3], size=(64, 64), mode='nearest') != 0
            z_mask_flat = z_mask.permute(0, 2, 3, 1).reshape(-1)
            
            cf = c.permute(0, 2, 3, 1).reshape(-1, c.shape[1])
            fg_features = cf[z_mask_flat].cpu().numpy()
            if len(fg_features) > 0:
                lst.append(fg_features)
                
        if len(lst) == 0:
            print("Warning: no foreground features extracted.")
            self.memory_bank = torch.zeros((1, c.shape[1])).to(self.device)
            return
            
        f = np.vstack(lst)
        np.random.seed(42)
        n_samples = max(1, int(len(f) * self.subsample_ratio))
        idx = np.random.choice(len(f), n_samples, replace=False)
        
        self.memory_bank = torch.from_numpy(np.ascontiguousarray(f[idx], dtype=np.float32)).to(self.device)
        print(f"Bank size on GPU: {self.memory_bank.shape}")

    def compute_anomaly_scores(self, tl):
        import tqdm
        s_scores, s_labels, p_scores, p_labels = [], [], [], []
        B_SIZE = 2000
        for b in tqdm.tqdm(tl):
            rgb, xyz, labels, gts = b['rgb'].to(self.device), b['xyz'].to(self.device), b['label'].numpy(), b.get('gt', None)
            c = self.extract_features(rgb, xyz)
            cf = c.permute(0, 2, 3, 1).reshape(-1, c.shape[1])
            
            if len(cf) == 0 or len(self.memory_bank) == 0:
                distances = torch.zeros(rgb.shape[0]*64*64).cpu().numpy()
            else:
                dists = []
                for start in range(0, cf.shape[0], B_SIZE):
                    batch_q = cf[start:start+B_SIZE]
                    dist = torch.cdist(batch_q, self.memory_bank, p=2)
                    min_dist, _ = torch.min(dist, dim=1)
                    dists.append(min_dist)
                distances = torch.cat(dists).cpu().numpy()
            
            maps = torch.from_numpy(distances.reshape(rgb.shape[0], 64, 64)).unsqueeze(1)
            maps256 = F.interpolate(maps, size=256, mode='bilinear', align_corners=False).squeeze(1).numpy()
            
            for i in range(rgb.shape[0]):
                m = maps256[i]
                if self.blur_radius > 0:
                    kernel_size = 2 * self.blur_radius + 1
                    m = cv2.GaussianBlur(m, (kernel_size, kernel_size), self.blur_radius)
                
                v = (xyz[i, 2] != 0).cpu().numpy().astype(bool)
                pts = m[v]
                s_scores.append(np.sort(pts)[-self.top_k:].mean() if len(pts)>0 else m.max())
                s_labels.append(labels[i])
                if gts is not None and gts[i].sum() >= 0:
                    p_scores.extend(pts)
                    p_labels.extend(gts[i].squeeze(0).numpy()[v])
        
        res = {'sample_level': {'auroc': roc_auc_score(s_labels, s_scores), 'ap': average_precision_score(s_labels, s_scores)}}
        if p_labels:
            res['point_level'] = {'auroc': roc_auc_score(p_labels, p_scores), 'ap': average_precision_score(p_labels, p_scores)}
        return res


    def visualize_results(self, test_loader, output_dir, num_samples=5):
        import os
        import numpy as np
        import cv2
        import matplotlib.pyplot as plt
        import torch.nn.functional as F

        os.makedirs(output_dir, exist_ok=True)
        self.model.eval()
        
        saved_good = 0
        saved_bad = 0
        
        for batch in test_loader:
            with torch.no_grad():
                rgb = batch['rgb'].to(self.device)
                xyz = batch['xyz'].to(self.device)
                labels = batch['label'].numpy()
                gts = batch['gt'] if 'gt' in batch else None
                categories = batch.get('category', [''] * rgb.shape[0])
                defect_types = batch.get('defect', [''] * rgb.shape[0])
                
                features = self.extract_features(rgb, xyz)
                z_mask = F.interpolate(xyz[:, 2:3], size=(64, 64), mode='nearest') != 0
                z_mask_flat = z_mask.permute(0, 2, 3, 1).reshape(-1)
                
                cf = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
                fg_features = cf[z_mask_flat].cpu()
                
                distances = torch.zeros(cf.shape[0]).cpu().numpy()
                
                if len(fg_features) > 0 and len(self.memory_bank) > 0:
                    if torch.cuda.is_available():
                        fg_cuda = fg_features.cuda()
                        B_SIZE = 2000
                        dists = []
                        for start in range(0, fg_cuda.shape[0], B_SIZE):
                            batch_q = fg_cuda[start:start+B_SIZE]
                            dist = torch.cdist(batch_q, self.memory_bank, p=2)
                            min_dist, _ = torch.min(dist, dim=1)
                            dists.append(min_dist.cpu())
                        fg_distances = torch.cat(dists).numpy()
                    else:
                        dists = []
                        for start in range(0, fg_features.shape[0], 16384):
                            batch_q = fg_features[start:start+16384]
                            dist = torch.cdist(batch_q, self.memory_bank, p=2)
                            min_dist, _ = torch.min(dist, dim=1)
                            dists.append(min_dist)
                        fg_distances = torch.cat(dists).numpy()
                        
                    distances[z_mask_flat.cpu().numpy()] = fg_distances
                    
                maps = torch.from_numpy(distances.reshape(rgb.shape[0], 64, 64)).unsqueeze(1)
                maps256 = F.interpolate(maps, size=256, mode='bilinear', align_corners=False).squeeze(1).numpy()
                valid_mask = (xyz[:, 2, :, :] != 0).cpu().numpy()
                
                dist_map_smooth = maps256.copy()
                if self.blur_radius > 0:
                    k = self.blur_radius * 2 + 1
                    dist_map_smooth = np.array([cv2.GaussianBlur(m, (k, k), self.blur_radius) for m in maps256])
                
                B = rgb.shape[0]
                for i in range(B):
                    is_defect = labels[i] == 1
                    if is_defect and saved_bad >= num_samples:
                        continue
                    if not is_defect and saved_good >= num_samples:
                        continue
                        
                    mask = valid_mask[i]
                    rgb_img = rgb[i].permute(1, 2, 0).cpu().numpy()
                    rgb_img = (rgb_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
                    rgb_img = np.clip(rgb_img, 0, 1)
                    
                    z_img = xyz[i, 2].cpu().numpy().copy()
                    z_img[~mask] = np.nan
                    
                    heat = dist_map_smooth[i].copy()
                    heat[~mask] = np.nan
                    pts = heat[mask]
                    sample_s = np.sort(pts)[-self.top_k:].mean() if len(pts)>0 else 0
                    
                    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                    axes[0].imshow(rgb_img)
                    axes[0].set_title(f"RGB ({defect_types[i]})")
                    axes[0].axis('off')
                    
                    im_z = axes[1].imshow(z_img, cmap='viridis')
                    axes[1].set_title("Depth (Z)")
                    axes[1].axis('off')
                    
                    im_h = axes[2].imshow(heat, cmap='jet')
                    axes[2].set_title(f"Heatmap (Score: {sample_s:.2f})")
                    axes[2].axis('off')
                    plt.colorbar(im_h, ax=axes[2], fraction=0.046, pad=0.04)
                    
                    if gts is not None and len(gts[i]) > 0:
                        gt_img = gts[i].squeeze(0).numpy()
                        axes[3].imshow(gt_img, cmap='gray')
                        axes[3].set_title("Ground Truth")
                    axes[3].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"{categories[i]}_{defect_types[i]}_{saved_good+saved_bad}.png"))
                    plt.close(fig)
                    
                    if is_defect:
                        saved_bad += 1
                    else:
                        saved_good += 1
                        
                if saved_good >= num_samples and saved_bad >= num_samples:
                    break


def get_evaluator(category, config=None, device=None):
    if category == 'cable_gland':
        print(f"[Router] Using Spatial PatchCore (High Spatial Weight) for: {category}")
        return SpatialPatchCoreEvaluator(config, device, xyz_weight=15.0, subsample_ratio=0.02, blur_radius=4)
        
    elif category == 'foam':
        print(f"[Router] Using Spatial PatchCore (Low Spatial Weight) for: {category}")
        # Foam pores are random, spatial xyz weighting should be low or zero so it behaves like pure 2D PatchCore
        return SpatialPatchCoreEvaluator(config, device, xyz_weight=0.0, subsample_ratio=0.2, blur_radius=8, top_k=400)
        
    elif category == 'tire':
        print(f"[Router] Using Spatial PatchCore (Ultra Spatial Weight) for: {category}")
        return SpatialPatchCoreEvaluator(config, device, xyz_weight=0.0, subsample_ratio=0.2, blur_radius=2, top_k=50)

    elif category == 'rope':
        print(f"[Router] Using Spatial PatchCore for: {category} (Tuned for high-freq texture)")
        return SpatialPatchCoreEvaluator(config, device, xyz_weight=0.0, subsample_ratio=0.2, blur_radius=8, top_k=400)
        
    else:
        print(f"[Router] Using Default ASTEvaluator for: {category}")
        return SpatialPatchCoreEvaluator(config, device, xyz_weight=0.0, subsample_ratio=0.2, blur_radius=2, top_k=50)
