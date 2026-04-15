import os
import cv2
import numpy as np
import torch
import tifffile as tiff
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MVTec3D2DDataset(Dataset):
    def __init__(self, config, category, split='train'):
        self.root = config.raw_data_root
        self.category = category
        self.split = split
        self.image_size = config.image_size
        
        self.rgb_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        self.samples = self._load_samples()
        
    def _load_samples(self):
        base_dir = os.path.join(self.root, self.category, self.split)
        samples = []
        defect_types = os.listdir(base_dir) if self.split == 'test' else ['good']
        
        for defect in defect_types:
            rgb_dir = os.path.join(base_dir, defect, 'rgb')
            xyz_dir = os.path.join(base_dir, defect, 'xyz')
            gt_dir = os.path.join(base_dir, defect, 'gt')
            
            if not os.path.exists(rgb_dir): continue
            
            for f in sorted(os.listdir(rgb_dir)):
                if not f.endswith('.png'): continue
                
                name = os.path.splitext(f)[0]
                rgb_path = os.path.join(rgb_dir, f)
                xyz_path = os.path.join(xyz_dir, name + '.tiff')
                gt_path = os.path.join(gt_dir, f) if defect != 'good' else None
                
                label = 0 if defect == 'good' else 1
                
                samples.append({
                    'rgb': rgb_path,
                    'xyz': xyz_path,
                    'gt': gt_path,
                    'defect': defect,
                    'label': label,
                    'category': self.category
                })
        return samples
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        s = self.samples[idx]
        
        rgb = Image.open(s['rgb']).convert('RGB')
        rgb_tensor = self.rgb_transform(rgb)
        
        xyz = tiff.imread(s['xyz'])
        xyz = cv2.resize(xyz, (self.image_size, self.image_size))
        xyz = np.nan_to_num(xyz, nan=0.0)
        
        xyz_valid = xyz[np.where((xyz!=0).all(axis=2))]
        if len(xyz_valid) > 0:
            mean = xyz_valid.mean(axis=0)
            std = xyz_valid.std(axis=0) + 1e-6
            xyz_norm = (xyz - mean) / std
        else:
            xyz_norm = xyz
            
        xyz_tensor = torch.from_numpy(xyz_norm).float().permute(2, 0, 1)
        
        gt_tensor = torch.zeros(1, self.image_size, self.image_size)
        if s['gt'] and os.path.exists(s['gt']):
            gt = Image.open(s['gt']).convert('L')
            gt = gt.resize((self.image_size, self.image_size), Image.NEAREST)
            gt_tensor = (torch.from_numpy(np.array(gt)) > 0).float().unsqueeze(0)
            
        return {
            'rgb': rgb_tensor,
            'xyz': xyz_tensor,
            'gt': gt_tensor,
            'label': s['label'],
            'defect': s['defect'],
            'category': s['category']
        }

def create_dataloaders(config, categories=None, evaluate_only=False):
    if categories is None:
        categories = config.categories
        
    category = categories[0]
    
    train_dataset = MVTec3D2DDataset(config, category, split='train')
    val_dataset = MVTec3D2DDataset(config, category, split='validation')
    test_dataset = MVTec3D2DDataset(config, category, split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=False, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False, num_workers=config.num_workers)
    
    return train_loader, val_loader, test_loader
