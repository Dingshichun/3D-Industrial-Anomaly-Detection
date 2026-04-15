import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class FeatureExtractor(nn.Module):
    """
    预训练特征提取器 (纯推断无需梯度)。
    基于 ResNet18 构建，用来抽取出 RGB 颜色特征 或 XYZ/法向坐标 特征。
    """
    def __init__(self):
        super().__init__()
        base = models.resnet18(pretrained=True)
        # Layer1: 取到 ResNet 的首层最大池化后输出
        self.layer1 = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool, base.layer1)
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        
        # 冻结模型所有参数（仅作为 Teacher 特征提取用）
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        feat1 = self.layer1(x) # [Batch, 64, H/4, W/4]
        feat2 = self.layer2(feat1) # [Batch, 128, H/8, W/8]
        feat3 = self.layer3(feat2) # [Batch, 256, H/16, W/16]
        return feat1, feat2, feat3

class Bottleneck(nn.Module):
    """
    Reverse Distillation (反向蒸馏) 模型的倒置瓶颈层。
    在解码前对最深层次特征进行非线性通道变换与重整。
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1), # 1x1 卷积用于调整通道数，减少计算量并引入非线性变换。
            nn.BatchNorm2d(256), # BN 层有助于稳定训练过程，尤其在小批量训练时可以缓解内部协变量偏移问题。
            nn.ReLU(inplace=True) # 直接修改输入张量，不保存输入的副本，节省内存并加速计算。
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv2(self.conv1(x))

class UpBlock(nn.Module):
    """
    解码器的上采样块，用于在上采样的同时与跳跃连接（Skip Connection）传入的同级特征图融合。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, skip=None):
        # 宽、高上采样 2 倍
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if skip is not None:
            # 融合上一层解码结果和同级的 Teacher 特征
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class StudentDecoder(nn.Module):
    """
    Student (学生网络) 解码器。
    用于在多模态反向蒸馏中，根据高阶语义映射出包含异常的高级特征结构并试图逼近正常特征流形。
    若传入测试样本包含未见过的缺陷特征，由于学生网络没用异常数据训练过，因此输出会与 Teacher 特征拉开差距。
    """
    def __init__(self, c1=128, c2=256, c3=512): 
        super().__init__()
        self.bottleneck = Bottleneck(c3)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, c3, kernel_size=3, padding=1), 
            nn.BatchNorm2d(c3), 
            nn.ReLU(inplace=True)
        )
        self.dec2 = UpBlock(c3 + c2, c2)
        self.dec1 = UpBlock(c2 + c1, c1)
        
        self.out3 = nn.Conv2d(c3, c3, kernel_size=1)
        self.out2 = nn.Conv2d(c2, c2, kernel_size=1)
        self.out1 = nn.Conv2d(c1, c1, kernel_size=1)

    def forward(self, f1, f2, f3):
        b = self.bottleneck(f3) 
        r3 = self.dec3(b) 
        out3 = self.out3(r3)
        
        r2 = self.dec2(r3, skip=f2) 
        out2 = self.out2(r2)
        
        r1 = self.dec1(r2, skip=f1) 
        out1 = self.out1(r1)
        
        # 返回 3 个层级的对应特征重建图
        return out1, out2, out3

class MultimodalReverseDistillation(nn.Module):
    """
    多模态反向蒸馏核心架构 (AST Baseline 模型核心)。
    融合 2D RGB 特征和 3D 点云（XYZ或法向）特征，构建教师流形空间，并训练学生网络追踪重建该特征。
    """
    def __init__(self):
        super().__init__()
        self.teacher_rgb = FeatureExtractor()      # 专用 RGB 分支
        self.teacher_xyz = FeatureExtractor()      # 专用 PointCloud/Normal 分支
        self.teacher_rgb.eval()
        self.teacher_xyz.eval()
        # 学生解码器拼接由于是两路特征的 concat，维度需要按 (64+64, 128+128, 256+256) 修改
        self.student = StudentDecoder(c1=128, c2=256, c3=512)
        
    def train(self, mode=True):
        # 重写 train 函数，保证两路 Teacher 永远是 eval（推理模式，不更新BN）
        super().train(mode)
        self.teacher_rgb.eval()
        self.teacher_xyz.eval()

    def forward(self, rgb, xyz_normals):
        # 冻结梯度提取老师特征
        with torch.no_grad():
            r_f1, r_f2, r_f3 = self.teacher_rgb(rgb)
            x_f1, x_f2, x_f3 = self.teacher_xyz(xyz_normals)
            
        # 多模态通道融合
        f1 = torch.cat([r_f1, x_f1], dim=1) 
        f2 = torch.cat([r_f2, x_f2], dim=1) 
        f3 = torch.cat([r_f3, x_f3], dim=1) 
        
        # 学生网络还原
        out1, out2, out3 = self.student(f1, f2, f3)
        
        # 训练和评估时需要比照 Student 产出和 Teacher 教样来计算均方/余弦距离偏差
        return (f1, f2, f3), (out1, out2, out3)
