import torch
import torch.nn as nn

# 导入你之前的两个完整模型，目的是为了安全地提取它们的主干网络 (Backbone)
from train_student_causal import SingleLeadCausalECGNet
from model_inception_causal import InceptionCausalECGNet, AttentionFusion

# ==========================================
# 1. ResNet 宏观单分支因果网络 (Ablation 1)
# ==========================================
class ResNetCausal_MacroOnly(nn.Module):
    def __init__(self, num_classes=12, hidden_channels=64):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        
        # 安全提取原始模型的主干网络，不破坏原代码
        orig_model = SingleLeadCausalECGNet(num_classes=num_classes, hidden_channels=hidden_channels)
        self.backbone = orig_model.backbone
        
        # 只保留捕捉宏观节律的大卷积核 T2
        self.T_macro = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=50, padding=25)
        
        # 单一的注意力融合与分类器
        self.af_macro = AttentionFusion(feature_dim=hidden_channels, num_classes=num_classes)
        self.mlp_macro = nn.Sequential(nn.Linear(hidden_channels * 2, hidden_channels), nn.ReLU())
        self.classifier = nn.Linear(hidden_channels, num_classes)
        
        self.register_buffer('confounder_dict', torch.zeros(num_classes, hidden_channels))

    def forward(self, x, labels=None, current_epoch=0):
        f_m = self.backbone(x) 
        Q_macro = self.T_macro(f_m).mean(dim=-1)  
        
        warmup_epochs = 5 
        if self.training and labels is not None and current_epoch >= warmup_epochs:
            with torch.no_grad():
                alpha = 0.9 
                for c in range(self.num_classes):
                    mask = (labels[:, c] > 0.5) 
                    if mask.any():
                        class_feat = Q_macro[mask].mean(dim=0)
                        if torch.sum(self.confounder_dict[c]) == 0:
                            self.confounder_dict[c] = class_feat
                        else:
                            self.confounder_dict[c] = alpha * self.confounder_dict[c] + (1 - alpha) * class_feat
        
        f_fused = self.af_macro(Q_macro, self.confounder_dict) 
        f_cau = self.mlp_macro(f_fused) 
        pred = self.classifier(f_cau)
        
        # 注意：这里直接返回单一预测值，不需要再区分 upper 和 lower
        return pred

# ==========================================
# 2. Inception 宏观单分支因果网络 (Ablation 2)
# ==========================================
class InceptionCausal_MacroOnly(nn.Module):
    def __init__(self, num_classes=12, hidden_channels=64):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        
        # 安全提取 Inception 原创模型的主干网络
        orig_model = InceptionCausalECGNet(num_classes=num_classes, hidden_channels=hidden_channels)
        self.backbone = orig_model.backbone
        
        # 只保留捕捉宏观节律的大卷积核 T2
        self.T_macro = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=50, padding=25)
        
        # 单一的注意力融合与分类器
        self.af_macro = AttentionFusion(feature_dim=hidden_channels, num_classes=num_classes)
        self.mlp_macro = nn.Sequential(nn.Linear(hidden_channels * 2, hidden_channels), nn.ReLU())
        self.classifier = nn.Linear(hidden_channels, num_classes)
        
        self.register_buffer('confounder_dict', torch.zeros(num_classes, hidden_channels))

    def forward(self, x, labels=None, current_epoch=0):
        f_m = self.backbone(x) 
        Q_macro = self.T_macro(f_m).mean(dim=-1)  
        
        warmup_epochs = 5 
        if self.training and labels is not None and current_epoch >= warmup_epochs:
            with torch.no_grad():
                alpha = 0.9 
                for c in range(self.num_classes):
                    mask = (labels[:, c] > 0.5) 
                    if mask.any():
                        class_feat = Q_macro[mask].mean(dim=0)
                        if torch.sum(self.confounder_dict[c]) == 0:
                            self.confounder_dict[c] = class_feat
                        else:
                            self.confounder_dict[c] = alpha * self.confounder_dict[c] + (1 - alpha) * class_feat
        
        f_fused = self.af_macro(Q_macro, self.confounder_dict) 
        f_cau = self.mlp_macro(f_fused) 
        pred = self.classifier(f_cau)
        return pred

class InceptionCausal_SmallKernel(nn.Module):
    """
    消融实验 1：将宏观分支的大卷积核(K=50)退化为小卷积核(K=5)
    """
    def __init__(self, num_classes=12, hidden_channels=64):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        
        orig_model = InceptionCausalECGNet(num_classes=num_classes, hidden_channels=hidden_channels)
        self.backbone = orig_model.backbone
        
        # 🚀 核心消融点：kernel_size 从 50 降到 5，padding 从 25 降到 2 保持维度对齐
        self.T_macro = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2)
        
        self.af_macro = AttentionFusion(feature_dim=hidden_channels, num_classes=num_classes)
        self.mlp_macro = nn.Sequential(nn.Linear(hidden_channels * 2, hidden_channels), nn.ReLU())
        self.classifier = nn.Linear(hidden_channels, num_classes)
        
        self.register_buffer('confounder_dict', torch.zeros(num_classes, hidden_channels))

    def forward(self, x, labels=None, current_epoch=0):
        f_m = self.backbone(x) 
        Q_macro = self.T_macro(f_m).mean(dim=-1)  
        
        warmup_epochs = 5 
        if self.training and labels is not None and current_epoch >= warmup_epochs:
            with torch.no_grad():
                alpha = 0.9 
                for c in range(self.num_classes):
                    mask = (labels[:, c] > 0.5) 
                    if mask.any():
                        class_feat = Q_macro[mask].mean(dim=0)
                        if torch.sum(self.confounder_dict[c]) == 0:
                            self.confounder_dict[c] = class_feat
                        else:
                            self.confounder_dict[c] = alpha * self.confounder_dict[c] + (1 - alpha) * class_feat
        
        f_fused = self.af_macro(Q_macro, self.confounder_dict) 
        f_cau = self.mlp_macro(f_fused) 
        return self.classifier(f_cau)

class ResNetCausal_MacroOnly(nn.Module):
    """
    消融实验 2：用单尺度 ResNet 替换多尺度 Inception 主干网络
    """
    def __init__(self, num_classes=12, hidden_channels=64):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        
        # 🚀 核心消融点：提取 ResNet 的 Backbone
        orig_model = SingleLeadCausalECGNet(num_classes=num_classes, hidden_channels=hidden_channels)
        self.backbone = orig_model.backbone
        
        # 保留 K=50 大卷积核和因果字典逻辑
        self.T_macro = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=50, padding=25)
        
        self.af_macro = AttentionFusion(feature_dim=hidden_channels, num_classes=num_classes)
        self.mlp_macro = nn.Sequential(nn.Linear(hidden_channels * 2, hidden_channels), nn.ReLU())
        self.classifier = nn.Linear(hidden_channels, num_classes)
        
        self.register_buffer('confounder_dict', torch.zeros(num_classes, hidden_channels))

    def forward(self, x, labels=None, current_epoch=0):
        f_m = self.backbone(x) 
        Q_macro = self.T_macro(f_m).mean(dim=-1)  
        
        warmup_epochs = 5 
        if self.training and labels is not None and current_epoch >= warmup_epochs:
            with torch.no_grad():
                alpha = 0.9 
                for c in range(self.num_classes):
                    mask = (labels[:, c] > 0.5) 
                    if mask.any():
                        class_feat = Q_macro[mask].mean(dim=0)
                        if torch.sum(self.confounder_dict[c]) == 0:
                            self.confounder_dict[c] = class_feat
                        else:
                            self.confounder_dict[c] = alpha * self.confounder_dict[c] + (1 - alpha) * class_feat
        
        f_fused = self.af_macro(Q_macro, self.confounder_dict) 
        f_cau = self.mlp_macro(f_fused) 
        return self.classifier(f_cau)
    
class Inception_NoCausal_MacroOnly(nn.Module):
    """
    消融实验 3：彻底移除因果干预字典和注意力融合，直接输出预测
    """
    def __init__(self, num_classes=12, hidden_channels=64):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        
        orig_model = InceptionCausalECGNet(num_classes=num_classes, hidden_channels=hidden_channels)
        self.backbone = orig_model.backbone
        
        # 依然保留 K=50 宏观卷积核
        self.T_macro = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=50, padding=25)
        
        # 🚀 核心消融点 1：去掉了 self.af_macro 和 self.confounder_dict
        
        # 🚀 核心消融点 2：因为没有了注意力特征拼接，输入维度从 hidden_channels*2 改为 hidden_channels
        self.mlp_macro = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU())
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, labels=None, current_epoch=0):
        # 基础特征提取
        f_m = self.backbone(x) 
        # 宏观特征捕捉
        Q_macro = self.T_macro(f_m).mean(dim=-1)  
        
        # 🚀 核心消融点 3：彻底删除了动量更新字典的 warmup 代码块
        
        # 特征直接穿透 MLP 和分类器
        f_out = self.mlp_macro(Q_macro) 
        return self.classifier(f_out)