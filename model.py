import torch
import torch.nn as nn

# ==========================================
# 1. 残差块 (ResNet-1D Block)
# ==========================================
class ResNet1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=7):
        super(ResNet1DBlock, self).__init__()
        padding = kernel_size // 2
        
        # 第一层卷积
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 第二层卷积
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 残差连接 (Skip Connection)
        self.downsample = nn.Sequential()
        # 如果步长不为1或输入输出通道数不同，需要使用 1x1 卷积调整维度以对齐相加
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 残差相加
        out += identity
        out = self.relu(out)

        return out

# ==========================================
# 2. 基于 ResNet 的主干网络
# ==========================================
class ResNetBackbone(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=64):
        super(ResNetBackbone, self).__init__()
        
        # 初始降采样层：快速减小时间序列长度，提取浅层边缘特征
        self.first_layer = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # 堆叠 4 个 ResNet Block，逐步增加通道数并缩短时间维度
        self.layer1 = ResNet1DBlock(32, 32, stride=1, kernel_size=7)
        self.layer2 = ResNet1DBlock(32, hidden_channels, stride=2, kernel_size=7)
        self.layer3 = ResNet1DBlock(hidden_channels, hidden_channels, stride=2, kernel_size=7)
        self.layer4 = ResNet1DBlock(hidden_channels, hidden_channels, stride=2, kernel_size=7)

        # 最终池化：将时间维度统一压缩到 125，与后续的 TDFE 模块无缝衔接
        self.pool = nn.AdaptiveAvgPool1d(125)

    def forward(self, x):
        x = self.first_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return x

# ==========================================
# 3. 注意力融合与因果推理模块
# ==========================================
class AttentionFusion(nn.Module):
    def __init__(self, feature_dim=64, num_classes=12):
        super().__init__()
        self.W_k = nn.Linear(feature_dim, feature_dim)
        self.W_v = nn.Linear(feature_dim, feature_dim)
        self.scale = feature_dim ** 0.5
        
    def forward(self, Q, f_dict):
        K = self.W_k(f_dict) 
        V = self.W_v(f_dict) 
        
        attn_scores = torch.matmul(Q, K.transpose(0, 1)) / self.scale 
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, V) 
        
        f_fused = torch.cat([Q, attn_out], dim=-1) 
        return f_fused

# ==========================================
# 4. 完整的 Causal ECGNet 网络
# ==========================================
class CausalECGNet(nn.Module):
    def __init__(self, num_classes=12, hidden_channels=64):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        
        # === 核心改进：替换为 ResNet-1D 主干网络 ===
        self.backbone = ResNetBackbone(in_channels=1, hidden_channels=hidden_channels)
        
        self.T1 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2)
        self.T2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=50, padding=25)
        
        self.af_upper = AttentionFusion(feature_dim=hidden_channels, num_classes=num_classes)
        self.af_lower = AttentionFusion(feature_dim=hidden_channels, num_classes=num_classes)
        
        self.mlp_upper = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU()
        )
        self.mlp_lower = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU()
        )
        
        self.classifier_upper = nn.Linear(hidden_channels, num_classes)
        self.classifier_lower = nn.Linear(hidden_channels, num_classes)
        
        self.register_buffer('confounder_dict', torch.zeros(num_classes, hidden_channels))

    # 请在 CausalECGNet 类中，完全替换原来的 forward 函数：
    def forward(self, x, labels=None, current_epoch=0):
        views = torch.split(x, 2, dim=1) 
        
        upper_features = []
        lower_features = []
        
        for view in views:
            f_m = self.backbone(view) 
            f_up = self.T1(f_m).mean(dim=-1)   
            f_low = self.T2(f_m).mean(dim=-1)  
            
            upper_features.append(f_up)
            lower_features.append(f_low)
            
        Q_upper = torch.stack(upper_features, dim=1).mean(dim=1) 
        Q_lower = torch.stack(lower_features, dim=1).mean(dim=1) 
        
        # === 核心改进：字典的 Warm-up 与初始化策略 ===
        warmup_epochs = 5 # 前 5 个 epoch (0~4) 不更新字典
        if self.training and labels is not None and current_epoch >= warmup_epochs:
            with torch.no_grad():
                alpha = 0.9 
                for c in range(self.num_classes):
                    mask = (labels[:, c] == 1.0) 
                    if mask.any():
                        class_feat = ((Q_upper[mask] + Q_lower[mask]) / 2).mean(dim=0)
                        
                        # 如果是全 0（第一次更新），直接赋值，避免极小值陷阱
                        if torch.sum(self.confounder_dict[c]) == 0:
                            self.confounder_dict[c] = class_feat
                        else:
                            # 之后进行正常的 EMA 平滑更新
                            self.confounder_dict[c] = alpha * self.confounder_dict[c] + (1 - alpha) * class_feat
        
        f_fused_upper = self.af_upper(Q_upper, self.confounder_dict) 
        f_fused_lower = self.af_lower(Q_lower, self.confounder_dict) 
        
        f_cau_upper = self.mlp_upper(f_fused_upper) 
        f_cau_lower = self.mlp_lower(f_fused_lower) 
        
        pred_upper = self.classifier_upper(f_cau_upper)
        pred_lower = self.classifier_lower(f_cau_lower)
        
        if self.training:
            return pred_upper, pred_lower
        else:
            return (pred_upper + pred_lower) / 2