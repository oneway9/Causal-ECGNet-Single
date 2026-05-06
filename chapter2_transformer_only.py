import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

# 导入你的公共模块
from chapter2_combined_improvements import (
    Integrated_ECG_Dataset, AsymmetricLoss, find_optimal_thresholds, calculate_metrics_with_custom_thresholds
)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ==========================================
# 1. 深度优化后的 Vanilla Transformer
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500): # 这里的 max_len 随 stride 调整
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Standard_1D_Transformer(nn.Module):
    def __init__(self, c_in=1, c_out=12, d_model=64, nhead=8, num_layers=4):
        super().__init__()
        # 🚀 核心优化：将 stride 提高到 10
        # 5000 长度会被压缩到 500，Attention 矩阵大小从 1250^2 降至 500^2，计算量骤降 6.25 倍
        self.stem = nn.Sequential(
            nn.Conv1d(c_in, d_model, kernel_size=15, stride=10, padding=7, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        self.pos_encoder = PositionalEncoding(d_model, max_len=500) 
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, c_out)

    def forward(self, x):
        x = self.stem(x) 
        x = x.transpose(1, 2)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.fc(x.mean(dim=1))

def main():
    PTBXL_PATH = './ptb-xl/' 
    MIT_NPY_PATH = './outputs/mit_teacher_labeled.npy'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_NAME = "Vanilla_Transformer"
    
    # 🚀 核心优化：降低 Batch Size 至 16，甚至 8（如果你的显存小于 8GB）
    BATCH_SIZE = 16 
    
    print(f"\n⚡ 专项攻坚：正在加载整合数据集评测 {MODEL_NAME}...")
    train_loader = DataLoader(Integrated_ECG_Dataset(PTBXL_PATH, MIT_NPY_PATH, 'train'), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(Integrated_ECG_Dataset(PTBXL_PATH, MIT_NPY_PATH, 'val'), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(Integrated_ECG_Dataset(PTBXL_PATH, MIT_NPY_PATH, 'test'), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = Standard_1D_Transformer(c_in=1, c_out=12).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05).to(DEVICE)
    
    best_val_auc = 0.0
    save_path = f'./outputs/baseline_integrated_{MODEL_NAME}_best.pth'
    
    #for epoch in range(30):
    #    model.train()
     #   train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/30', leave=False)
    #    for signals, labels in train_bar:
     #       signals, labels = signals.to(DEVICE), labels.to(DEVICE)
     #       optimizer.zero_grad()
    #        loss = criterion(model(signals), labels)
    #        loss.backward()
     #       optimizer.step()
    #    scheduler.step()

      #  # 验证集评估
     #   model.eval()
    #    v_p, v_l = [], []
     #   with torch.no_grad():
     #       for s, l in val_loader:
     #           v_p.extend(torch.sigmoid(model(s.to(DEVICE))).cpu().numpy())
     #           v_l.extend(l.numpy())
     #   val_auc, _, _, _, _ = calculate_metrics_with_custom_thresholds(np.array(v_l), np.array(v_p), np.full(12, 0.5))
     #   
     #   print(f"[*] Epoch {epoch+1} Val AUC: {val_auc:.2f}%")
     #   if val_auc > best_val_auc:
       #     best_val_auc = val_auc
        #    torch.save(model.state_dict(), save_path)

    # 最终盲测
    print(f"\n[*] 开始终极盲测...")
    # 直接加载刚才千辛万苦跑出来的最佳权重
    model.load_state_dict(torch.load(save_path))
    model.eval()
    
    v_p, v_l, t_p, t_l = [], [], [], [] # 规范变量名
    with torch.no_grad():
        for s, l in val_loader: 
            v_p.extend(torch.sigmoid(model(s.to(DEVICE))).cpu().numpy())
            v_l.extend(l.numpy())
            
        opt_t = find_optimal_thresholds(np.array(v_l), np.array(v_p))
        
        for s, l in test_loader: 
            t_p.extend(torch.sigmoid(model(s.to(DEVICE))).cpu().numpy())
            t_l.extend(l.numpy()) # 🚀 修复点 1：tl 改为 t_l
    
    # 🚀 修复点 2：传入 t_l 和 t_p
    res = calculate_metrics_with_custom_thresholds(np.array(t_l), np.array(t_p), opt_t)
    
    print("\n" + "★"*60)
    print(f" {MODEL_NAME} 最终成绩：")
    print(f" Mac-AUC: {res[0]:.2f}%")
    print(f" Mac-SEN: {res[1]:.2f}%")
    print(f" Mac-SPE: {res[2]:.2f}%")
    print(f" Mac-F1:  {res[3]:.2f}%")
    print(f" Mic-F1:  {res[4]:.2f}%")
    print("★"*60)

if __name__ == '__main__':
    main()