import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

# ==========================================
# 1. 导入整合数据集与完美裁判系统
# ==========================================
from chapter2_combined_improvements import (
    Integrated_ECG_Dataset, AsymmetricLoss, find_optimal_thresholds, calculate_metrics_with_custom_thresholds
)

# ==========================================
# 2. 导入 baselines 里的参赛选手
# ==========================================
from baselines.basic_conv1d import FCN
from baselines.inception1d import Inception1d
from baselines.xresnet1d import xresnet1d101
from baselines.bilstm import Standalone_BiLSTM
from baselines.crnn import CRNN
from baselines.xception_time import XceptionTime

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ==========================================
# 3. 内嵌 Vanilla Transformer (1D 适配版)
# ==========================================
#class PositionalEncoding(nn.Module):
#    def __init__(self, d_model, max_len=5000):
 #       super().__init__()
 #       pe = torch.zeros(max_len, d_model)
 #       position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
 #       div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
 #       pe[:, 0::2] = torch.sin(position * div_term)
#        pe[:, 1::2] = torch.cos(position * div_term)
#        self.register_buffer('pe', pe.unsqueeze(0))
 #   def forward(self, x):
  #      return x + self.pe[:, :x.size(1)]

class Standard_1D_Transformer(nn.Module):
    def __init__(self, c_in=1, c_out=12, d_model=64, nhead=8, num_layers=4):
        super().__init__()
        # 前端卷积降维层: Stride=4 将 5000 长度压缩到 1250，大幅降低显存占用
        self.stem = nn.Sequential(
            nn.Conv1d(c_in, d_model, kernel_size=15, stride=4, padding=7, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        self.pos_encoder = PositionalEncoding(d_model, max_len=1250)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, c_out)

    def forward(self, x):
        x = self.stem(x) 
        x = x.transpose(1, 2)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.fc(x.mean(dim=1))

# ==========================================
# 4. 核心训练与评估逻辑
# ==========================================
def train_and_evaluate(model_name, model, train_loader, val_loader, test_loader, device):
    print(f"\n" + "="*60)
    print(f" 🚀 正在评测基线模型: {model_name} [数据集: PTB-XL + MIT-BIH 整合]")
    print("="*60)
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
    # 统一使用高级非对称损失函数，解决医疗数据长尾不平衡问题
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05).to(device)
    
    best_val_auc = 0.0
    model_save_path = f'./outputs/baseline_integrated_{model_name}_best.pth'
    
    # 1. 统一训练 30 轮
    for epoch in range(30):
        model.train()
        for signals, labels in tqdm(train_loader, desc=f'[{model_name}] Epoch {epoch+1}/30', leave=False):
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(signals), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for signals, labels in val_loader:
                all_probs.extend(torch.sigmoid(model(signals.to(device))).cpu().numpy())
                all_labels.extend(labels.numpy())
                
        val_auc, _, _, _, _ = calculate_metrics_with_custom_thresholds(
            np.array(all_labels), np.array(all_probs), np.full(12, 0.5)
        )
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), model_save_path)
            
    # 2. 统一使用 F1-Maximized 进行终极盲测
    print(f"\n[*] {model_name} 训练完成，正在进行动态阈值校准及盲测...")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()
    
    val_probs, val_labels = [], []
    with torch.no_grad():
        for signals, labels in val_loader:
            val_probs.extend(torch.sigmoid(model(signals.to(device))).cpu().numpy())
            val_labels.extend(labels.numpy())
    optimal_thresholds = find_optimal_thresholds(np.array(val_labels), np.array(val_probs))
    
    test_probs, test_labels = [], []
    with torch.no_grad():
        for signals, labels in test_loader:
            test_probs.extend(torch.sigmoid(model(signals.to(device))).cpu().numpy())
            test_labels.extend(labels.numpy())
            
    mac_auc, mac_sen, mac_spe, mac_f1, micro_f1 = calculate_metrics_with_custom_thresholds(
        np.array(test_labels), np.array(test_probs), optimal_thresholds
    )
    
    # 返回该模型的所有核心指标
    return {
        "Model": model_name,
        "Macro-AUC": mac_auc,
        "Macro-SEN": mac_sen,
        "Macro-SPE": mac_spe,
        "Macro-F1": mac_f1,
        "Micro-F1": micro_f1
    }

def main():
    PTBXL_PATH = './ptb-xl/' 
    MIT_NPY_PATH = './outputs/mit_teacher_labeled.npy'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n📦 正在加载整合数据集 (PTB-XL + MIT-BIH, 单导联 Lead II)...")
    train_loader = DataLoader(Integrated_ECG_Dataset(PTBXL_PATH, MIT_NPY_PATH, 'train'), batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(Integrated_ECG_Dataset(PTBXL_PATH, MIT_NPY_PATH, 'val'), batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(Integrated_ECG_Dataset(PTBXL_PATH, MIT_NPY_PATH, 'test'), batch_size=64, shuffle=False, num_workers=0)

    # ==========================================
    # 5. 注册论文要求对比的 7 大基线模型
    # ==========================================
    # 如果出现参数报错，说明你 baselines 文件里的参数命名习惯不同，微调即可。
    models_to_test = {
        # 恢复为你最初正确的参数名：
        "FCN": FCN(in_channels=1, out_channels=12),
        "InceptionTime1D": Inception1d(in_channels=1, num_classes=12),
        
        # 下面这些保留 c_in 和 c_out (根据你 batch2 和 batch3 的代码)：
        "xresnet1d101": xresnet1d101(c_in=1, c_out=12),
        "BiLSTM": Standalone_BiLSTM(c_in=1, c_out=12),
        "CRNN_Hybrid": CRNN(c_in=1, c_out=12),
        "XceptionTime": XceptionTime(c_in=1, c_out=12),
        #"Vanilla_Transformer": Standard_1D_Transformer(c_in=1, c_out=12)
    }

    results = []
    
    # 自动化遍历跑分
    for name, model in models_to_test.items():
        metrics = train_and_evaluate(name, model, train_loader, val_loader, test_loader, DEVICE)
        results.append(metrics)
        
    # ==========================================
    # 6. 打印最终的论文对比表格
    # ==========================================
    print("\n" + "★"*85)
    print(" 📊 论文第二章：整合数据集 (PTB-XL + MIT) 基线模型终极对比成绩单")
    print("★"*85)
    print(f"{'Model Name':<20} | {'Mac-AUC':<8} | {'Mac-SEN':<8} | {'Mac-SPE':<8} | {'Mac-F1':<8} | {'Mic-F1':<8}")
    print("-" * 85)
    for res in results:
        print(f"{res['Model']:<20} | {res['Macro-AUC']:>7.2f}% | {res['Macro-SEN']:>7.2f}% | {res['Macro-SPE']:>7.2f}% | {res['Macro-F1']:>7.2f}% | {res['Micro-F1']:>7.2f}%")
    print("★"*85)
    
    # 顺手帮你保存成 CSV，方便贴进论文
    os.makedirs('./outputs', exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv('./outputs/chapter2_integrated_dataset_results.csv', index=False)
    print(f"✅ 结果已保存至: ./outputs/chapter2_integrated_dataset_results.csv")

if __name__ == '__main__':
    main()