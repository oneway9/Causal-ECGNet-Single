import os
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from chapter2_combined_improvements import (
    PTBXL_Lead2_Dataset, MIT_Lead2_Dataset, AsymmetricLoss, find_optimal_thresholds, calculate_metrics_with_custom_thresholds
)

# 直接从你的 baselines 文件夹导入纯净版模型！绝对不会有依赖报错！
from baselines.bilstm import Standalone_BiLSTM
from baselines.xception_time import XceptionTime
from baselines.crnn import CRNN

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def train_and_evaluate(model_name, model, train_loader, val_loader, test_loader, device):
    print(f"\n" + "="*60)
    print(f" 🚀 正在评测第二批时序基线: {model_name}")
    print("="*60)
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05).to(device)
    
    best_val_auc = 0.0
    model_save_path = f'./outputs/baseline_{model_name}_best.pth'
    
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
            
    print(f"\n[*] {model_name} 训练完成，正在进行 F1 动态阈值校准及盲测...")
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
    
    return {"Model": model_name, "Macro-AUC": mac_auc, "Macro-SEN": mac_sen, "Macro-SPE": mac_spe, "Macro-F1": mac_f1, "Micro-F1": micro_f1}

def main():
    #PTBXL_PATH = './ptb-xl/' 
    MIT_PATH = './outputs/mit_teacher_labeled.npy'
    
    print("\n📦 正在加载 MIT-BIH (Lead II) 软标签数据集...")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #print("\n📦 正在加载 PTB-XL (Lead II) 数据集...")
    #train_loader = DataLoader(PTBXL_Lead2_Dataset(PTBXL_PATH, 'train'), batch_size=32, shuffle=True, num_workers=0)
    #val_loader = DataLoader(PTBXL_Lead2_Dataset(PTBXL_PATH, 'val'), batch_size=64, shuffle=False, num_workers=0)
    #test_loader = DataLoader(PTBXL_Lead2_Dataset(PTBXL_PATH, 'test'), batch_size=64, shuffle=False, num_workers=0)
    train_loader = DataLoader(MIT_Lead2_Dataset(MIT_PATH, 'train'), batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(MIT_Lead2_Dataset(MIT_PATH, 'val'), batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(MIT_Lead2_Dataset(MIT_PATH, 'test'), batch_size=64, shuffle=False, num_workers=0)

    # ==========================================
    # 3. 注册第二批对比模型
    # ==========================================
    models_to_test = {
        "BiLSTM_Standalone": Standalone_BiLSTM(c_in=1, c_out=12),
        "XceptionTime_1D": XceptionTime(c_in=1, c_out=12),
        "CRNN_Hybrid": CRNN(c_in=1, c_out=12)
    }

    results = []
    for name, model in models_to_test.items():
        metrics = train_and_evaluate(name, model, train_loader, val_loader, test_loader, DEVICE)
        results.append(metrics)
        
    print("\n" + "★"*80)
    print(" 📊 论文第二章：第二批次时序基线成绩单")
    print("★"*80)
    print(f"{'Model Name':<20} | {'Mac-AUC':<8} | {'Mac-SEN':<8} | {'Mac-SPE':<8} | {'Mac-F1':<8} | {'Mic-F1':<8}")
    print("-" * 80)
    for res in results:
        print(f"{res['Model']:<20} | {res['Macro-AUC']:.2f}%  | {res['Macro-SEN']:.2f}%  | {res['Macro-SPE']:.2f}%  | {res['Macro-F1']:.2f}%  | {res['Micro-F1']:.2f}%")
    print("★"*80)

if __name__ == '__main__':
    main()