import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

# ==========================================
# 1. 导入修复过防泄露的公共组件
# ==========================================
from chapter2_combined_improvements import (
    PTBXL_Lead2_Dataset, 
    MIT_Lead2_Dataset, 
    AsymmetricLoss, 
    find_optimal_thresholds, 
    calculate_metrics_with_custom_thresholds
)

# ==========================================
# 2. 导入所有基线模型 (全部设置为单通道 c_in=1)
# ==========================================
from baselines.basic_conv1d import FCN
from baselines.bilstm import Standalone_BiLSTM
from baselines.crnn import CRNN
from baselines.inception1d import Inception1d
from baselines.stanford_resnet import StanfordResNet34
from baselines.xception_time import XceptionTime
from baselines.xresnet1d import xresnet1d101

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_integrated_loaders():
    """构建 PTB-XL + MIT-BIH 整合数据集的 DataLoader"""
    PTBXL_PATH = './ptb-xl/'
    MIT_NPY_PATH = './outputs/mit_teacher_labeled.npy'
    BATCH_SIZE = 32

    print("\n📦 正在构建多源整合数据集 (PTB-XL + MIT-BIH)...")
    
    # 训练集拼接
    train_ptb = PTBXL_Lead2_Dataset(PTBXL_PATH, fold_type='train')
    train_mit = MIT_Lead2_Dataset(MIT_NPY_PATH, fold_type='train')
    train_ds = ConcatDataset([train_ptb, train_mit])
    
    # 验证集拼接
    val_ptb = PTBXL_Lead2_Dataset(PTBXL_PATH, fold_type='val')
    val_mit = MIT_Lead2_Dataset(MIT_NPY_PATH, fold_type='val')
    val_ds = ConcatDataset([val_ptb, val_mit])
    
    # 测试集拼接
    test_ptb = PTBXL_Lead2_Dataset(PTBXL_PATH, fold_type='test')
    test_mit = MIT_Lead2_Dataset(MIT_NPY_PATH, fold_type='test')
    test_ds = ConcatDataset([test_ptb, test_mit])

    print(f"[*] 整合完成！")
    print(f"    - Train 总量: {len(train_ds)}")
    print(f"    - Val   总量: {len(val_ds)}")
    print(f"    - Test  总量: {len(test_ds)}")

    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
        DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=2),
        DataLoader(test_ds, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=2)
    )

def main():
    print(f"\n🚀 [启动] 论文第二章：整合数据集 (PTB-XL + MIT-BIH) 基线全面测评")
    print(f"[*] 计算设备: {DEVICE}")

    train_loader, val_loader, test_loader = get_integrated_loaders()

    # 注册你要对比的模型
    models_to_test = {
        "FCN_Baseline": FCN(in_channels=1, out_channels=12), # 某些代码可能是 in_channels
        "BiLSTM": Standalone_BiLSTM(c_in=1, c_out=12),
        "CRNN": CRNN(c_in=1, c_out=12),
        "Vanilla_Inception1D": Inception1d(in_channels=1, num_classes=12),
        "XceptionTime": XceptionTime(c_in=1, c_out=12),
        "Stanford_ResNet": StanfordResNet34(c_in=1, c_out=12),
        "xResNet101_1D": xresnet1d101(c_in=1, c_out=12)
    }

    results = []
    os.makedirs('./outputs/integrated_checkpoints', exist_ok=True)

    for model_name, model in models_to_test.items():
        print(f"\n" + "="*80)
        print(f" 🏃 正在训练模型: {model_name} (数据源: Integrated)")
        print("="*80)
        
        model = model.to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
        # 统一使用带有抗长尾分布特性的损失函数
        criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05).to(DEVICE)
        
        best_val_auc = 0.0
        save_path = f'./outputs/integrated_checkpoints/{model_name}_best.pth'
        
        # --- A. 训练循环 (30 Epochs) ---
        for epoch in range(30):
            model.train()
            for signals, labels in tqdm(train_loader, desc=f'[{model_name}] Epoch {epoch+1}/30', leave=False):
                signals, labels = signals.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(signals), labels)
                loss.backward()
                optimizer.step()
            scheduler.step()

            # --- B. 验证集评估 ---
            model.eval()
            val_probs, val_labels = [], []
            with torch.no_grad():
                for signals, labels in val_loader:
                    val_probs.extend(torch.sigmoid(model(signals.to(DEVICE))).cpu().numpy())
                    val_labels.extend(labels.numpy())
            
            # 使用 Macro-AUC 来保存最佳模型
            val_auc, _, _, _, _ = calculate_metrics_with_custom_thresholds(
                np.array(val_labels), np.array(val_probs), np.full(12, 0.5)
            )
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), save_path)

        # --- C. 终极测试 (动态阈值寻优 + 盲测) ---
        print(f"\n[*] {model_name} 训练完毕，正在加载最佳权重执行盲测...")
        model.load_state_dict(torch.load(save_path, map_location=DEVICE))
        model.eval()
        
        v_p, v_l, t_p, t_l = [], [], [], []
        with torch.no_grad():
            for s, l in val_loader:
                v_p.extend(torch.sigmoid(model(s.to(DEVICE))).cpu().numpy()); v_l.extend(l.numpy())
            # 在验证集寻找对 F1 最友好的阈值
            opt_thresholds = find_optimal_thresholds(np.array(v_l), np.array(v_p))
            
            for s, l in test_loader:
                t_p.extend(torch.sigmoid(model(s.to(DEVICE))).cpu().numpy()); t_l.extend(l.numpy())
        
        # 计算核心金标准指标
        mac_auc, mac_sen, mac_spe, mac_f1, mic_f1 = calculate_metrics_with_custom_thresholds(
            np.array(t_l), np.array(t_p), opt_thresholds
        )
        
        print(f" -> 结果: AUC={mac_auc:.2f}% | SEN={mac_sen:.2f}% | F1={mac_f1:.2f}%")
        
        results.append({
            "Model": model_name,
            "Mac-AUC": mac_auc,
            "Mac-SEN": mac_sen,
            "Mac-SPE": mac_spe,
            "Mac-F1": mac_f1,
            "Mic-F1": mic_f1
        })

    # ==========================================
    # 3. 打印论文最终对比表格并保存 CSV
    # ==========================================
    print("\n" + "★"*85)
    print(" 📊 论文第二章：整合数据集 (PTB-XL + MIT-BIH) 基线模型终极对比成绩单")
    print("★"*85)
    print(f"{'Model Name':<20} | {'Mac-AUC':<8} | {'Mac-SEN':<8} | {'Mac-SPE':<8} | {'Mac-F1':<8} | {'Mic-F1':<8}")
    print("-" * 85)
    for res in results:
        print(f"{res['Model']:<20} | {res['Mac-AUC']:>7.2f}% | {res['Mac-SEN']:>7.2f}% | {res['Mac-SPE']:>7.2f}% | {res['Mac-F1']:>7.2f}% | {res['Mic-F1']:>7.2f}%")
    print("★"*85)
    
    df = pd.DataFrame(results)
    csv_path = "./outputs/chapter2_integrated_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[Success] 成绩单已导出至: {csv_path}")

if __name__ == '__main__':
    main()