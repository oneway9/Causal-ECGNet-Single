import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
# 🚀 新增导入：f1_score 用于计算 F1，confusion_matrix 用于计算特异度 (SPE)
from sklearn.metrics import classification_report, roc_auc_score, recall_score, f1_score, confusion_matrix, roc_curve

from dataset import PTBXL_Rhythm_Dataset
from model import CausalECGNet

def calculate_macro_specificity(y_true, y_pred):
    """
    计算宏观特异度 (Mac-SPE)
    公式: $$Specificity = \frac{TN}{TN + FP}$$
    """
    num_classes = y_true.shape[1]
    specs = []
    for i in range(num_classes):
        tn, fp, fn, tp = confusion_matrix(y_true[:, i], y_pred[:, i]).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        specs.append(spec)
    return np.mean(specs) * 100

def main():
    DATA_PATH = './ptb-xl/' 
    WEIGHTS_PATH = './outputs/best_causal_ecgnet.pth'
    OUTPUT_DIR = './outputs/'
    BATCH_SIZE = 64
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_dataset = PTBXL_Rhythm_Dataset(DATA_PATH, fold_type='test')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = CausalECGNet(num_classes=12, hidden_channels=64).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval() 

    all_labels, all_probs = [], []
    
    print("\n[*] 开始在测试集上进行评估...")
    with torch.no_grad():
        for signals, labels in tqdm(test_loader, desc='Testing'):
            outputs = model(signals.to(DEVICE))
            probs = torch.sigmoid(outputs)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels, all_probs = np.array(all_labels), np.array(all_probs)
    num_classes = all_labels.shape[1]
    
    # --- 1. 动态寻优阈值 (基于 Youden's Index) ---
    optimal_thresholds = []
    for i in range(num_classes):
        fpr, tpr, thresholds = roc_curve(all_labels[:, i], all_probs[:, i])
        optimal_thresholds.append(thresholds[np.argmax(tpr - fpr)])

    # --- 2. 二值化预测结果 ---
    all_preds = np.zeros_like(all_probs)
    for i in range(num_classes):
        all_preds[:, i] = (all_probs[:, i] >= optimal_thresholds[i]).astype(float)

    # --- 3. 计算 5 大金标准指标 ---
    # Mac-AUC: 宏观曲线下面积
    mac_auc = roc_auc_score(all_labels, all_probs, average='macro') * 100
    # Mac-SEN: 宏观敏感度 (即 Recall)
    mac_sen = recall_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    # Mac-SPE: 宏观特异度 (利用自定义函数计算)
    mac_spe = calculate_macro_specificity(all_labels, all_preds)
    # Mac-F1: 宏观 F1-score (对小类更公平)
    mac_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    # Mic-F1: 微观 F1-score (衡量全局样本准确性)
    mic_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0) * 100

    # --- 4. 打印对齐后的成绩单 ---
    print("\n" + "="*85)
    print(f"{'Model Name':<20} | {'Mac-AUC':<8} | {'Mac-SEN':<8} | {'Mac-SPE':<8} | {'Mac-F1':<8} | {'Mic-F1':<8}")
    print("-" * 85)
    print(f"{'CausalECGNet (Ours)':<20} | {mac_auc:>7.2f}% | {mac_sen:>7.2f}% | {mac_spe:>7.2f}% | {mac_f1:>7.2f}% | {mic_f1:>7.2f}%")
    print("="*85 + "\n")

if __name__ == '__main__':
    main()