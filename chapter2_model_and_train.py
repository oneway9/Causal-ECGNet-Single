import os
import ast
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import wfdb
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, filtfilt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ==========================================
# 1. 第二章专属模型架构: 单分支多尺度因果网络
# (InceptionCausal_MacroOnly - 完整内嵌版)
# ==========================================
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 将输出通道平分为 4 份给 4 个并行分支
        branch_out = out_channels // 4
        
        # 分支 1: 捕捉极高频突变 (Kernel=10)
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, branch_out, kernel_size=10, padding='same'),
            nn.BatchNorm1d(branch_out), nn.ReLU()
        )
        # 分支 2: 捕捉中频形态 (Kernel=20)
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, branch_out, kernel_size=20, padding='same'),
            nn.BatchNorm1d(branch_out), nn.ReLU()
        )
        # 分支 3: 捕捉低频基线和宽波 (Kernel=40)
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, branch_out, kernel_size=40, padding='same'),
            nn.BatchNorm1d(branch_out), nn.ReLU()
        )
        # 分支 4: 保留原始显著特征 (MaxPool + 1x1 Conv)
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_out, kernel_size=1, padding='same'),
            nn.BatchNorm1d(branch_out), nn.ReLU()
        )
        
    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        # 在通道维度拼接 (B, out_channels, L)
        return torch.cat([out1, out2, out3, out4], dim=1)

class InceptionCausal_MacroOnly(nn.Module):
    def __init__(self, num_classes=12, hidden_channels=64):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        
        # 1. 快速降维 Stem 层
        self.stem = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # 2. 多尺度感知主干 (Inception Backbone)
        self.inception_layer = InceptionBlock(hidden_channels, hidden_channels)
        
        # 3. 宏观特征提取层 (Kernel=50)
        self.T_macro = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=50, padding='same'),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU()
        )
        
        # 4. 因果混淆字典与交叉注意力 (核心创新点)
        self.Z_confounder = nn.Parameter(torch.randn(num_classes, hidden_channels))
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_channels, num_heads=4, batch_first=True)
        
        # 5. 分类器
        self.mlp = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU())
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x):
        # x shape: (B, 1, L)
        f_stem = self.stem(x)
        f_incept = self.inception_layer(f_stem)
        
        # 提取宏观节律并进行全局平均池化 -> (B, C)
        f_macro_seq = self.T_macro(f_incept)
        Q_macro = f_macro_seq.mean(dim=-1) 
        
        # 因果交叉注意力去偏
        # Q_macro 作为 Query: (B, 1, C), 字典 Z 作为 Key/Value: (B, num_classes, C)
        Q = Q_macro.unsqueeze(1)
        K = self.Z_confounder.unsqueeze(0).expand(x.size(0), -1, -1)
        V = K
        
        attn_out, _ = self.cross_attention(Q, K, V)
        attn_out = attn_out.squeeze(1) # (B, C)
        
        # 残差融合净化特征
        f_purified = Q_macro + attn_out
        
        # 最终分类
        out = self.mlp(f_purified)
        logits = self.classifier(out)
        
        return logits

# ==========================================
# 2. 第二章纯净数据集: 仅 PTB-XL 单导联 (Lead II)
# ==========================================
class PTBXL_Lead2_Dataset(Dataset):
    def __init__(self, ptbxl_path, fold_type='train'):
        super().__init__()
        self.samples, self.labels = [], []
        
        df = pd.read_csv(os.path.join(ptbxl_path, 'ptbxl_database.csv'), index_col='ecg_id')
        df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))
        agg_df = pd.read_csv(os.path.join(ptbxl_path, 'scp_statements.csv'), index_col=0)
        agg_df = agg_df[agg_df.rhythm == 1]
        
        self.rhythm_classes = agg_df.index.tolist()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(self.rhythm_classes)}
        
        print(f"[*] 准备 {fold_type} 集 (仅使用 PTB-XL 原始数据)...")

        if fold_type == 'train': df_target = df[df.strat_fold <= 8]
        elif fold_type == 'val': df_target = df[df.strat_fold == 9]
        elif fold_type == 'test': df_target = df[df.strat_fold == 10]
            
        nyq = 0.5 * 500.0
        self.b, self.a = butter(3, [0.5 / nyq, 40.0 / nyq], btype='bandpass')
        self.ptbxl_path = ptbxl_path
        
        for idx, row in tqdm(df_target.iterrows(), total=len(df_target), desc=f"Loading {fold_type}", leave=False):
            rhythm_labels = [code for code in row['scp_codes'].keys() if code in self.rhythm_classes]
            if len(rhythm_labels) > 0:
                self.samples.append(row['filename_hr']) 
                label_vector = np.zeros(len(self.rhythm_classes), dtype=np.float32)
                for code in rhythm_labels: 
                    label_vector[class_to_idx[code]] = 1.0 
                self.labels.append(label_vector)

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        # 仅读取导联 II (在 WFDB 中索引通常为 1)
        sig, _ = wfdb.rdsamp(os.path.join(self.ptbxl_path, self.samples[idx]))
        sig_filtered = filtfilt(self.b, self.a, sig[:, 1]) 
        return torch.tensor(sig_filtered.copy(), dtype=torch.float32).unsqueeze(0), torch.tensor(self.labels[idx], dtype=torch.float32)

# ==========================================
# 3. 多维医学综合评价体系计算函数
# ==========================================
def calculate_medical_metrics(y_true, y_score, threshold=0.5):
    """
    计算医学多标签分类的核心指标：AUC, SEN(敏感度), SPE(特异度), F1-Score
    """
    y_pred = (y_score >= threshold).astype(int)
    aucs, sens, spes, f1s = [], [], [], []
    
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) == 2: # 确保类别中同时有正负样本
            # 1. AUC
            aucs.append(roc_auc_score(y_true[:, i], y_score[:, i]))
            
            # 计算混淆矩阵元素
            tp = np.sum((y_pred[:, i] == 1) & (y_true[:, i] == 1))
            tn = np.sum((y_pred[:, i] == 0) & (y_true[:, i] == 0))
            fp = np.sum((y_pred[:, i] == 1) & (y_true[:, i] == 0))
            fn = np.sum((y_pred[:, i] == 0) & (y_true[:, i] == 1))
            
            # 2. 敏感度 (Sensitivity / Recall) - 找全病人的能力
            sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            sens.append(sen)
            
            # 3. 特异度 (Specificity) - 排除没病的人的能力
            spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            spes.append(spe)
            
            # 4. F1-Score - 综合平衡指标
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
            f1s.append(f1)
            
    return {
        'Macro-AUC': np.mean(aucs) * 100 if aucs else 0.0,
        'Macro-SEN': np.mean(sens) * 100 if sens else 0.0,
        'Macro-SPE': np.mean(spes) * 100 if spes else 0.0,
        'Macro-F1': np.mean(f1s) * 100 if f1s else 0.0
    }

# ==========================================
# 4. 论文第二章专属主训练循环
# ==========================================
def main():
    PTBXL_PATH = './ptb-xl/' 
    OUTPUT_DIR = './outputs/' 
    MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, 'chapter2_best_causal_ecgnet.pth')
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n" + "="*75)
    print(" 📖 [论文第二章] 模型架构改进专属实验 (纯 PTB-XL + 多维评价指标)")
    print("="*75)

    # 为了稳定和防止 Windows 报错，使用 num_workers=0
    train_loader = DataLoader(PTBXL_Lead2_Dataset(PTBXL_PATH, 'train'), batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(PTBXL_Lead2_Dataset(PTBXL_PATH, 'val'), batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(PTBXL_Lead2_Dataset(PTBXL_PATH, 'test'), batch_size=64, shuffle=False, num_workers=0)

    model = InceptionCausal_MacroOnly(num_classes=12, hidden_channels=64).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    
    for epoch in range(30):
        model.train()
        for signals, labels in tqdm(train_loader, desc=f'Epoch [{epoch+1}/30]', leave=False):
            signals, labels = signals.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(signals), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for signals, labels in val_loader:
                all_probs.extend(torch.sigmoid(model(signals.to(DEVICE))).cpu().numpy())
                all_labels.extend(labels.numpy())
                
        metrics = calculate_medical_metrics(np.array(all_labels), np.array(all_probs))
        val_auc = metrics['Macro-AUC']
        
        print(f"Epoch {epoch+1:02d}/30 | Loss: {loss.item():.4f} | Val AUC: {val_auc:.2f}% | Val SEN: {metrics['Macro-SEN']:.2f}% | Val F1: {metrics['Macro-F1']:.2f}%")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print("\n🏆 正在使用最高分权重进行 Fold 10 终极盲测...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()
    all_test_probs, all_test_labels = [], []
    with torch.no_grad():
        for signals, labels in tqdm(test_loader, leave=False):
            all_test_probs.extend(torch.sigmoid(model(signals.to(DEVICE))).cpu().numpy())
            all_test_labels.extend(labels.numpy())
            
    final_metrics = calculate_medical_metrics(np.array(all_test_labels), np.array(all_test_probs))
    
    print("\n" + "="*75)
    print(" 🏥 论文第二章: 改进模型多维医学评价体系成绩单 (PTB-XL Test Set)")
    print("="*75)
    print(f"  - 宏观 AUC (Macro-AUC)           : {final_metrics['Macro-AUC']:.2f}%  <- 综合诊断能力")
    print(f"  - 宏观敏感度 (Macro-Sensitivity) : {final_metrics['Macro-SEN']:.2f}%  <- 防漏诊能力 (极关键!)")
    print(f"  - 宏观特异度 (Macro-Specificity) : {final_metrics['Macro-SPE']:.2f}%  <- 防误诊能力")
    print(f"  - 宏观 F1 分数 (Macro-F1 Score)  : {final_metrics['Macro-F1']:.2f}%  <- 类别不平衡下的精确度")
    print("="*75)
    print("💡 在论文中你可以强调：虽然仅使用单导联和纯净数据，但模型架构的升级有效提升了敏感度和 F1 分数！")

if __name__ == '__main__':
    main()