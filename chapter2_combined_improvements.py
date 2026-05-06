import os
import ast
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import wfdb
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, filtfilt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, recall_score, f1_score, confusion_matrix, roc_curve

# ==========================================
# 1. 核心损失函数：Asymmetric Loss (ASL)
# ==========================================
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        xs_pos = torch.sigmoid(x)
        xs_neg = 1 - xs_pos

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        los_pos = y * torch.log(xs_pos.clamp(min=self.eps)) * (1 - xs_pos).pow(self.gamma_pos)
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps)) * (1 - xs_neg).pow(self.gamma_neg)
        loss = los_pos + los_neg
        return -loss.sum()

# ==========================================
# 2. 基础数据集类：PTB-XL 单导联 (Lead II)
# ==========================================
class PTBXL_Lead2_Dataset(Dataset):
    def __init__(self, data_path='./ptb-xl/', fold_type='train'):
        super().__init__()
        self.data_path = data_path
        df = pd.read_csv(os.path.join(data_path, 'ptbxl_database.csv'), index_col='ecg_id')
        df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))
        
        agg_df = pd.read_csv(os.path.join(data_path, 'scp_statements.csv'), index_col=0)
        agg_df = agg_df[agg_df.rhythm == 1]
        self.rhythm_classes = agg_df.index.tolist()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.rhythm_classes)}
        
        if fold_type == 'train': self.df = df[df.strat_fold <= 8]
        elif fold_type == 'val': self.df = df[df.strat_fold == 9]
        elif fold_type == 'test': self.df = df[df.strat_fold == 10]
            
        self.samples, self.labels = [], []
        for idx, row in self.df.iterrows():
            rhythm_labels = [code for code in row['scp_codes'].keys() if code in self.rhythm_classes]
            if len(rhythm_labels) > 0:
                self.samples.append(row['filename_hr']) 
                label_vector = np.zeros(len(self.rhythm_classes), dtype=np.float32)
                for code in rhythm_labels:
                    label_vector[self.class_to_idx[code]] = 1.0
                self.labels.append(label_vector)
        
        nyq = 0.5 * 500.0
        self.b, self.a = butter(3, [0.5 / nyq, 40.0 / nyq], btype='bandpass')

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        record_path = os.path.join(self.data_path, self.samples[idx])
        signal, _ = wfdb.rdsamp(record_path)
        sig_lead2 = filtfilt(self.b, self.a, signal[:, 1]) 
        return torch.tensor(sig_lead2.copy(), dtype=torch.float32).unsqueeze(0), \
               torch.tensor(self.labels[idx], dtype=torch.float32)

# ==========================================
# 3. 终极整合数据集类：自适应阈值过滤版本 (PTB + MIT)
# ==========================================
class Integrated_ECG_Dataset(Dataset):
    def __init__(self, ptbxl_path='./ptb-xl/', mit_npy_path='./outputs/mit_teacher_labeled.npy', fold_type='train'):
        super().__init__()
        self.samples, self.labels = [], []
        self.ptbxl_path = ptbxl_path
        
        # 加载分类体系
        agg_df = pd.read_csv(os.path.join(ptbxl_path, 'scp_statements.csv'), index_col=0)
        agg_df = agg_df[agg_df.rhythm == 1]
        self.rhythm_classes = agg_df.index.tolist()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(self.rhythm_classes)}
        
        # 1. 加载 PTB-XL 部分
        df = pd.read_csv(os.path.join(ptbxl_path, 'ptbxl_database.csv'), index_col='ecg_id')
        df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))
        if fold_type == 'train': df_target = df[df.strat_fold <= 8]
        elif fold_type == 'val': df_target = df[df.strat_fold == 9]
        else: df_target = df[df.strat_fold == 10]

        for _, row in df_target.iterrows():
            rhythm_labels = [code for code in row['scp_codes'].keys() if code in self.rhythm_classes]
            if len(rhythm_labels) > 0:
                self.samples.append({'source': 'ptbxl', 'path': row['filename_hr']})
                label_vector = np.zeros(len(self.rhythm_classes), dtype=np.float32)
                for code in rhythm_labels: label_vector[class_to_idx[code]] = 1.0
                self.labels.append(label_vector)

        # 2. 整合 MIT-BIH 部分 (仅在训练阶段启用自适应过滤)
        if fold_type == 'train' and os.path.exists(mit_npy_path):
            mit_data = np.load(mit_npy_path, allow_pickle=True)
            # 自适应阶梯阈值逻辑
            thresholds = np.full(len(self.rhythm_classes), 0.30) # 默认普通病阈值
            thresholds[class_to_idx['SR']] = 1.10 # 屏蔽 SR
            for rare_cls in ['BIGU', 'AFLT', 'SVTAC', 'PSVT', 'TRIGU']:
                if rare_cls in class_to_idx: thresholds[class_to_idx[rare_cls]] = 0.10 # 罕见病放宽
            
            for item in mit_data:
                probs = item['pseudo_label'].astype(np.float32)
                if np.any(probs > thresholds):
                    self.samples.append({'source': 'mit', 'signal': item['signal']})
                    self.labels.append(probs) # 训练时使用软标签

        nyq = 0.5 * 500.0
        self.b, self.a = butter(3, [0.5 / nyq, 40.0 / nyq], btype='bandpass')
        print(f"[*] Integrated Dataset ({fold_type}): Loaded {len(self.samples)} samples.")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        item = self.samples[idx]
        if item['source'] == 'ptbxl':
            sig, _ = wfdb.rdsamp(os.path.join(self.ptbxl_path, item['path']))
            sig_f = filtfilt(self.b, self.a, sig[:, 1])
        else:
            sig_f = item['signal']
        
        # 验证/测试阶段如果是软标签，需二值化以进行指标计算
        label = self.labels[idx]
        if not np.all((label == 0) | (label == 1)): # 检测是否为软标签
            label = (label > 0.5).astype(np.float32)
            
        return torch.tensor(sig_f.copy(), dtype=torch.float32).unsqueeze(0), \
               torch.tensor(label, dtype=torch.float32)

# ==========================================
# 4. 评价指标工具函数
# ==========================================
def find_optimal_thresholds(y_true, y_probs):
    thresholds_opt = []
    for i in range(y_true.shape[1]):
        try:
            fpr, tpr, thresholds = roc_curve(y_true[:, i], y_probs[:, i])
            thresholds_opt.append(thresholds[np.argmax(tpr - fpr)])
        except:
            thresholds_opt.append(0.5)
    return np.array(thresholds_opt)

def calculate_metrics_with_custom_thresholds(y_true, y_probs, thresholds):
    y_pred = (y_probs >= thresholds).astype(int)
    m_auc = roc_auc_score(y_true, y_probs, average='macro') * 100
    m_sen = recall_score(y_true, y_pred, average='macro', zero_division=0) * 100
    m_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100
    mi_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0) * 100
    
    specs = []
    for i in range(y_true.shape[1]):
        tn, fp, fn, tp = confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1]).ravel()
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    m_spe = np.mean(specs) * 100
    
    return m_auc, m_sen, m_spe, m_f1, mi_f1