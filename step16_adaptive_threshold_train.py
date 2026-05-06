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
from sklearn.metrics import f1_score, recall_score
# 假设你的寻优函数在 chapter2_combined_improvements.py 中
from chapter2_combined_improvements import find_optimal_thresholds, calculate_metrics_with_custom_thresholds

# 导入咱们的最强单分支架构
from models_ablation import InceptionCausal_MacroOnly

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ==========================================
# 1. 自适应阈值过滤数据集 (带内置分布统计)
# ==========================================
class Adaptive_Filtered_Dataset(Dataset):
    def __init__(self, ptbxl_path, mit_npy_path=None, fold_type='train'):
        super().__init__()
        self.samples, self.labels = [], []
        
        df = pd.read_csv(os.path.join(ptbxl_path, 'ptbxl_database.csv'), index_col='ecg_id')
        df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))
        agg_df = pd.read_csv(os.path.join(ptbxl_path, 'scp_statements.csv'), index_col=0)
        agg_df = agg_df[agg_df.rhythm == 1]
        
        self.rhythm_classes = agg_df.index.tolist()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(self.rhythm_classes)}
        
        # 【核心创新】：为每个类别定制专属入场阈值
        self.thresholds = np.zeros(len(self.rhythm_classes), dtype=np.float32)
        for cls_name, idx in class_to_idx.items():
            if cls_name == 'SR':
                self.thresholds[idx] = 1.10 # 绝对屏蔽 SR 的独立入场权
            elif cls_name in ['AFIB', 'STACH', 'SARRH', 'SBRAD']:
                self.thresholds[idx] = 0.30 #见病保持严苛
            elif cls_name in ['PACE', 'SVARR']:
                self.thresholds[idx] = 0.20 #频病变适度放宽
            else:
                # 极罕见病 ['BIGU', 'AFLT', 'SVTAC', 'PSVT', 'TRIGU']
                self.thresholds[idx] = 0.10# 极罕见病极限放宽！
                
        print(f"\n[*] 准备 {fold_type} 集，启用自适应阈值过滤 (Adaptive Thresholding)...")

        if fold_type == 'train': df_target = df[df.strat_fold <= 8]
        elif fold_type == 'val': df_target = df[df.strat_fold == 9]
        elif fold_type == 'test': df_target = df[df.strat_fold == 10]
            
        nyq = 0.5 * 500.0
        self.b, self.a = butter(3, [0.5 / nyq, 40.0 / nyq], btype='bandpass')
        self.ptbxl_path = ptbxl_path
        
        # --- 用于统计分布的计数器 ---
        ptbxl_counts = np.zeros(len(self.rhythm_classes), dtype=int)
        mit_counts = np.zeros(len(self.rhythm_classes), dtype=int)
        
        # 1. 完整加载 PTB-XL
        for idx, row in tqdm(df_target.iterrows(), total=len(df_target), desc=f"Loading PTBXL {fold_type}", leave=False):
            rhythm_labels = [code for code in row['scp_codes'].keys() if code in self.rhythm_classes]
            if len(rhythm_labels) > 0:
                self.samples.append({'source': 'ptbxl', 'path': row['filename_hr']}) 
                label_vector = np.zeros(len(self.rhythm_classes), dtype=np.float32)
                for code in rhythm_labels: 
                    label_vector[class_to_idx[code]] = 1.0 
                    ptbxl_counts[class_to_idx[code]] += 1 # 统计 PTB-XL 数量
                self.labels.append(label_vector)

        # 2. MIT-BIH 自适应软标签引入 (仅在 Train 阶段)
        if fold_type == 'train' and mit_npy_path is not None:
            mit_data = np.load(mit_npy_path, allow_pickle=True)
            mit_kept, mit_dropped = 0, 0
            
            for item in mit_data:
                probs_12d = item['pseudo_label'].astype(np.float32)
                
                # 【核心逻辑】：只要该样本在任意病理上的概率，超过了该病理专属的阈值，就放行！
                if np.any(probs_12d > self.thresholds):
                    self.samples.append({'source': 'mit', 'signal': item['signal']})
                    self.labels.append(probs_12d) 
                    mit_kept += 1
                    
                    # 统计 MIT 数量 (使用 > 0.5 作为硬标签统计阈值，保持与之前图表一致)
                    hard_labels = (probs_12d > 0.5).astype(int)
                    mit_counts += hard_labels
                else:
                    mit_dropped += 1
            print(f"    - MIT-BIH: 自适应策略保留了 {mit_kept} 个样本 (拦截了 {mit_dropped} 个冗余数据)。")
            
            # 【内置统计面板打印】
            print("\n" + "="*80)
            print(f" 📊 训练集 (Train Fold) 自适应增广数据分布统计面板")
            print("="*80)
            print(f"{'节律类别 (Class)':<15} | {'PTB-XL (基础)':<15} | {'MIT-BIH (自适应增量)':<20} | {'混合后总数':<15}")
            print("-" * 80)
            
            # 排序打印，让最多和最少的病理一目了然
            sorted_indices = np.argsort(ptbxl_counts)[::-1]
            total_ptb, total_mit = 0, 0
            
            for i in sorted_indices:
                cls_name = self.rhythm_classes[i]
                p_c = ptbxl_counts[i]
                m_c = mit_counts[i]
                total_ptb += p_c
                total_mit += m_c
                marker = " (*)" if cls_name == 'SR' else ""
                print(f"{cls_name+marker:<15} | {p_c:<15} | {m_c:<20} | {p_c + m_c:<15}")
                
            print("-" * 80)
            print(f"{'标签总人次':<15} | {total_ptb:<15} | {total_mit:<20} | {total_ptb + total_mit:<15}")
            print("="*80 + "\n")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        item = self.samples[idx]
        if item['source'] == 'ptbxl':
            sig, _ = wfdb.rdsamp(os.path.join(self.ptbxl_path, item['path']))
            sig_filtered = filtfilt(self.b, self.a, sig[:, 1]) 
        else:
            sig_filtered = item['signal']
        return torch.tensor(sig_filtered.copy(), dtype=torch.float32).unsqueeze(0), torch.tensor(self.labels[idx], dtype=torch.float32)

def safe_macro_auc(y_true, y_score):
    aucs = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) == 2:
            aucs.append(roc_auc_score(y_true[:, i], y_score[:, i]))
    return np.mean(aucs) * 100 if aucs else 0.0

def main():
    PTBXL_PATH = './ptb-xl/' 
    #!!!MIT_NPY_PATH = './outputs/mit_teacher_labeled.npy'
    MIT_NPY_PATH = None
    OUTPUT_DIR = './outputs/' 
    #!!!MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, 'best_adaptive_threshold_inception.pth')
    MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, 'best_ptb_only_macro_inception.pth')
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n" + "="*75)
    print(" 🎯 [Step 16] 终极数据工程：自适应阶梯阈值病理增广 (Adaptive Thresholding)")
    print("="*75)

    train_loader = DataLoader(Adaptive_Filtered_Dataset(PTBXL_PATH, MIT_NPY_PATH, 'train'), batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(Adaptive_Filtered_Dataset(PTBXL_PATH, fold_type='val'), batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(Adaptive_Filtered_Dataset(PTBXL_PATH, fold_type='test'), batch_size=64, shuffle=False, num_workers=0)

    # 12 分类的顶级宏观单分支模型
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
            loss = criterion(model(signals, labels, epoch), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for signals, labels in val_loader:
                all_probs.extend(torch.sigmoid(model(signals.to(DEVICE))).cpu().numpy())
                all_labels.extend(labels.numpy())
        val_auc = safe_macro_auc(np.array(all_labels), np.array(all_probs))
        
        print(f"Epoch {epoch+1:02d}/30 | Val AUC: {val_auc:.2f}%")
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print("\n🏆 正在进行 Fold 10 终极全指标审计...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()
    
    val_probs, val_labels = [], []
    test_probs, test_labels = [], []

    with torch.no_grad():
        # A. 在验证集寻找最佳阈值
        for signals, labels in val_loader:
            val_probs.extend(torch.sigmoid(model(signals.to(DEVICE))).cpu().numpy())
            val_labels.extend(labels.numpy())
        
        # 寻优
        opt_t = find_optimal_thresholds(np.array(val_labels), np.array(val_probs))
        
        # B. 在测试集应用阈值
        for signals, labels in tqdm(test_loader, desc="Testing", leave=False):
            test_probs.extend(torch.sigmoid(model(signals.to(DEVICE))).cpu().numpy())
            test_labels.extend(labels.numpy())
            
    test_labels = np.array(test_labels)
    test_probs = np.array(test_probs)

    # C. 调用全指标计算函数
    m_auc, m_sen, m_spe, m_f1, mi_f1 = calculate_metrics_with_custom_thresholds(
        test_labels, test_probs, opt_t
    )
    
    print("\n" + "★"*80)
    print(f" 📊 第二章基准实验成绩单：InceptionCausal (仅 PTB-XL 单源)")
    print("★"*80)
    print(f"{'指标 (Metrics)':<20} | {'得分 (Score)':<15}")
    print("-" * 40)
    print(f"{'Macro-AUC':<20} | {m_auc:>10.2f}%")
    print(f"{'Macro-SEN (Recall)':<20} | {m_sen:>10.2f}%")
    print(f"{'Macro-SPE':<20} | {m_spe:>10.2f}%")
    print(f"{'Macro-F1':<20} | {m_f1:>10.2f}%  <-- 重点对比项")
    print(f"{'Micro-F1':<20} | {mi_f1:>10.2f}%")
    print("★"*80 + "\n")

if __name__ == '__main__':
    main()