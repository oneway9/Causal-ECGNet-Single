import os
import ast
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import wfdb
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.signal import butter, filtfilt
from sklearn.metrics import roc_auc_score

# ==========================================
# 【核心修改】只导入咱们刚刚写好的新架构模型
# ==========================================
from model_inception_causal import InceptionCausalECGNet

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ==========================================
# 1. 混合数据集 (PTB-XL Train + MIT Pseudo)
# ==========================================
class Mixed_SingleLead_Dataset(Dataset):
    def __init__(self, ptbxl_path, mit_npy_path=None, fold_type='train'):
        super().__init__()
        self.samples = []
        self.labels = []
        
        # --- 加载 PTB-XL 数据 ---
        df = pd.read_csv(os.path.join(ptbxl_path, 'ptbxl_database.csv'), index_col='ecg_id')
        df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))
        
        agg_df = pd.read_csv(os.path.join(ptbxl_path, 'scp_statements.csv'), index_col=0)
        agg_df = agg_df[agg_df.rhythm == 1]
        self.rhythm_classes = agg_df.index.tolist()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(self.rhythm_classes)}
        
        if fold_type == 'train':
            df_target = df[df.strat_fold <= 8]
        elif fold_type == 'val':
            df_target = df[df.strat_fold == 9]
            
        nyq = 0.5 * 500.0
        self.b, self.a = butter(3, [0.5 / nyq, 40.0 / nyq], btype='bandpass')
        self.ptbxl_path = ptbxl_path
        
        print(f"[*] 正在加载 PTB-XL ({fold_type}) 数据...")
        for idx, row in tqdm(df_target.iterrows(), total=len(df_target), desc=f"Loading PTBXL {fold_type}", leave=False):
            rhythm_labels = [code for code in row['scp_codes'].keys() if code in self.rhythm_classes]
            if len(rhythm_labels) > 0:
                self.samples.append({'source': 'ptbxl', 'path': row['filename_hr']}) 
                label_vector = np.zeros(len(self.rhythm_classes), dtype=np.float32)
                for code in rhythm_labels:
                    label_vector[class_to_idx[code]] = 1.0 # 绝对硬标签
                self.labels.append(label_vector)
        
        # --- 如果是训练集，注入 MIT 软标签数据 ---
        if fold_type == 'train' and mit_npy_path is not None:
            print(f"[*] 正在注入 MIT-BIH 外部软标签数据...")
            mit_data = np.load(mit_npy_path, allow_pickle=True)
            for item in mit_data:
                self.samples.append({'source': 'mit', 'signal': item['signal']})
                self.labels.append(item['pseudo_label'].astype(np.float32)) # 软标签
                
        print(f"[*] {fold_type} 集准备完毕! 总样本数: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        label = self.labels[idx]
        
        if item['source'] == 'ptbxl':
            record_path = os.path.join(self.ptbxl_path, item['path'])
            signal, _ = wfdb.rdsamp(record_path)
            sig_filtered = filtfilt(self.b, self.a, signal[:, 1]) # 取第 2 导联
        else:
            sig_filtered = item['signal']
            
        sig_tensor = torch.tensor(sig_filtered.copy(), dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return sig_tensor, label_tensor

# ==========================================
# 2. 辅助函数
# ==========================================
def safe_macro_auc(y_true, y_score):
    aucs = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) == 2:
            aucs.append(roc_auc_score(y_true[:, i], y_score[:, i]))
    if len(aucs) == 0:
        return 0.0
    return np.mean(aucs) * 100

def plot_and_save_curves(history, output_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    plt.plot(epochs, history['val_loss'], label='Val Loss', marker='o')
    plt.title('Mixed Inception-Causal Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc', marker='o')
    plt.plot(epochs, history['val_acc'], label='Val Acc', marker='o')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['val_auc'], label='Val AUC', marker='o', color='green')
    plt.title('Validation Macro-AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'inception_causal_mixed_curves.png')
    plt.savefig(save_path, dpi=300)
    plt.close()

# ==========================================
# 3. 混合训练主循环
# ==========================================
def main():
    PTBXL_PATH = './ptb-xl/' 
    MIT_NPY_PATH = './outputs/mit_teacher_labeled.npy'
    OUTPUT_DIR = './outputs/' 
    
    BATCH_SIZE = 32 # 保守起见设为 32，因为 Inception 分支较多
    EPOCHS = 30
    LR = 0.001
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] 当前使用设备: {DEVICE}")

    train_dataset = Mixed_SingleLead_Dataset(PTBXL_PATH, MIT_NPY_PATH, fold_type='train')
    val_dataset = Mixed_SingleLead_Dataset(PTBXL_PATH, fold_type='val')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 实例化全新的融合架构
    model = InceptionCausalECGNet(num_classes=12, hidden_channels=64).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.BCEWithLogitsLoss()

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_auc': []}
    best_val_auc = 0.0

    print("\n[*] 开始训练全新的 Inception-Causal ECGNet (Mixed Data)...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        train_loop = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{EPOCHS}] [Train]', leave=False)
        for signals, labels in train_loop:
            signals, labels = signals.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            pred_upper, pred_lower = model(signals, labels, current_epoch=epoch)
            loss = criterion(pred_upper, labels) + criterion(pred_lower, labels)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            hard_labels = (labels > 0.5).float()
            probs = torch.sigmoid((pred_upper + pred_lower) / 2)
            predicted = (probs > 0.5).float()
            
            train_total += hard_labels.numel() 
            train_correct += (predicted == hard_labels).sum().item()
            train_loop.set_postfix(loss=f"{loss.item():.4f}")
            
        scheduler.step()
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100 * train_correct / train_total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_val_probs, all_val_hard_labels = [], []
        
        val_loop = tqdm(val_loader, desc=f'Epoch [{epoch+1}/{EPOCHS}] [Val  ]', leave=False)
        with torch.no_grad():
            for signals, labels in val_loop:
                signals, labels = signals.to(DEVICE), labels.to(DEVICE)
                pred = model(signals) 
                
                val_loss += criterion(pred, labels).item()
                probs = torch.sigmoid(pred)
                
                all_val_probs.extend(probs.cpu().numpy())
                all_val_hard_labels.extend(labels.cpu().numpy())
                
                predicted = (probs > 0.5).float()
                val_total += labels.numel()
                val_correct += (predicted == labels).sum().item()
                
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100 * val_correct / val_total
        
        all_val_hard_labels = np.array(all_val_hard_labels)
        all_val_probs = np.array(all_val_probs)
        epoch_val_auc = safe_macro_auc(all_val_hard_labels, all_val_probs)

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['val_auc'].append(epoch_val_auc)
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {epoch_train_loss:.4f} | PTBXL Val Loss: {epoch_val_loss:.4f} | PTBXL Val AUC: {epoch_val_auc:.2f}%")
        
        if epoch_val_auc > best_val_auc:
            best_val_auc = epoch_val_auc
            # 【核心修改】单独命名保存的权重文件，绝对不覆盖原来的老模型
            save_path = os.path.join(OUTPUT_DIR, 'best_inception_causal_ecgnet.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  --> [Saved] 发现新的最佳混合模型! 纯净验证集 AUC 提升至: {best_val_auc:.2f}%")

    print("\n[*] 联合训练完成！正在生成图表...")
    plot_and_save_curves(history, OUTPUT_DIR)

if __name__ == '__main__':
    main()