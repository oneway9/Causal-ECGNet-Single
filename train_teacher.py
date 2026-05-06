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

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ==========================================
# 1. 专属数据加载器：仅提取 PTB-XL 的第二导联
# ==========================================
class PTBXL_SingleLead_Dataset(Dataset):
    def __init__(self, data_path, fold_type='train'):
        super().__init__()
        self.data_path = data_path
        
        df = pd.read_csv(os.path.join(data_path, 'ptbxl_database.csv'), index_col='ecg_id')
        df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))
        
        agg_df = pd.read_csv(os.path.join(data_path, 'scp_statements.csv'), index_col=0)
        agg_df = agg_df[agg_df.rhythm == 1]
        self.rhythm_classes = agg_df.index.tolist()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.rhythm_classes)}
        
        if fold_type == 'train':
            self.df = df[df.strat_fold <= 8]
        elif fold_type == 'val':
            self.df = df[df.strat_fold == 9]
            
        self.samples = []
        self.labels = []
        
        for idx, row in self.df.iterrows():
            rhythm_labels = [code for code in row['scp_codes'].keys() if code in self.rhythm_classes]
            if len(rhythm_labels) > 0:
                self.samples.append(row['filename_hr']) 
                label_vector = np.zeros(len(self.rhythm_classes), dtype=np.float32)
                for code in rhythm_labels:
                    label_vector[self.class_to_idx[code]] = 1.0
                self.labels.append(label_vector)
                
        print(f"[*] 成功加载 {len(self.samples)} 条 {fold_type} 集数据 (单导联模式).")
        
        nyq = 0.5 * 500.0
        self.b, self.a = butter(3, [0.5 / nyq, 40.0 / nyq], btype='bandpass')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        record_path = os.path.join(self.data_path, self.samples[idx])
        signal, _ = wfdb.rdsamp(record_path)
        
        # 【核心修改】只提取第 2 导联 (索引为 1)
        lead_ii = signal[:, 1]
        
        # 滤波去噪
        lead_ii_filtered = filtfilt(self.b, self.a, lead_ii)
        
        # 转换为 PyTorch 需要的格式：(channels, length) -> (1, 5000)
        sig_tensor = torch.tensor(lead_ii_filtered.copy(), dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return sig_tensor, label_tensor

# ==========================================
# 2. InceptionTime 网络架构
# ==========================================
class InceptionModule1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[10, 20, 40], bottleneck_channels=32):
        super().__init__()
        # 瓶颈层：如果输入通道数较多，先用 1x1 卷积降维，减少计算量
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False) if in_channels > 1 else nn.Identity()
        b_channels = bottleneck_channels if in_channels > 1 else in_channels
        
        # 三个并行的多尺度卷积分支
        self.convs = nn.ModuleList([
            nn.Conv1d(b_channels, out_channels, kernel_size=k, padding=k//2, bias=False) for k in kernel_sizes
        ])
        
        # 一个并行的最大池化分支
        self.maxconv = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        
        # 合并后的 Batch Normalization 和 ReLU
        self.bn = nn.BatchNorm1d(out_channels * len(kernel_sizes) + out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_b = self.bottleneck(x)
        conv_outs = []
        
        for conv in self.convs:
            out = conv(x_b)
            # 保证序列长度完全对齐
            if out.size(2) > x.size(2):
                out = out[:, :, :x.size(2)]
            conv_outs.append(out)
            
        max_out = self.maxconv(x)
        if max_out.size(2) > x.size(2):
            max_out = max_out[:, :, :x.size(2)]
        conv_outs.append(max_out)
        
        # 在通道维度拼接所有提取到的特征
        out = torch.cat(conv_outs, dim=1)
        return self.relu(self.bn(out))

class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_modules=3):
        super().__init__()
        self.modules_list = nn.ModuleList()
        current_in = in_channels
        
        # 连续堆叠多个 Inception 模块
        for _ in range(num_modules):
            self.modules_list.append(InceptionModule1D(current_in, hidden_channels))
            current_in = hidden_channels * 4 # 3个卷积分支 + 1个池化分支
            
        # 残差连接 (Skip connection)
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, current_in, kernel_size=1, bias=False),
            nn.BatchNorm1d(current_in)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.shortcut(x)
        out = x
        for module in self.modules_list:
            out = module(out)
        out = out + res
        return self.relu(out)

class InceptionTimeTeacher(nn.Module):
    def __init__(self, in_channels=1, num_classes=12, hidden_channels=32, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList()
        current_in = in_channels
        
        # 堆叠 2 个大 Block (共计 6 个 Inception 模块)
        for _ in range(num_blocks):
            self.blocks.append(InceptionBlock1D(current_in, hidden_channels))
            current_in = hidden_channels * 4
            
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(current_in, num_classes)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.pool(x).squeeze(-1)
        x = self.classifier(x)
        return x

# ==========================================
# 3. 画图函数
# ==========================================
def plot_and_save_curves(history, output_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    plt.plot(epochs, history['val_loss'], label='Val Loss', marker='o')
    plt.title('Training and Validation Loss (Teacher)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc', marker='o')
    plt.plot(epochs, history['val_acc'], label='Val Acc', marker='o')
    plt.title('Training and Validation Accuracy (Teacher)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['val_auc'], label='Val AUC', marker='o', color='green')
    plt.title('Validation Macro-AUC (Teacher)')
    plt.xlabel('Epochs')
    plt.ylabel('AUC (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'teacher_training_curves.png')
    plt.savefig(save_path, dpi=300)
    plt.close()

# ==========================================
# 4. 训练主循环
# ==========================================
def main():
    DATA_PATH = './ptb-xl/' 
    OUTPUT_DIR = './outputs/' 
    BATCH_SIZE = 32
    EPOCHS = 30
    LR = 0.001
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] 当前使用设备: {DEVICE}")

    train_dataset = PTBXL_SingleLead_Dataset(DATA_PATH, fold_type='train')
    val_dataset = PTBXL_SingleLead_Dataset(DATA_PATH, fold_type='val')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = InceptionTimeTeacher(in_channels=1, num_classes=12).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.BCEWithLogitsLoss()

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_auc': []}
    best_val_auc = 0.0

    print("\n[*] 开始在 PTB-XL 的【单导联】上训练打标老师...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        train_loop = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{EPOCHS}] [Train]', leave=False)
        for signals, labels in train_loop:
            signals, labels = signals.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()
            
            train_total += labels.numel()
            train_correct += (predicted == labels).sum().item()
            train_loop.set_postfix(loss=f"{loss.item():.4f}")
            
        scheduler.step()
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100 * train_correct / train_total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_val_probs, all_val_labels = [], []
        
        val_loop = tqdm(val_loader, desc=f'Epoch [{epoch+1}/{EPOCHS}] [Val  ]', leave=False)
        with torch.no_grad():
            for signals, labels in val_loop:
                signals, labels = signals.to(DEVICE), labels.to(DEVICE)
                outputs = model(signals) 
                val_loss += criterion(outputs, labels).item()
                
                probs = torch.sigmoid(outputs)
                all_val_probs.extend(probs.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
                
                predicted = (probs > 0.5).float()
                val_total += labels.numel()
                val_correct += (predicted == labels).sum().item()
                
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100 * val_correct / val_total

        all_val_labels = np.array(all_val_labels)
        all_val_probs = np.array(all_val_probs)
        try:
            epoch_val_auc = roc_auc_score(all_val_labels, all_val_probs, average='macro') * 100
        except ValueError:
            epoch_val_auc = 0.0 

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['val_auc'].append(epoch_val_auc)
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val AUC: {epoch_val_auc:.2f}%")
        
        if epoch_val_auc > best_val_auc:
            best_val_auc = epoch_val_auc
            save_path = os.path.join(OUTPUT_DIR, 'best_inception_teacher.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  --> [Saved] 发现新的最佳老师模型! 单导联验证集 AUC: {best_val_auc:.2f}%, 已保存至 {save_path}")

    plot_and_save_curves(history, OUTPUT_DIR)
    print(f"\n[*] 训练完成！老师模型已准备就绪，可以前往给 MIT-BIH 打标了。")

if __name__ == '__main__':
    main()