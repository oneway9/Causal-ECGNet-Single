import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ==========================================
# 1. MIT-BIH 专属数据集加载器
# ==========================================
class MIT_Pseudo_Dataset(Dataset):
    def __init__(self, npy_path):
        super().__init__()
        print(f"[*] 正在加载 MIT-BIH 伪标签数据集: {npy_path}")
        self.data = np.load(npy_path, allow_pickle=True)
        print(f"[*] 成功加载 {len(self.data)} 条数据段。")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # 原始 signal 形状是 (5000,)，PyTorch CNN 需要 (channels, length) -> (1, 5000)
        signal = torch.tensor(item['signal'], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(item['pseudo_label'], dtype=torch.float32)
        return signal, label

# ==========================================
# 2. 单导联版 Causal ECGNet 相关模块
# ==========================================
class ResNet1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=7):
        super(ResNet1DBlock, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class ResNetBackboneSingle(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=64):
        super().__init__()
        self.first_layer = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = ResNet1DBlock(32, 32, stride=1, kernel_size=7)
        self.layer2 = ResNet1DBlock(32, hidden_channels, stride=2, kernel_size=7)
        self.layer3 = ResNet1DBlock(hidden_channels, hidden_channels, stride=2, kernel_size=7)
        self.layer4 = ResNet1DBlock(hidden_channels, hidden_channels, stride=2, kernel_size=7)
        self.pool = nn.AdaptiveAvgPool1d(125)

    def forward(self, x):
        x = self.first_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.pool(x)

class AttentionFusion(nn.Module):
    def __init__(self, feature_dim=64, num_classes=12):
        super().__init__()
        self.W_k = nn.Linear(feature_dim, feature_dim)
        self.W_v = nn.Linear(feature_dim, feature_dim)
        self.scale = feature_dim ** 0.5
        
    def forward(self, Q, f_dict):
        K = self.W_k(f_dict) 
        V = self.W_v(f_dict) 
        attn_scores = torch.matmul(Q, K.transpose(0, 1)) / self.scale 
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, V) 
        return torch.cat([Q, attn_out], dim=-1)

class SingleLeadCausalECGNet(nn.Module):
    def __init__(self, num_classes=12, hidden_channels=64):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        
        # 核心修改：输入通道改为 1
        self.backbone = ResNetBackboneSingle(in_channels=1, hidden_channels=hidden_channels)
        
        self.T1 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2)
        self.T2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=50, padding=25)
        
        self.af_upper = AttentionFusion(feature_dim=hidden_channels, num_classes=num_classes)
        self.af_lower = AttentionFusion(feature_dim=hidden_channels, num_classes=num_classes)
        
        self.mlp_upper = nn.Sequential(nn.Linear(hidden_channels * 2, hidden_channels), nn.ReLU())
        self.mlp_lower = nn.Sequential(nn.Linear(hidden_channels * 2, hidden_channels), nn.ReLU())
        
        self.classifier_upper = nn.Linear(hidden_channels, num_classes)
        self.classifier_lower = nn.Linear(hidden_channels, num_classes)
        
        self.register_buffer('confounder_dict', torch.zeros(num_classes, hidden_channels))

    def forward(self, x, labels=None, current_epoch=0):
        # 核心修改：无需再做 view 切分，直接跑 backbone
        f_m = self.backbone(x) 
        f_up = self.T1(f_m).mean(dim=-1)   
        f_low = self.T2(f_m).mean(dim=-1)  
        
        # 单导联直接作为 Query，无需 average
        Q_upper = f_up 
        Q_lower = f_low 
        
        warmup_epochs = 5 
        if self.training and labels is not None and current_epoch >= warmup_epochs:
            with torch.no_grad():
                alpha = 0.9 
                for c in range(self.num_classes):
                    mask = (labels[:, c] == 1.0) 
                    if mask.any():
                        class_feat = ((Q_upper[mask] + Q_lower[mask]) / 2).mean(dim=0)
                        if torch.sum(self.confounder_dict[c]) == 0:
                            self.confounder_dict[c] = class_feat
                        else:
                            self.confounder_dict[c] = alpha * self.confounder_dict[c] + (1 - alpha) * class_feat
        
        f_fused_upper = self.af_upper(Q_upper, self.confounder_dict) 
        f_fused_lower = self.af_lower(Q_lower, self.confounder_dict) 
        
        f_cau_upper = self.mlp_upper(f_fused_upper) 
        f_cau_lower = self.mlp_lower(f_fused_lower) 
        
        pred_upper = self.classifier_upper(f_cau_upper)
        pred_lower = self.classifier_lower(f_cau_lower)
        
        if self.training:
            return pred_upper, pred_lower
        else:
            return (pred_upper + pred_lower) / 2

# ==========================================
# 3. 辅助画图函数
# ==========================================
def plot_and_save_curves(history, output_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    plt.plot(epochs, history['val_loss'], label='Val Loss', marker='o')
    plt.title('Training and Validation Loss (MIT-BIH)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc', marker='o')
    plt.plot(epochs, history['val_acc'], label='Val Acc', marker='o')
    plt.title('Training and Validation Accuracy (MIT-BIH)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['val_auc'], label='Val AUC', marker='o', color='green')
    plt.title('Validation Macro-AUC (MIT-BIH)')
    plt.xlabel('Epochs')
    plt.ylabel('AUC (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'mit_training_curves.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\n[Info] 训练曲线已保存至: {save_path}")

# ==========================================
# 4. 训练主循环
# ==========================================
def main():
    DATASET_PATH = './outputs/mit_pseudo_labeled.npy'
    OUTPUT_DIR = './outputs/' 
    BATCH_SIZE = 64
    EPOCHS = 30
    LR = 0.001
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] 使用设备: {DEVICE}")

    full_dataset = MIT_Pseudo_Dataset(DATASET_PATH)
    
    # 按照 8:2 划分训练集和验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"[*] 训练集大小: {train_size}, 验证集大小: {val_size}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = SingleLeadCausalECGNet(num_classes=12, hidden_channels=64).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.BCEWithLogitsLoss()

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_auc': []}
    best_val_auc = 0.0

    print("\n[*] 开始在 MIT-BIH 跨库数据上训练...")
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
            
            probs = torch.sigmoid((pred_upper + pred_lower) / 2)
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
                pred = model(signals) 
                val_loss += criterion(pred, labels).item()
                
                probs = torch.sigmoid(pred)
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
            save_path = os.path.join(OUTPUT_DIR, 'best_mit_single_lead_ecgnet.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  --> [Saved] 发现新的最佳模型! 验证集 AUC 提升至: {best_val_auc:.2f}%, 已保存至 {save_path}")

    print("\n[*] 训练完成！正在生成训练曲线图...")
    plot_and_save_curves(history, OUTPUT_DIR)

if __name__ == '__main__':
    main()