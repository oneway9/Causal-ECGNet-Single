import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from dataset import PTBXL_Rhythm_Dataset
from model import CausalECGNet

def plot_and_save_curves(history, output_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    plt.plot(epochs, history['val_loss'], label='Val Loss', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc', marker='o')
    plt.plot(epochs, history['val_acc'], label='Val Acc', marker='o')
    plt.title('Training and Validation Accuracy')
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
    save_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\n[Info] 训练曲线已保存至: {save_path}")

def main():
    DATA_PATH = './ptb-xl/' 
    OUTPUT_DIR = './outputs/' 
    
    BATCH_SIZE = 64
    EPOCHS = 30
    LR = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nLoading datasets...")
    train_dataset = PTBXL_Rhythm_Dataset(DATA_PATH, fold_type='train')
    val_dataset = PTBXL_Rhythm_Dataset(DATA_PATH, fold_type='val')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = CausalECGNet(num_classes=12, hidden_channels=64).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # 核心修改 1：多标签二分类必须使用 BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()

    history = {
        'train_loss': [], 'val_loss': [], 
        'train_acc': [], 'val_acc': [],
        'val_auc': []  
    }
    
    best_val_auc = 0.0

    print("\nStarting Training...")
    for epoch in range(EPOCHS):
        # -------------------
        # Train Phase
        # -------------------
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_loop = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{EPOCHS}] [Train]', leave=False)
        for signals, labels in train_loop:
            signals, labels = signals.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            # 由于标签变成了多热编码，混淆字典的更新逻辑在模型内部能够自动兼容
            # 找到这行旧代码：
            # pred_upper, pred_lower = model(signals, labels)
            
            # 将它替换为：
            pred_upper, pred_lower = model(signals, labels, current_epoch=epoch)
            
            loss_upper = criterion(pred_upper, labels)
            loss_lower = criterion(pred_lower, labels)
            loss = loss_upper + loss_lower
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # 核心修改 2：多标签准确率计算 (Sigmoid + 阈值判断)
            pred = (pred_upper + pred_lower) / 2
            probs = torch.sigmoid(pred)
            predicted = (probs > 0.5).float()
            
            # 统计总预测数量时，需要乘以类别数 (Batch Size * 12)
            train_total += labels.numel() 
            train_correct += (predicted == labels).sum().item()
            
            current_acc = 100 * train_correct / train_total
            train_loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{current_acc:.2f}%")
            
        scheduler.step()
        
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100 * train_correct / train_total

        # -------------------
        # Validation Phase
        # -------------------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        all_val_probs = []
        all_val_labels = []
        
        val_loop = tqdm(val_loader, desc=f'Epoch [{epoch+1}/{EPOCHS}] [Val  ]', leave=False)
        with torch.no_grad():
            for signals, labels in val_loop:
                signals, labels = signals.to(DEVICE), labels.to(DEVICE)
                
                pred = model(signals) 
                loss = criterion(pred, labels)
                val_loss += loss.item()
                
                # 核心修改 3：验证集同样使用 Sigmoid
                probs = torch.sigmoid(pred)
                all_val_probs.extend(probs.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
                
                predicted = (probs > 0.5).float()
                val_total += labels.numel()
                val_correct += (predicted == labels).sum().item()
                
                current_val_acc = 100 * val_correct / val_total
                val_loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{current_val_acc:.2f}%")
                
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100 * val_correct / val_total

        all_val_labels = np.array(all_val_labels)
        all_val_probs = np.array(all_val_probs)
        try:
            # 核心修改 4：多标签场景直接传入 multi-hot 矩阵即可，不需要 multi_class='ovr'
            epoch_val_auc = roc_auc_score(all_val_labels, all_val_probs, average='macro') * 100
        except ValueError as e:
            epoch_val_auc = 0.0 

        # -------------------
        # 记录与保存
        # -------------------
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['val_auc'].append(epoch_val_auc)
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
              f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | "
              f"Val Acc: {epoch_val_acc:.2f}% | Val AUC: {epoch_val_auc:.2f}%")
        
        if epoch_val_auc > best_val_auc:
            best_val_auc = epoch_val_auc
            save_path = os.path.join(OUTPUT_DIR, 'best_causal_ecgnet.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  --> [Saved] 发现新的最佳模型! 验证集 AUC 提升至: {best_val_auc:.2f}%, 已保存至 {save_path}")

    print("\nTraining completed! Generating plots...")
    plot_and_save_curves(history, OUTPUT_DIR)

if __name__ == '__main__':
    main()