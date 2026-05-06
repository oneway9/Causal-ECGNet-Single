import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# 导入 Inception 消融模型
from models_ablation import InceptionCausal_MacroOnly
from train_mixed_inception_causal import Mixed_SingleLead_Dataset, safe_macro_auc

def main():
    PTBXL_PATH = './ptb-xl/' 
    MIT_NPY_PATH = './outputs/mit_teacher_labeled.npy'
    OUTPUT_DIR = './outputs/' 
    
    BATCH_SIZE = 32 # 稳妥起见保持 32
    EPOCHS = 30
    LR = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = Mixed_SingleLead_Dataset(PTBXL_PATH, MIT_NPY_PATH, fold_type='train')
    val_dataset = Mixed_SingleLead_Dataset(PTBXL_PATH, fold_type='val')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 实例化 Inception 单分支模型
    model = InceptionCausal_MacroOnly(num_classes=12, hidden_channels=64).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0

    print("\n[*] 开始消融实验 2：训练 Inception 宏观单分支模型...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        train_loop = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{EPOCHS}] [Train]', leave=False)
        for signals, labels in train_loop:
            signals, labels = signals.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            pred = model(signals, labels, current_epoch=epoch)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_loop.set_postfix(loss=f"{loss.item():.4f}")
            
        scheduler.step()
        epoch_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        all_val_probs, all_val_hard_labels = [], []
        with torch.no_grad():
            for signals, labels in val_loader:
                signals, labels = signals.to(DEVICE), labels.to(DEVICE)
                pred = model(signals) 
                probs = torch.sigmoid(pred)
                all_val_probs.extend(probs.cpu().numpy())
                all_val_hard_labels.extend(labels.cpu().numpy())
                
        epoch_val_auc = safe_macro_auc(np.array(all_val_hard_labels), np.array(all_val_probs))
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {epoch_train_loss:.4f} | Val AUC: {epoch_val_auc:.2f}%")
        
        if epoch_val_auc > best_val_auc:
            best_val_auc = epoch_val_auc
            save_path = os.path.join(OUTPUT_DIR, 'best_ablation_inception_macro.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  --> [Saved] 发现新的最佳 Inception 单分支模型! AUC: {best_val_auc:.2f}%")

if __name__ == '__main__':
    main()