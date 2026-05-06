import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from chapter2_combined_improvements import (
    PTBXL_Lead2_Dataset, MIT_Lead2_Dataset, AsymmetricLoss, find_optimal_thresholds, calculate_metrics_with_custom_thresholds
)
# 1. 导入你刚刚保存的斯坦福 ResNet
from baselines.stanford_resnet import StanfordResNet34

# 2. 直接在这个文件里内嵌标准 1D-Transformer (免得你再去建文件了)
import math
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Standard_1D_Transformer(nn.Module):
    def __init__(self, c_in=1, c_out=12, d_model=64, nhead=8, num_layers=4):
        super().__init__()
        # 🚀 救命神器：加入前端卷积降维层
        # Stride=4 将 5000 长度直接压缩到 1250，注意力计算量暴降 16 倍！
        self.stem = nn.Sequential(
            nn.Conv1d(c_in, d_model, kernel_size=15, stride=4, padding=7, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        
        # 删除了原来的 input_proj，因为 stem 已经把通道数变成了 d_model
        self.pos_encoder = PositionalEncoding(d_model, max_len=1250) # 长度也相应改小
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, c_out)

    def forward(self, x):
        # 1. 先通过 CNN 降维: (Batch, 1, 5000) -> (Batch, d_model, 1250)
        x = self.stem(x) 
        # 2. 转换维度适配 Transformer: -> (Batch, 1250, d_model)
        x = x.transpose(1, 2)
        # 3. 加上位置编码并进入 Transformer
        x = self.pos_encoder(x)
        x = self.transformer(x)
        # 4. 全局平均池化并分类
        return self.fc(x.mean(dim=1))

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #PTBXL_PATH = './ptb-xl/' 
    MIT_PATH = './outputs/mit_teacher_labeled.npy'
    
    print("\n📦 正在加载 MIT-BIH (Lead II) 软标签数据集...")
    
    #print("\n📦 加载数据中...")
    #train_loader = DataLoader(PTBXL_Lead2_Dataset(PTBXL_PATH, 'train'), batch_size=32, shuffle=True, num_workers=2)
    #val_loader = DataLoader(PTBXL_Lead2_Dataset(PTBXL_PATH, 'val'), batch_size=64, shuffle=False, num_workers=0)
    #test_loader = DataLoader(PTBXL_Lead2_Dataset(PTBXL_PATH, 'test'), batch_size=64, shuffle=False, num_workers=0)
    train_loader = DataLoader(MIT_Lead2_Dataset(MIT_PATH, 'train'), batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(MIT_Lead2_Dataset(MIT_PATH, 'val'), batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(MIT_Lead2_Dataset(MIT_PATH, 'test'), batch_size=64, shuffle=False, num_workers=0)

    # 注册最后一批王者基线
    models_to_test = {
        "Stanford_ResNet34": StanfordResNet34(c_in=1, c_out=12),
        "Vanilla_Transformer": Standard_1D_Transformer(c_in=1, c_out=12)
    }

    results = []
    # 为了代码精简，我把训练逻辑直接套进来
    for name, model in models_to_test.items():
        print(f"\n" + "="*50 + f"\n 🚀 训练 {name} \n" + "="*50)
        model = model.to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
        criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05).to(DEVICE)
        
        best_auc = 0.0
        save_path = f'./outputs/baseline_{name}_best.pth'
        
        for epoch in range(30):
            model.train()
            for signals, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/30', leave=False):
                optimizer.zero_grad()
                loss = criterion(model(signals.to(DEVICE)), labels.to(DEVICE))
                loss.backward()
                optimizer.step()
            scheduler.step()

            model.eval()
            all_p, all_l = [], []
            with torch.no_grad():
                for signals, labels in val_loader:
                    all_p.extend(torch.sigmoid(model(signals.to(DEVICE))).cpu().numpy())
                    all_l.extend(labels.numpy())
            auc, _, _, _, _ = calculate_metrics_with_custom_thresholds(np.array(all_l), np.array(all_p), np.full(12, 0.5))
            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), save_path)
                
        print(f"[*] 训练完成，动态校准盲测...")
        model.load_state_dict(torch.load(save_path, map_location=DEVICE))
        model.eval()
        vp, vl, tp, tl = [], [], [], []
        with torch.no_grad():
            for s, l in val_loader: vp.extend(torch.sigmoid(model(s.to(DEVICE))).cpu().numpy()); vl.extend(l.numpy())
            opt_t = find_optimal_thresholds(np.array(vl), np.array(vp))
            for s, l in test_loader: tp.extend(torch.sigmoid(model(s.to(DEVICE))).cpu().numpy()); tl.extend(l.numpy())
        
        m_auc, m_sen, m_spe, m_f1, mi_f1 = calculate_metrics_with_custom_thresholds(np.array(tl), np.array(tp), opt_t)
        results.append({"Model": name, "Mac-AUC": m_auc, "Mac-SEN": m_sen, "Mac-SPE": m_spe, "Mac-F1": m_f1, "Mic-F1": mi_f1})

    print("\n" + "★"*80 + "\n 📊 论文第二章：最后一批顶流基线成绩单 \n" + "★"*80)
    for r in results:
        print(f"{r['Model']:<20} | AUC:{r['Mac-AUC']:.2f}% | SEN:{r['Mac-SEN']:.2f}% | F1:{r['Mac-F1']:.2f}% | Mic-F1:{r['Mic-F1']:.2f}%")

if __name__ == '__main__':
    main()