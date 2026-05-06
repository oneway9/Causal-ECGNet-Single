import os
import ast
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import wfdb
from scipy.signal import butter, filtfilt

class PTBXL_Rhythm_Dataset(Dataset):
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
        elif fold_type == 'test':
            self.df = df[df.strat_fold == 10]
            
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
                
        print(f"Loaded {len(self.samples)} samples for {fold_type} set.")
        
        # 初始化带通滤波器 (0.5Hz ~ 40Hz, 采样率 500Hz)
        nyq = 0.5 * 500.0
        low = 0.5 / nyq
        high = 40.0 / nyq
        self.b, self.a = butter(3, [low, high], btype='bandpass')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        record_path = os.path.join(self.data_path, self.samples[idx])
        signal, _ = wfdb.rdsamp(record_path)
        
        # 应用零相位滤波消除基线漂移和高频噪声
        signal = filtfilt(self.b, self.a, signal, axis=0)
        
        signal = torch.tensor(signal.copy(), dtype=torch.float32).transpose(0, 1)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return signal, label