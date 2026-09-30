import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import random
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")

from data import PianoRollDataset
from model import TemporalDualHeadTransformer

MAX_SEQ_LEN = 2000

CONFIG = {
    'root_dir': '../Datas',
    'batch_size': 16,     # 【更動】若顯示卡記憶體充足可改回 32，若 OOM 則降至 16 或 8
    'lr': 3e-4,       
    'epochs': 500,
    'warmup_epochs': 10,  # 【更動】大模型需要稍長一點的 Warmup 預熱
    'num_workers': 4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_path': 'best_temporal_smooth_model_large.pth',
}

# 聯合損失權重調整
ALPHA = 0.05  
BETA  = 2.0   

# 實作 Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

def collate_fn(batch):
    imgs = torch.stack([item[0] for item in batch], dim=0)
    src_labels = torch.stack([item[1] for item in batch], dim=0)
    sty_labels = torch.stack([item[2] for item in batch], dim=0)
    return imgs, src_labels, sty_labels

def get_file_level_predictions(out_logits, original_labels):
    probs = torch.softmax(out_logits, dim=-1)
    
    if probs.shape[-1] == 21:
        valid_probs = probs[:, :, :20].mean(dim=1) 
    else:
        valid_probs = probs.mean(dim=1)
        
    _, preds = torch.max(valid_probs, dim=1)
    true_labels = original_labels[:, 0]
    return preds, true_labels

def train_model():
    base_dataset = PianoRollDataset(root_dir=CONFIG['root_dir'], is_train=False, max_seq_len=MAX_SEQ_LEN)
    indices = list(range(len(base_dataset)))
    random.shuffle(indices)
    train_size = int(0.8 * len(base_dataset)) 
    
    train_ds = Subset(PianoRollDataset(root_dir=CONFIG['root_dir'], is_train=True, max_seq_len=MAX_SEQ_LEN), indices[:train_size])
    val_ds   = Subset(base_dataset, indices[train_size:])

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True, collate_fn=collate_fn)

    # 【更動】呼叫加深加寬版的模型，並明確給定參數
    model = TemporalDualHeadTransformer(
        num_styles=21, 
        max_seq_len=200,
        d_model=256,   
        nhead=8,       
        num_layers=6   
    ).to(CONFIG['device'])
    
    criterion_src = nn.CrossEntropyLoss(label_smoothing=0.2)
    criterion_sty = FocalLoss(gamma=2.0, label_smoothing=0.2)
    
    # 【更動】降低 weight_decay，原來的 0.2 對於深寬模型懲罰過強，降回標準值 0.05
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=0.05)
    
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=CONFIG['warmup_epochs'])
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'] - CONFIG['warmup_epochs'])
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[CONFIG['warmup_epochs']])
    
    scaler = torch.amp.GradScaler('cuda') 
    best_sty_acc = 0.0
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss, train_src_corr, train_sty_corr, total_count = 0.0, 0, 0, 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        
        for inputs, src_labels, sty_labels in train_bar:
            inputs, src_labels, sty_labels = inputs.to(CONFIG['device']), src_labels.to(CONFIG['device']), sty_labels.to(CONFIG['device'])
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                out_src, out_sty = model(inputs)
                
                out_src_t = out_src.reshape(-1, out_src.shape[-1])
                out_sty_t = out_sty.reshape(-1, out_sty.shape[-1])
                src_labels_t = src_labels.reshape(-1)
                sty_labels_t = sty_labels.reshape(-1)
                
                loss_src = criterion_src(out_src_t, src_labels_t)
                loss_sty = criterion_sty(out_sty_t, sty_labels_t)
                loss = (ALPHA * loss_src) + (BETA * loss_sty)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * inputs.size(0)
            
            preds_src, t_src = get_file_level_predictions(out_src, src_labels)
            preds_sty, t_sty = get_file_level_predictions(out_sty, sty_labels)
            
            train_src_corr += (preds_src == t_src).sum().item()
            train_sty_corr += (preds_sty == t_sty).sum().item()
            total_count += inputs.size(0)
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
            
        t_src_acc = train_src_corr / total_count
        t_sty_acc = train_sty_corr / total_count
        
        model.eval()
        val_loss, val_src_corr, val_sty_corr, val_total = 0.0, 0, 0, 0
        with torch.no_grad():
            for inputs, src_labels, sty_labels in val_loader:
                inputs, src_labels, sty_labels = inputs.to(CONFIG['device']), src_labels.to(CONFIG['device']), sty_labels.to(CONFIG['device'])
                with torch.amp.autocast('cuda'):
                    out_src, out_sty = model(inputs)
                    out_src_t, out_sty_t = out_src.reshape(-1, out_src.shape[-1]), out_sty.reshape(-1, out_sty.shape[-1])
                    
                    loss_src = criterion_src(out_src_t, src_labels.reshape(-1))
                    loss_sty = criterion_sty(out_sty_t, sty_labels.reshape(-1))
                    loss = (ALPHA * loss_src) + (BETA * loss_sty)
                           
                val_loss += loss.item() * inputs.size(0)
                preds_src, t_src = get_file_level_predictions(out_src, src_labels)
                preds_sty, t_sty = get_file_level_predictions(out_sty, sty_labels)
                
                val_src_corr += (preds_src == t_src).sum().item()
                val_sty_corr += (preds_sty == t_sty).sum().item()
                val_total += inputs.size(0)
                
        v_src_acc = val_src_corr / val_total
        v_sty_acc = val_sty_corr / val_total
        scheduler.step()
        
        print(f"-> Train | Loss: {(train_loss/total_count):.4f} | Src: {t_src_acc:.4f} | Sty: {t_sty_acc:.4f}")
        print(f"-> Val   | Loss: {(val_loss/val_total):.4f} | Src: {v_src_acc:.4f} | Sty: {v_sty_acc:.4f}")
        
        if v_sty_acc > best_sty_acc:
            best_sty_acc = v_sty_acc
            torch.save(model.state_dict(), CONFIG['save_path'])
            print(f"*** 🚀 新的最佳模型儲存 (Style Acc: {best_sty_acc:.4f}) ***")

if __name__ == '__main__':
    train_model()