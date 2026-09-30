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
    'batch_size': 32, 
    'lr': 1e-4,       # 🌟 微調使用較小的 Learning Rate
    'epochs': 100,    # 微調通常不需要太多 Epoch
    'warmup_epochs': 2,
    'num_workers': 4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'pretrained_model': 'best_temporal_smooth_model.pth', # 🌟 指定舊模型權重路徑
    'save_path': 'best_finetuned_source_model.pth',
}

# 🌟 為了補強 Source，我們翻轉權重比例！逼迫模型重視 Source 的 Loss
ALPHA = 2.0  # Source 權重 (原 0.5 -> 改為 2.0)
BETA  = 0.5  # Style 權重 (原 2.0 -> 改為 0.5)

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

# 🌟 改為 Frame-level 的準確率計算，不再使用 File-level
def get_frame_level_predictions(out_logits, original_labels):
    # out_logits: (B, T, C), original_labels: (B, T)
    preds = torch.argmax(out_logits, dim=-1)
    return preds.reshape(-1), original_labels.reshape(-1)

def train_model():
    base_dataset = PianoRollDataset(root_dir=CONFIG['root_dir'], is_train=False, max_seq_len=MAX_SEQ_LEN)
    indices = list(range(len(base_dataset)))
    random.shuffle(indices)
    train_size = int(0.8 * len(base_dataset)) 
    
    train_ds = Subset(PianoRollDataset(root_dir=CONFIG['root_dir'], is_train=True, max_seq_len=MAX_SEQ_LEN), indices[:train_size])
    val_ds   = Subset(base_dataset, indices[train_size:])

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True, collate_fn=collate_fn)

    # 確保 max_seq_len=250 與舊權重的 pos_embedding 相符
    model = TemporalDualHeadTransformer(num_styles=21, max_seq_len=250).to(CONFIG['device'])
    
    # 🌟 載入預訓練權重
    if os.path.exists(CONFIG['pretrained_model']):
        print(f"-> 成功載入預訓練權重: {CONFIG['pretrained_model']}")
        model.load_state_dict(torch.load(CONFIG['pretrained_model'], map_location=CONFIG['device']))
    else:
        print("-> ⚠️ 找不到預訓練權重，將從頭開始訓練！")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-1)
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=CONFIG['warmup_epochs'])
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'] - CONFIG['warmup_epochs'])
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[CONFIG['warmup_epochs']])
    
    scaler = torch.amp.GradScaler('cuda') 
    best_src_acc = 0.0 # 🌟 這次以 Source 的準確率作為儲存最佳模型的依據
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss, train_src_corr, train_sty_corr, total_frames = 0.0, 0, 0, 0
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
                
                loss_src = criterion(out_src_t, src_labels_t)
                loss_sty = criterion(out_sty_t, sty_labels_t)
                loss = (ALPHA * loss_src) + (BETA * loss_sty)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * inputs.size(0)
            
            # 取得 Frame-level 的預測與真實標籤並計算正確數
            preds_src, t_src = get_frame_level_predictions(out_src, src_labels)
            preds_sty, t_sty = get_frame_level_predictions(out_sty, sty_labels)
            
            train_src_corr += (preds_src == t_src).sum().item()
            train_sty_corr += (preds_sty == t_sty).sum().item()
            total_frames += t_src.size(0) 
            
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
            
        t_src_acc = train_src_corr / total_frames
        t_sty_acc = train_sty_corr / total_frames
        
        # Validation
        model.eval()
        val_loss, val_src_corr, val_sty_corr, val_total_frames = 0.0, 0, 0, 0
        with torch.no_grad():
            for inputs, src_labels, sty_labels in val_loader:
                inputs, src_labels, sty_labels = inputs.to(CONFIG['device']), src_labels.to(CONFIG['device']), sty_labels.to(CONFIG['device'])
                with torch.amp.autocast('cuda'):
                    out_src, out_sty = model(inputs)
                    
                    out_src_t = out_src.reshape(-1, out_src.shape[-1])
                    out_sty_t = out_sty.reshape(-1, out_sty.shape[-1])
                    src_labels_t = src_labels.reshape(-1)
                    sty_labels_t = sty_labels.reshape(-1)
                    
                    loss = (ALPHA * criterion(out_src_t, src_labels_t)) + (BETA * criterion(out_sty_t, sty_labels_t))
                           
                val_loss += loss.item() * inputs.size(0)
                preds_src, t_src = get_frame_level_predictions(out_src, src_labels)
                preds_sty, t_sty = get_frame_level_predictions(out_sty, sty_labels)
                
                val_src_corr += (preds_src == t_src).sum().item()
                val_sty_corr += (preds_sty == t_sty).sum().item()
                val_total_frames += t_src.size(0)
                
        v_src_acc = val_src_corr / val_total_frames
        v_sty_acc = val_sty_corr / val_total_frames
        scheduler.step()
        
        print(f"-> Train | Loss: {(train_loss/len(train_loader.dataset)):.4f} | Src(Frame): {t_src_acc:.4f} | Sty(Frame): {t_sty_acc:.4f}")
        print(f"-> Val   | Loss: {(val_loss/len(val_loader.dataset)):.4f} | Src(Frame): {v_src_acc:.4f} | Sty(Frame): {v_sty_acc:.4f}")
        
        # 🌟 這次我們以 Source 準確率作為存檔標準
        if v_src_acc > best_src_acc:
            best_src_acc = v_src_acc
            torch.save(model.state_dict(), CONFIG['save_path'])
            print(f"*** 🚀 新的 Source 最佳模型儲存 (Frame Src Acc: {best_src_acc:.4f}) ***")

if __name__ == '__main__':
    train_model()