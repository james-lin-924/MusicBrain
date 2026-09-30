import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
from collections import defaultdict

# 引入自定義的模組
from data import PianoRollDataset
from model import AttentionConvNeXt

CONFIG = {
    'root_dir': '../Datas',
    'batch_size': 16,
    'lr': 1e-4,
    'epochs': 250,
    'num_workers': 4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_path': 'best_model.pth'
}

def calculate_metrics(preds, labels, dataset_info):
    """
    從 60 類預測中解析出 Source 和 Style 的準確率
    """
    source_correct = 0
    style_correct = 0
    total = len(preds)
    
    # 用於統計每個 Class 的詳細準確率
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for p, t in zip(preds, labels):
        p_idx, t_idx = p.item(), t.item()
        
        # 取得對應的資訊字典
        p_info = dataset_info[p_idx]
        t_info = dataset_info[t_idx]
        
        # 1. 檢查 Source 是否一致
        if p_info['source'] == t_info['source']:
            source_correct += 1
            
        # 2. 檢查 Style (Folder Name) 是否一致
        if p_info['style'] == t_info['style']:
            style_correct += 1
            
        # 3. 檢查 60-Class 是否完全命中
        class_total[t_idx] += 1
        if p_idx == t_idx:
            class_correct[t_idx] += 1

    return source_correct, style_correct, class_correct, class_total

def train_model():
    print(f"Starting training on device: {CONFIG['device']}")
    
    # 1. 準備資料集
    full_dataset = PianoRollDataset(root_dir=CONFIG['root_dir'], clip_duration=15)
    
    if len(full_dataset) == 0:
        print("No images found.")
        return

    num_classes = len(full_dataset.classes)
    print(f"Model will output {num_classes} classes.")

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # 重要：傳遞 class_info 給驗證階段使用
    dataset_info = full_dataset.class_info

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True)

    # 2. 初始化模型
    model = AttentionConvNeXt(num_classes=num_classes)
    model = model.to(CONFIG['device'])
    
    # 3. 損失函數與優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])

    best_loss = float('inf')
    
    # 4. 訓練迴圈
    for epoch in range(CONFIG['epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        print('-' * 10)
        
        # --- Training ---
        model.train()
        running_loss = 0.0
        train_source_corr = 0
        train_style_corr = 0
        total_samples = 0
        
        train_bar = tqdm(train_loader, desc="Training", leave=False)
        
        for inputs, labels in train_bar:
            inputs = inputs.to(CONFIG['device'])
            labels = labels.to(CONFIG['device']) # 0~59 labels
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            
            # 計算統計
            _, preds = torch.max(outputs, 1)
            src_c, sty_c, _, _ = calculate_metrics(preds, labels, dataset_info)
            train_source_corr += src_c
            train_style_corr += sty_c
            
            train_bar.set_postfix(loss=loss.item())
            
        train_loss = running_loss / total_samples
        train_src_acc = train_source_corr / total_samples
        train_sty_acc = train_style_corr / total_samples
        
        print(f"Train Loss: {train_loss:.4f} | Source Acc: {train_src_acc:.4f} | Style Acc: {train_sty_acc:.4f}")
        
        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_source_corr = 0
        val_style_corr = 0
        val_total = 0
        
        # 統計每個類別的準確率
        all_class_correct = defaultdict(int)
        all_class_total = defaultdict(int)
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(CONFIG['device'])
                labels = labels.to(CONFIG['device'])
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                val_total += inputs.size(0)
                
                _, preds = torch.max(outputs, 1)
                src_c, sty_c, cls_c, cls_t = calculate_metrics(preds, labels, dataset_info)
                
                val_source_corr += src_c
                val_style_corr += sty_c
                
                # 累加每個 batch 的詳細統計
                for k, v in cls_c.items(): all_class_correct[k] += v
                for k, v in cls_t.items(): all_class_total[k] += v

        avg_val_loss = val_loss / val_total
        val_src_acc = val_source_corr / val_total
        val_sty_acc = val_style_corr / val_total
        
        print(f"Val Loss:   {avg_val_loss:.4f} | Source Acc: {val_src_acc:.4f} | Style Acc: {val_sty_acc:.4f}")

        # --- 每個 Epoch 結束時印出各類別 Accuracy (可選：為了版面整潔，可以設定每10個epoch印一次) ---
        if (epoch + 1) % 10 == 0 or (epoch + 1) == CONFIG['epochs']:
            print("\n--- Per-Class Accuracy (Validation) ---")
            # 排序輸出
            sorted_indices = sorted(all_class_total.keys())
            for idx in sorted_indices:
                cls_name = dataset_info[idx]['name']
                acc = all_class_correct[idx] / all_class_total[idx] if all_class_total[idx] > 0 else 0.0
                print(f"Class {idx:02d} [{cls_name}]: {acc:.2%}")
            print("---------------------------------------\n")

        scheduler.step()
        
        # 根據 Val Loss 保存最佳模型
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), CONFIG['save_path'])
            print(f">>> Best model saved with Val Loss: {best_loss:.4f}")

    print("Training Complete.")

if __name__ == '__main__':
    train_model()