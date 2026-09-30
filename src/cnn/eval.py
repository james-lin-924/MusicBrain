import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import os

# 引入自定義模組
from data import PianoRollDataset
from model import AttentionConvNeXt

# ==========================================
# 設定
# ==========================================
CONFIG = {
    'root_dir': '../Datas',
    'batch_size': 16,
    'num_workers': 4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'model_path': 'best_model.pth'
}

def plot_confusion_matrix(cm, classes, title, filename):
    """
    繪製並儲存混淆矩陣
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved: {filename}")

def evaluate():
    print(f"Using device: {CONFIG['device']}")
    
    # 1. 載入資料集
    full_dataset = PianoRollDataset(root_dir=CONFIG['root_dir'], clip_duration=15)
    
    # 確保類別數量正確 (應為 60)
    num_classes = len(full_dataset.classes)
    print(f"Total classes: {num_classes}")
    
    if num_classes != 60:
        print("Warning: Class count is not 60. Check data.py logic.")

    # 切分出驗證集 (與訓練時保持一致的比例，建議設定 seed 以確保一致性)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    # 設定 Seed 確保與訓練時的驗證集盡可能接近 (若訓練沒設 seed 則無法完全保證)
    generator = torch.Generator().manual_seed(42) 
    _, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        num_workers=CONFIG['num_workers']
    )

    # 2. 載入模型
    model = AttentionConvNeXt(num_classes=num_classes)
    if os.path.exists(CONFIG['model_path']):
        model.load_state_dict(torch.load(CONFIG['model_path'], map_location=CONFIG['device']))
        print("Model weights loaded successfully.")
    else:
        print(f"Error: Model file '{CONFIG['model_path']}' not found.")
        return

    model = model.to(CONFIG['device'])
    model.eval()

    # 3. 推論 (Inference)
    all_preds = []
    all_labels = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(CONFIG['device'])
            labels = labels.to(CONFIG['device'])
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 4. 解析結果
    # 假設 ID 映射: 0-19 Human, 20-39 Large, 40-59 Melody
    # 每個區間內的 0-19 順序對應相同的 Style
    
    # --- 準備標籤名稱 ---
    sources = ['Human', 'Large', 'Melody']
    # 取得前20個類別的 Style 名稱 (去除 'Human::' 前綴)
    # class_info 結構: {id: {'source':..., 'style':..., 'name':...}}
    style_names = [full_dataset.class_info[i]['style'] for i in range(20)]

    # --- 轉換為 Source (0-2) 與 Style (0-19) ---
    # Source ID = Label // 20
    # Style ID  = Label % 20
    
    true_sources = all_labels // 20
    pred_sources = all_preds // 20
    
    true_styles = all_labels % 20
    pred_styles = all_preds % 20

    # ==========================================
    # 圖 1: Source Confusion Matrix (3x3)
    # ==========================================
    cm_source = confusion_matrix(true_sources, pred_sources)
    plot_confusion_matrix(cm_source, sources, "Source Confusion Matrix", "confusion_matrix_source.png")
    
    acc_source = accuracy_score(true_sources, pred_sources)
    print(f"Total Source Accuracy: {acc_source:.4f}")

    # ==========================================
    # 圖 2, 3, 4: Composer-Style Confusion Matrix (20x20) per Source
    # ==========================================
    
    for i, source_name in enumerate(sources):
        # 篩選出該 Source 的真實樣本 (True Positive Source candidates)
        # 我們關注的是：對於真的是 'Human' 的資料，模型預測的 'Style' 是什麼？
        # 注意：即使模型預測錯 Source (例如預測成 Large 的 Bach)，這裡 pred_styles 依然是 Bach (index 一樣)
        # 這能反映模型對 Style 特徵的辨識力，無論 Source 為何。
        
        mask = (true_sources == i)
        
        if np.sum(mask) == 0:
            print(f"Warning: No samples found for source {source_name}")
            continue
            
        filtered_true_styles = true_styles[mask]
        filtered_pred_styles = pred_styles[mask]
        
        cm_style = confusion_matrix(filtered_true_styles, filtered_pred_styles, labels=list(range(20)))
        
        # 繪圖
        filename = f"confusion_matrix_style_{source_name}.png"
        plot_confusion_matrix(cm_style, style_names, f"Style Confusion Matrix ({source_name})", filename)
        
        acc_style = accuracy_score(filtered_true_styles, filtered_pred_styles)
        print(f"Style Accuracy for {source_name}: {acc_style:.4f}")

if __name__ == '__main__':
    evaluate()