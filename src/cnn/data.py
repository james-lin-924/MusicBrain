import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
import warnings
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

class PianoRollDataset(Dataset):
    def __init__(self, root_dir, fs=100, clip_duration=15, transform=None):
        self.root_dir = root_dir
        self.fs = fs
        self.clip_width = fs * clip_duration
        self.transform = transform
        self.samples = []
        
        # 儲存類別資訊，用於之後反查
        # 格式: { label_id: {'source': 'Human', 'style': 'Bach-Baroque', 'name': 'Human_Bach-Baroque'} }
        self.class_info = {} 
        self.classes = [] # 列表形式儲存類別名稱
        
        current_label_idx = 0
        
        # 定義來源順序以確保編號固定 (Human -> large -> melody)
        sorted_sources = sorted(['Human', 'large', 'melody'])
        
        if os.path.exists(root_dir):
            for source_name in sorted_sources:
                source_path = os.path.join(root_dir, source_name)
                if not os.path.isdir(source_path): continue
                
                # 遍歷該來源下的所有作曲家資料夾 (並排序)
                for folder_name in sorted(os.listdir(source_path)):
                    class_path = os.path.join(source_path, folder_name)
                    if not os.path.isdir(class_path): continue
                    
                    # 定義這個組合為一個唯一的類別
                    # 例如: Human/Bach-Baroque
                    unique_class_name = f"{source_name}::{folder_name}"
                    
                    self.classes.append(unique_class_name)
                    self.class_info[current_label_idx] = {
                        'source': source_name,
                        'style': folder_name, # 這裡的 style 指的是 folder 名稱 (如 Bach-Baroque)
                        'name': unique_class_name
                    }
                    
                    # 讀取該資料夾下的圖片
                    for img_name in os.listdir(class_path):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.samples.append((
                                os.path.join(class_path, img_name),
                                current_label_idx
                            ))
                    
                    current_label_idx += 1
            
            print(f"Dataset Initialized: Found {len(self.samples)} images.")
            print(f"Total Unique Classes: {len(self.classes)} (Target should be 60)")
            
        else:
            print(f"Error: Root directory '{root_dir}' not found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 現在只回傳 圖片 和 一個 Label ID (0-59)
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('L')
            width, height = image.size
            
            if width > self.clip_width:
                start_x = np.random.randint(0, width - self.clip_width)
                box = (start_x, 0, start_x + self.clip_width, height)
                image = image.crop(box)
            else:
                padding = Image.new('L', (self.clip_width, height), 0)
                padding.paste(image, (0, 0))
                image = padding

            image = image.resize((224, 224), resample=Image.BILINEAR)
            img_np = np.array(image)
            img_tensor = torch.from_numpy(img_np).float().unsqueeze(0) / 255.0
            
            return img_tensor, label
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return torch.zeros(1, 224, 224), label