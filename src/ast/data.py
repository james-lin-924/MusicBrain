import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
import warnings

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

class PianoRollDataset(Dataset):
    def __init__(self, root_dir, is_train=True, max_seq_len=2000):
        self.root_dir = root_dir
        self.target_height = 128 
        self.is_train = is_train
        self.max_seq_len = max_seq_len 
        self.patch_w = 10  # 1 Frame = 10 pixels
        
        self.samples = []
        self.source_map = {'Human': 0, 'large': 1, 'melody': 2}
        self.style_map = {}
        
        sorted_sources = sorted(['Human', 'large', 'melody'])
        style_idx = 0
        
        if os.path.exists(root_dir):
            for source_name in sorted_sources:
                source_path = os.path.join(root_dir, source_name)
                if not os.path.isdir(source_path): continue
                for folder_name in sorted(os.listdir(source_path)):
                    class_path = os.path.join(source_path, folder_name)
                    if not os.path.isdir(class_path): continue
                    
                    if folder_name not in self.style_map:
                        self.style_map[folder_name] = style_idx
                        style_idx += 1
                        
                    for img_name in os.listdir(class_path):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.samples.append((os.path.join(class_path, img_name), self.source_map[source_name], self.style_map[folder_name]))

    def __len__(self):
        return len(self.samples)

    def _load_and_augment_single(self, idx):
        """讀取單一檔案並進行基礎強化，回傳 Tensor 與標籤"""
        img_path, src_label, sty_label = self.samples[idx]
        image = Image.open(img_path).convert('L')
        w, h = image.size
        
        if self.is_train:
            scale_factor = random.uniform(0.8, 1.2)
            w = int(w * scale_factor)
            
        image = image.resize((w, self.target_height), resample=Image.BILINEAR)
        img_tensor = torch.from_numpy(np.array(image)).float().unsqueeze(0) / 255.0
        
        if self.is_train:
            # 力度縮放
            if random.random() < 0.5:
                vel_scale = random.uniform(0.7, 1.3)
                img_tensor = torch.clamp(img_tensor * vel_scale, 0.0, 1.0)
            
            # 八度平移與移調
            if random.random() < 0.5:
                octave_shift = random.choice([-12, 0, 12])
                fine_shift = random.randint(-5, 5)
                total_shift = octave_shift + fine_shift
                if total_shift != 0:
                    img_tensor = torch.roll(img_tensor, shifts=total_shift, dims=1)
                    if total_shift > 0: img_tensor[:, :total_shift, :] = 0.0
                    else: img_tensor[:, total_shift:, :] = 0.0
                    
            # 隨機漏音
            if random.random() < 0.5:
                active_notes = img_tensor > 0
                dropout_mask = active_notes & (torch.rand_like(img_tensor) < 0.05)
                img_tensor[dropout_mask] = 0.0

        return img_tensor, src_label, sty_label

    def __getitem__(self, idx):
        target_len = self.max_seq_len
        current_len = 0
        
        img_pieces = []
        src_pieces = []
        sty_pieces = []
        
        # 確保裁切都是 10 pixels (1 Frame) 的整數倍，最短限制為 100 pixels (10 Frames)
        min_crop_frames = 10 
        
        # 🌟 核心拼接迴圈：直到總長度達到 max_seq_len (2000 pixels)
        while current_len < target_len:
            # 第一段使用當前 idx，後續隨機抽取其他曲子來拼接
            curr_idx = idx if current_len == 0 else random.randint(0, len(self.samples) - 1)
            img_tensor, src_label, sty_label = self._load_and_augment_single(curr_idx)
            
            actual_w = img_tensor.shape[2]
            actual_frames = actual_w // self.patch_w
            
            # 決定這段要切多長
            if actual_frames <= min_crop_frames:
                crop_frames = actual_frames
                start_frame = 0
            else:
                max_possible_frames = min(actual_frames, (target_len - current_len) // self.patch_w)
                if max_possible_frames < min_crop_frames:
                    crop_frames = max_possible_frames
                else:
                    crop_frames = random.randint(min_crop_frames, max_possible_frames)
                start_frame = random.randint(0, actual_frames - crop_frames)
            
            if crop_frames <= 0: continue
                
            start_px = start_frame * self.patch_w
            crop_px = crop_frames * self.patch_w
            
            # 裁切圖像
            img_crop = img_tensor[:, :, start_px : start_px + crop_px]
            img_pieces.append(img_crop)
            
            # 給這段裁切賦予 Frame-level 的標籤
            src_pieces.append(torch.full((crop_frames,), src_label, dtype=torch.long))
            sty_pieces.append(torch.full((crop_frames,), sty_label, dtype=torch.long))
            
            current_len += crop_px
            
        # 組合所有片段
        final_img = torch.cat(img_pieces, dim=2)
        
        # 截斷多餘的部分，確保剛好是 max_seq_len
        final_img = final_img[:, :, :target_len]
        final_src = torch.cat(src_pieces, dim=0)[: target_len // self.patch_w]
        final_sty = torch.cat(sty_pieces, dim=0)[: target_len // self.patch_w]
        
        # SpecAugment (在拼接完成的整張圖上執行)
        if self.is_train:
            if random.random() < 0.5:
                mask_h = random.randint(1, 12)
                h_start = random.randint(0, max(1, self.target_height - mask_h))
                final_img[:, h_start : h_start + mask_h, :] = 0.0
            if random.random() < 0.5:
                mask_w = random.randint(10, 50)
                w_start = random.randint(0, max(1, target_len - mask_w))
                final_img[:, :, w_start : w_start + mask_w] = 0.0

        # 無特徵判定 (第 21 類)
        frames = final_img.squeeze(0).view(128, -1, self.patch_w)
        energy = frames.sum(dim=(0, 2))
        featureless_mask = energy < 2.0 
        final_sty[featureless_mask] = 20
        
        return final_img, final_src, final_sty