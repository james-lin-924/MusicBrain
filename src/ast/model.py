import torch
import torch.nn as nn

class TemporalDualHeadTransformer(nn.Module):
    # 【加深加寬】d_model 提升至 256，層數提升至 6，頭數提升至 8
    def __init__(self, num_sources=3, num_styles=21, d_model=256, nhead=8, num_layers=6, max_seq_len=200):
        super(TemporalDualHeadTransformer, self).__init__()
        
        self.d_model = d_model
        
        # 【更動 1】動態決定特徵融合層：自動抽取第一層、中間層與最後一層
        self.extract_layers = list(set([0, num_layers // 2, num_layers - 1]))
        num_extracted = len(self.extract_layers)
        
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=d_model, kernel_size=(128, 10), stride=(128, 10)),
            nn.BatchNorm2d(d_model),
            nn.GELU()
        )
        
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        # 【更動 2】調降 Dropout：大型模型不需要過高的 Dropout，否則難以收斂
        self.pos_drop = nn.Dropout(p=0.1) 
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, 
                dropout=0.1, batch_first=True, activation='gelu' # 內部 Dropout 降至 0.1
            ) for _ in range(num_layers)
        ])
        
        self.source_fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.3), # Dropout 從 0.6 降至 0.3
            nn.Linear(d_model, num_sources)
        )
        
        # 動態對應抽取出的特徵數量
        self.style_fc = nn.Sequential(
            nn.LayerNorm(d_model * num_extracted),
            nn.Dropout(0.3),
            nn.Linear(d_model * num_extracted, d_model),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(d_model, num_styles)
        )
        
        self.style_temporal_smoother = nn.Conv1d(
            in_channels=num_styles, out_channels=num_styles, 
            kernel_size=5, padding=2, groups=num_styles 
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2) 
        
        T = x.shape[1]
        x = x + self.pos_embedding[:, :T, :]
        x = self.pos_drop(x)
        
        features = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # 動態抽取指定層數的特徵
            if i in self.extract_layers: 
                features.append(x)
                
        out_src_logits = self.source_fc(x) 
        
        # 將抽取出來的多層特徵進行拼接
        combined_style_features = torch.cat(features, dim=-1) 
        raw_sty_logits = self.style_fc(combined_style_features) 
        
        raw_sty_logits_t = raw_sty_logits.transpose(1, 2)
        smoothed_sty_logits_t = self.style_temporal_smoother(raw_sty_logits_t)
        out_sty_logits = smoothed_sty_logits_t.transpose(1, 2)
        
        return out_src_logits, out_sty_logits