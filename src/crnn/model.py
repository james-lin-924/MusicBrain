import torch
import torch.nn as nn

class TemporalDualHeadTransformer(nn.Module):
    def __init__(self, num_sources=3, num_styles=21, d_model=384, nhead=8, num_layers=6, max_seq_len=250):
        super(TemporalDualHeadTransformer, self).__init__()
        
        self.d_model = d_model
        
        # 1D Conv 在時間軸上降採樣，完全保留 128 維的音高特徵
        self.patch_embedding = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=d_model, kernel_size=10, stride=10),
            nn.BatchNorm1d(d_model),
            nn.GELU()
        )
        
        # Bi-LSTM 提供長距離時間覆蓋
        self.lstm = nn.LSTM(
            input_size=d_model, 
            hidden_size=d_model // 2, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True, 
            dropout=0.4
        )
        
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        self.pos_drop = nn.Dropout(p=0.4) 
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, 
                dropout=0.4, batch_first=True, activation='gelu'
            ) for _ in range(num_layers)
        ])
        
        # Source Head
        self.source_fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.5),
            nn.Linear(d_model, num_sources)
        )
        
        # Style Head
        self.style_fc = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Dropout(0.6),
            nn.Linear(d_model * 3, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(d_model // 2, num_styles)
        )
        
        # 在 Logits 空間進行時序平滑
        self.style_temporal_smoother = nn.Conv1d(
            in_channels=num_styles, out_channels=num_styles, 
            kernel_size=5, padding=2, groups=1
        )

    def forward(self, x):
        B = x.shape[0]
        x = x.squeeze(1) 
        
        x = self.patch_embedding(x)
        x = x.transpose(1, 2) 
        x, _ = self.lstm(x)
        
        T = x.shape[1]
        x = x + self.pos_embedding[:, :T, :]
        x = self.pos_drop(x)
        
        features = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in [1, 3, 5]: features.append(x)
                
        # Source 預測
        out_src_logits = self.source_fc(x.detach()) 
        
        # Style 預測
        combined_style_features = torch.cat(features, dim=-1) 
        raw_sty_logits = self.style_fc(combined_style_features) 
        
        raw_sty_logits_t = raw_sty_logits.transpose(1, 2)
        smoothed_sty_logits_t = self.style_temporal_smoother(raw_sty_logits_t)
        out_sty_logits = smoothed_sty_logits_t.transpose(1, 2)
        
        return out_src_logits, out_sty_logits