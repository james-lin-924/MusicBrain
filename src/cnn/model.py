import torch
import torch.nn as nn
import torchvision.models as models

class AttentionModule(nn.Module):
    """
    Squeeze-and-Excitation based Attention Module
    (完全保持不變)
    """
    def __init__(self, channels_low, channels_mid, channels_high, reduction=16):
        super(AttentionModule, self).__init__()
        self.total_channels = channels_low + channels_mid + channels_high
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.total_channels, self.total_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.total_channels // reduction, self.total_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, low, mid, high):
        target_size = high.size()[2:] 
        low_resized = nn.functional.interpolate(low, size=target_size, mode='bilinear', align_corners=False)
        mid_resized = nn.functional.interpolate(mid, size=target_size, mode='bilinear', align_corners=False)
        concat_features = torch.cat([low_resized, mid_resized, high], dim=1)
        b, c, _, _ = concat_features.size()
        y = self.avg_pool(concat_features).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return concat_features * y

class AttentionConvNeXt(nn.Module):
    def __init__(self, num_classes=60): # 預設改為 60
        super(AttentionConvNeXt, self).__init__()
        
        base_model = models.convnext_base(weights=None)
        
        original_layer = base_model.features[0][0]
        base_model.features[0][0] = nn.Conv2d(
            in_channels=1, 
            out_channels=original_layer.out_channels,
            kernel_size=original_layer.kernel_size,
            stride=original_layer.stride
        )
        
        self.features = base_model.features
        
        # Blocks 定義 (保持不變)
        self.low_block = nn.Sequential(self.features[0], self.features[1], self.features[2], self.features[3])
        self.mid_block = nn.Sequential(self.features[4], self.features[5])
        self.high_block = nn.Sequential(self.features[6], self.features[7])
        
        # Attention (保持不變)
        self.attention = AttentionModule(256, 512, 1024)
        
        self.total_channels = 256 + 512 + 1024
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 修改: 單一分類頭，輸出 60 類
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.total_channels, 1024),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes) # Output 60
        )

    def forward(self, x):
        low = self.low_block(x)
        mid = self.mid_block(low)
        high = self.high_block(mid)
        
        weighted_features = self.attention(low, mid, high)
        x = self.avg_pool(weighted_features)
        
        # 單一輸出
        out = self.classifier(x)
        
        return out