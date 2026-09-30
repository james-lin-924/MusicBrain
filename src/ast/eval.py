import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, accuracy_score
import warnings

warnings.filterwarnings("ignore")

from data import PianoRollDataset
from model import TemporalDualHeadTransformer

# 參數設定
CONFIG = {
    'root_dir': '../Datas',
    'batch_size': 16,
    'num_workers': 4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'model_path': 'best_temporal_smooth_model_large.pth',
    'max_seq_len': 2000,
}

def get_file_level_predictions(out_logits, original_labels):
    probs = torch.softmax(out_logits, dim=-1)
    
    # 過濾掉 index 20 (featureless)
    if probs.shape[-1] == 21:
        valid_probs = probs[:, :, :20].mean(dim=1) 
    else:
        valid_probs = probs.mean(dim=1)
        
    _, preds = torch.max(valid_probs, dim=1)
    true_labels = original_labels[:, 0]
    return preds.cpu().numpy(), true_labels.cpu().numpy()

def evaluate():
    print("-> 載入資料集...")
    # 若要測試數據大一點，您可以將此處指向包含大量測試資料的資料夾
    test_dataset = PianoRollDataset(root_dir=CONFIG['root_dir'], is_train=False, max_seq_len=CONFIG['max_seq_len'])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])

    print("-> 載入模型...")
    model = TemporalDualHeadTransformer(
        num_styles=21, max_seq_len=200, d_model=256, nhead=8, num_layers=6
    ).to(CONFIG['device'])
    
    model.load_state_dict(torch.load(CONFIG['model_path'], map_location=CONFIG['device'], weights_only=True))
    model.eval()

    all_src_preds, all_src_trues = [], []
    all_sty_preds, all_sty_trues = [], []

    print("-> 開始推論...")
    with torch.no_grad():
        for inputs, src_labels, sty_labels in test_loader:
            inputs = inputs.to(CONFIG['device'])
            
            with torch.amp.autocast('cuda'):
                out_src, out_sty = model(inputs)
            
            # 取得 File-level 預測與真實標籤
            p_src, t_src = get_file_level_predictions(out_src, src_labels)
            p_sty, t_sty = get_file_level_predictions(out_sty, sty_labels)
            
            all_src_preds.extend(p_src)
            all_src_trues.extend(t_src)
            all_sty_preds.extend(p_sty)
            all_sty_trues.extend(t_sty)

    # 輸出報告
    print("\n" + "="*50)
    print(">>> 來源 (Source) 分類報告 (3類) <<<")
    print("="*50)
    print(classification_report(all_src_trues, all_src_preds, target_names=['Human', 'large', 'melody'], digits=4))
    
    print("\n" + "="*50)
    print(">>> 風格 (Style) 分類報告 (20類) <<<")
    print("="*50)
    print(classification_report(all_sty_trues, all_sty_preds, digits=4))

if __name__ == '__main__':
    evaluate()