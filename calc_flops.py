import torch
from thop import profile
from model_crossattn import MaxMViT_MLP_CrossAttn

def calc():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MaxMViT_MLP_CrossAttn(num_classes=4, hidden_size=512, dropout_rate=0.3)
    model.to(device)
    model.eval()
    
    # Create dummy inputs [B, C, H, W]
    cqt = torch.randn(1, 1, 224, 224).to(device)
    mel = torch.randn(1, 1, 224, 224).to(device)
    
    macs, params = profile(model, inputs=(cqt, mel))
    
    print(f"Total MACs (Giga): {macs / 1e9:.2f} G")
    print(f"Total FLOPs (Giga): {(macs * 2) / 1e9:.2f} G") # 1 MAC = 2 FLOPs
    print(f"Total Parameters: {params / 1e6:.2f} M")
    
if __name__ == '__main__':
    calc()
