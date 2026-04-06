# Dinomaly/models_mambaAD/runners/train_fewshot_mambaad.py
import os, argparse, time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from dalr.fewshot_g2l import FewShotG2L, _global_vecs
from dalr.hybrid_losses import hybrid_loss

# ====== 你项目里的 encoder 加载（示例） ======
def build_encoder(name='dinov2_s14'):
    """
    请在这里对接你的 DINOv2 编码器（需要能输出3个尺度特征）。
    你可以封装一个 wrapper 带 extract_features(x)->[f1,f2,f3]
    """
    from backbones import encoder  # 例如你项目里的模块；若名称不同请改
    enc = encoder.load_dinov2_small_14()  # 示例
    enc.eval()
    return enc, [384, 768, 1024]              # 按实际通道返回

# ====== 构建数据 ======
def build_loaders(data_root, img_size=256, batch_size=16, shots=1):
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    # 兜底：ImageFolder，子目录为类名；若你有专用 Dataset，可替换
    train_set = datasets.ImageFolder(os.path.join(data_root, 'train'), transform=tfm)
    test_set  = datasets.ImageFolder(os.path.join(data_root, 'test'),  transform=tfm)

    # few-shot memory：每类取 shots 张
    cls_to_idx = {}
    mem_imgs = []
    count = {}
    for img, label in train_set:
        if count.get(label,0) < shots:
            mem_imgs.append(img.unsqueeze(0))
            count[label] = count.get(label,0) + 1
        if all(c>=shots for c in count.values()) and len(count)==len(train_set.classes):
            break
    mem_batch = torch.cat(mem_imgs, dim=0) if mem_imgs else None

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader, mem_batch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, required=True)
    ap.add_argument('--img_size', type=int, default=256)
    ap.add_argument('--shots', type=int, default=1)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--alpha', type=float, default=0.7)  # contrast
    ap.add_argument('--beta', type=float, default=0.3)   # recon
    ap.add_argument('--sim_w', type=float, default=0.5)
    ap.add_argument('--rec_w', type=float, default=0.5)
    ap.add_argument('--save', type=str, default='saved_results/mambaad.pth')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder, feat_dims = build_encoder()
    model = FewShotG2L(encoder, feat_dims, sim_weight=args.sim_w, rec_weight=args.rec_w).to(device)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)

    train_loader, test_loader, mem_batch = build_loaders(args.data_root, args.img_size, args.batch_size, args.shots)
    # build memory
    if mem_batch is not None:
        with torch.no_grad():
            model.build_memory(mem_batch.to(device))

    # train
    for epoch in range(1, args.epochs+1):
        model.train()
        t0 = time.time()
        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            feats, dec_feats = model(_
