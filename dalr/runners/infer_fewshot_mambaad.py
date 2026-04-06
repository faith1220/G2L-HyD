# Dinomaly/models_mambaAD/runners/infer_fewshot_mambaad.py
import os, argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
from dalr.fewshot_g2l import FewShotG2L
from dalr.utils.viz import save_heatmap

def build_encoder():
    from backbones import encoder
    enc = encoder.load_dinov2_small_14()
    enc.eval()
    return enc, [384, 768, 1024]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, required=True)
    ap.add_argument('--img_size', type=int, default=256)
    ap.add_argument('--shots', type=int, default=1)
    ap.add_argument('--model', type=str, required=True)
    ap.add_argument('--out_dir', type=str, default='saved_results/infer')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tfm = transforms.Compose([transforms.Resize((args.img_size,args.img_size)), transforms.ToTensor()])

    # 加载数据
    train_set = datasets.ImageFolder(os.path.join(args.data_root, 'train'), transform=tfm)
    test_set  = datasets.ImageFolder(os.path.join(args.data_root, 'test'),  transform=tfm)
    train_loader = DataLoader(train_set, batch_size=args.shots, shuffle=True)
    test_loader  = DataLoader(test_set,  batch_size=1, shuffle=False)

    encoder, feat_dims = build_encoder()
    model = FewShotG2L(encoder, feat_dims).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # memory
    with torch.no_grad():
        imgs,_ = next(iter(train_loader))
        model.build_memory(imgs.to(device))

    os.makedirs(args.out_dir, exist_ok=True)
    for i,(img,_) in enumerate(test_loader):
        img = img.to(device)
        with torch.no_grad():
            feats, dec_feats = model(img)
            score = model.anomaly_score(feats, dec_feats)  # [1,1,H,W]
        s = score[0,0].cpu().numpy()
        # 假设你有原图 BGR
        im = (img[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)[:, :, ::-1]
        save_heatmap(im, s, os.path.join(args.out_dir, f"heat_{i:05d}.png"))

    print("Done. Saved to", args.out_dir)

if __name__ == "__main__":
    main()
