
import os
import torch
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.nn.functional as F
from functools import partial

# ====== Project Modules ======
from dataset import MVTecDataset, get_data_transforms
from g2l_modules.g2l_model import G2LHyD, DALRBlock
from g2l_modules import encoder
from g2l_modules.gfsr_blocks import GFSRBlock, LinearAttention
from g2l_modules.gfsr_blocks import CGB, bMlp, SAB
from dinomaly_visa_uni_fewshot import ResidualScale, PostNorm
from utils import cal_anomaly_maps, min_max_norm, cvt2heatmap, show_cam_on_image, get_gaussian_kernel, create_custom_colormap

def make_decoder(kind, dim, heads, pattern=None, depth=8, num_reg_tokens=4):
    blocks = []
    if kind == 'vit':
        for _ in range(depth):
            blocks.append(
                GFSRBlock(
                    dim=dim,
                    num_heads=heads,
                    mlp_ratio=4.,
                    qkv_bias=True,
                    norm_layer=partial(torch.nn.LayerNorm, eps=1e-8),
                    attn=LinearAttention
                )
            )
    elif kind == 'mamba':
        for _ in range(depth):
            blk = DALRBlock(embed_dim=dim, num_register_tokens=num_reg_tokens)
            blk = ResidualScale(blk, init_scale=0.2)
            blk = PostNorm(blk, dim)
            blocks.append(blk)
    else:  # hybrid
        pat = [s.strip().lower() for s in (pattern or 'v,m,v,m,v,m,v,m').split(',')]
        assert len(pat) == depth
        for p in pat:
            if p == 'm':
                blk = DALRBlock(embed_dim=dim, num_register_tokens=num_reg_tokens)
                blk = ResidualScale(blk, init_scale=0.2)
                blk = PostNorm(blk, dim)
            elif p == 'v':
                blk = GFSRBlock(
                    dim=dim,
                    num_heads=heads,
                    mlp_ratio=4.,
                    qkv_bias=True,
                    norm_layer=partial(torch.nn.LayerNorm, eps=1e-8),
                    attn=LinearAttention
                )
            else:
                raise ValueError(f"Unknown marker: {p}")
            blocks.append(blk)
    return torch.nn.ModuleList(blocks)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ========== Model ==========
    encoder = encoder.load(args.encoder)
    num_reg_tokens = 4

    if 'small' in args.encoder:
        embed_dim, num_heads = 384, 6
    elif 'base' in args.encoder:
        embed_dim, num_heads = 768, 12
    elif 'large' in args.encoder:
        embed_dim, num_heads = 1024, 16
    else:
        raise RuntimeError("Architecture not in small/base/large.")

    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    bottleneck = torch.nn.ModuleList([
        SAB(embed_dim, hidden_features=embed_dim * 2, out_features=embed_dim,
            drop=0.1, grad=1.0, dw_kernel=5)
    ])

    decoder = make_decoder(args.decoder, embed_dim, num_heads,
                           pattern=args.hybrid_pat, depth=args.decode_depth,
                           num_reg_tokens=num_reg_tokens)

    model = G2LHyD(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
        target_layers=target_layers,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder
    )

    # Load trained weights
    if os.path.exists(args.model_path):
        # The training script saves the state_dict of the `trainable` module list.
        # So we load the weights into the bottleneck and decoder.
        state_dict = torch.load(args.model_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict['model'], strict=False)
        print(f"Loaded model from {args.model_path}")
    else:
        print(f"Error: Model path '{args.model_path}' not found. Please provide a valid path.")
        return

    model.to(device)
    model.eval()

    # ========== Dataset ==========
    data_transform, gt_transform = get_data_transforms(args.image_size, args.crop_size)
    
    # VISA classes
    item_list = [
        'candle', 'capsules', 'cashew', 'chewinggum',
        'fryum', 'macaroni1', 'macaroni2',
        'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'
    ]

    # ========== Visualization ==========
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    
    # 使用默认的 viridis colormap（与 Figure_A2 风格一致）

    for item in item_list:
        test_path = os.path.join(args.data_path, item)
        if not os.path.exists(test_path):
            print(f"Warning: Data path for class '{item}' not found at '{test_path}'. Skipping.")
            continue
            
        dataset = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        save_dir_class = os.path.join(args.save_dir, item)
        os.makedirs(save_dir_class, exist_ok=True)

        count = 0
        for img, gt, label, img_path in tqdm(dataloader, desc=f"Visualizing {item}"):
            if count >= 10:
                break

            img = img.to(device)

            with torch.no_grad():
                en, de = model(img)
                anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
                anomaly_map = gaussian_kernel(anomaly_map)

            anomaly_map = anomaly_map.cpu().numpy()
            
            for i in range(anomaly_map.shape[0]):
                if count >= 10:
                    break

                heatmap = min_max_norm(anomaly_map[i, 0])
                heatmap = cvt2heatmap(heatmap * 255)  # 默认使用 viridis colormap
                
                # Load original image for visualization
                vis_img = cv2.imread(img_path[i])
                vis_img = cv2.resize(vis_img, (args.crop_size, args.crop_size))

                hm_on_img = show_cam_on_image(vis_img, heatmap)

                # Save images
                fname = os.path.basename(img_path[i])
                cv2.imwrite(os.path.join(save_dir_class, fname), vis_img)
                cv2.imwrite(os.path.join(save_dir_class, f"heatmap_{fname}"), heatmap)
                cv2.imwrite(os.path.join(save_dir_class, f"overlay_{fname}"), hm_on_img)
                
                count += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Dinomaly Visualization", add_help=True)
    parser.add_argument("--model_path", type=str, default="backbones/mvtec_20000.pth", help="Path to the trained model checkpoint.")
    parser.add_argument('--data_path', type=str, default='data/VisA_pytorch/1cls')
    parser.add_argument('--save_dir', type=str, default='./visualization_results')
    parser.add_argument('--image_size', type=int, default=448)
    parser.add_argument('--crop_size', type=int, default=392)
    parser.add_argument('--encoder', type=str, default='dinov2reg_vit_base_14')
    parser.add_argument('--decoder', type=str, default='mamba', choices=['vit', 'mamba', 'hybrid'])
    parser.add_argument('--decode_depth', type=int, default=8)
    parser.add_argument('--hybrid_pat', type=str, default='v,m,v,m,v,m,v,m')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    main(args)
