#!/usr/bin/env python3
import os, sys, argparse, zipfile, glob, json, shutil
from pathlib import Path
from collections import Counter

def log(*a): print("[prep]", *a)

def find_zip(root: Path, given: str|None):
    if given:
        z = Path(given)
        if not z.exists(): raise FileNotFoundError(f"zip 不存在: {z}")
        return z
    cand = list(root.glob("realiad_jsons.zip"))
    if not cand:
        # 兜底找一切可能的 json 压缩包
        cand = list(root.glob("*.zip"))
        cand = [p for p in cand if "json" in p.name.lower()]
    if not cand:
        raise FileNotFoundError(f"在 {root} 下没有找到 json 压缩包")
    return cand[0]

def safe_symlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        try:
            if dst.resolve() == src.resolve():  # 已经是正确目标
                return "exists"
        except Exception:
            pass
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    try:
        os.symlink(src, dst, target_is_directory=True)
        return "symlink"
    except Exception as e:
        # 某些环境禁用链接 => 复制
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        return "copy"

def unzip_to(zip_path: Path, dst_dir: Path):
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dst_dir)

def build_alias(root: Path, json_pack: str):
    """
    创建 <root>/realiad_jsons/realiad_jsons -> <root>/<json_pack> 的别名
    支持两种情况：
      1) <root>/realiad_jsons/*.json          （别名指向 <root>/realiad_jsons 本身）
      2) <root>/<json_pack>/*.json            （别名指向这个 pack）
    """
    pack_dir = root / json_pack
    if not pack_dir.exists():
        raise FileNotFoundError(f"JSON 方案目录不存在: {pack_dir}")
    # 目标别名目录
    outer = root / "realiad_jsons"
    alias = outer / "realiad_jsons"
    outer.mkdir(parents=True, exist_ok=True)

    # 情形：如果选的是默认 pack==realiad_jsons，本质上就是 “在 realiad_jsons 里链接自己”
    target = pack_dir
    action = safe_symlink(target, alias)
    log(f"alias 建立完成: {alias} -> {target} [{action}]")

    # 基本检查：至少应该有若干 .json
    js = list(alias.glob("*.json"))
    if not js:
        raise RuntimeError(f"别名目录里没找到 json：{alias}")

def quick_check(root: Path, res: int, sample_k=3):
    """
    随机抽几个类，检查 json 中的 image_path/mask_path 是否能在
    <root>/realiad_{res}/{category}/ 下找到
    """
    json_dir = root / "realiad_jsons" / "realiad_jsons"
    img_root = root / f"realiad_{res}"
    if not img_root.exists():
        raise FileNotFoundError(f"图片目录不存在: {img_root}")
    js_files = sorted(json_dir.glob("*.json"))
    if not js_files:
        raise FileNotFoundError(f"没有找到 json：{json_dir}")

    misses = Counter()
    checked = 0
    for jf in js_files[:sample_k]:
        category = jf.stem
        data = json.loads(Path(jf).read_text(encoding="utf-8"))
        for split in ("train", "test"):
            if split not in data: continue
            # 抽头两个样本检查路径
            for sample in data[split][:2]:
                p_img = img_root / category / sample["image_path"]
                if not p_img.exists():
                    misses["image"] += 1
                if sample.get("anomaly_class") != "OK":
                    p_msk = img_root / category / sample.get("mask_path","")
                    if not p_msk.exists():
                        misses["mask"] += 1
            checked += 1
    if misses:
        log(f"路径缺失：{dict(misses)}  （可能是 res 选错或数据未完整下载）")
    else:
        log(f"抽查通过：{checked} 条记录的图像/掩码路径均存在。")

def main():
    ap = argparse.ArgumentParser("Prepare Real-IAD data")
    ap.add_argument("--root", default="/data/Anomaly_Detection/Real-IAD",
                    help="Real-IAD 顶层目录")
    ap.add_argument("--zip", default=None,
                    help="realiad_jsons.zip 的路径（不填则在 root 下自动查找）")
    ap.add_argument("--json-pack", default="realiad_jsons",
                    help="选择使用哪个 JSON 方案目录（如 realiad_jsons、realiad_jsons_fuiad_0.2 等）")
    ap.add_argument("--res", type=int, default=1024, choices=[256,512,1024],
                    help="使用的图像分辨率子目录")
    ap.add_argument("--no-check", action="store_true", help="跳过快速自检")
    args = ap.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    # 1) 解压 json 包
    zp = find_zip(root, args.zip)
    log(f"解压 {zp} -> {root}")
    unzip_to(zp, root)

    # 2) 建立别名目录 realiad_jsons/realiad_jsons
    build_alias(root, args.json_pack)

    # 3) 快速自检
    if not args.no_check:
        quick_check(root, args.res)

    # 4) 提示下一步命令
    cmd = f"""python test_realiad.py \\
  --data_path {root} \\
  --res {args.res} \\
  --cuda 1 --shots 1 --batch_size 8 --workers 4 \\
  --decoder hybrid --hybrid_pat v,v,v,v,m,m,m,m --decode_depth 8 \\
  --bn_mlp_ratio 2.67 --bn_drop 0.2 \\
  --topk_ratio 0.032 --fg_q 64 --gem_p 8 \\
  --ms_tta "0.75,1.0,1.25" --rot_tta none --z_norm \\
  --eval_every 1000 --save_name real_IAD"""
    log("✅ 预处理完成。可直接运行：\n" + cmd)

if __name__ == "__main__":
    main()
