#!/usr/bin/env python3
"""
下载 SAM3 / SAM3.1 模型权重（需已登录 HuggingFace 并通过 Meta 权重申请）。

使用方式:
    python scripts/download_sam3_weights.py                # 默认 sam3.1，存到 ./sam3_weights/
    python scripts/download_sam3_weights.py --version sam3  # sam3
    python scripts/download_sam3_weights.py -o /path/to/weights  # 指定输出目录

前置条件:
    huggingface-cli login   # 或设置 HF_TOKEN 环境变量
    # 并且你的账号已被 Meta 授权访问 facebook/sam3 / facebook/sam3.1
"""

import argparse
import os
import sys
from pathlib import Path

HF_REPOS = {
    "sam3":   {"repo_id": "facebook/sam3",   "ckpt_name": "sam3.pt"},
    "sam3.1": {"repo_id": "facebook/sam3.1", "ckpt_name": "sam3.1_multiplex.pt"},
}


def check_hf_login() -> str:
    try:
        from huggingface_hub import whoami
        user = whoami()
        return user.get("name", "unknown")
    except Exception:
        print("未检测到 HuggingFace 登录状态。请先执行:", file=sys.stderr)
        print("  huggingface-cli login", file=sys.stderr)
        print("或设置环境变量 HF_TOKEN。", file=sys.stderr)
        sys.exit(1)


def download(version: str, output_dir: str) -> str:
    from huggingface_hub import hf_hub_download

    info = HF_REPOS[version]
    repo_id = info["repo_id"]
    ckpt_name = info["ckpt_name"]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"下载 config.json  from {repo_id} ...")
    hf_hub_download(
        repo_id=repo_id,
        filename="config.json",
        local_dir=output_dir,
    )

    print(f"下载 {ckpt_name}  from {repo_id} ...")
    ckpt_path = hf_hub_download(
        repo_id=repo_id,
        filename=ckpt_name,
        local_dir=output_dir,
    )

    return ckpt_path


def main():
    parser = argparse.ArgumentParser(
        description="下载 SAM3/SAM3.1 模型权重（需已通过 Meta 授权）"
    )
    parser.add_argument(
        "--version", "-v",
        choices=["sam3", "sam3.1"],
        default="sam3.1",
        help="模型版本 (default: sam3.1)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./sam3_weights",
        help="权重存放目录 (default: ./sam3_weights)",
    )
    args = parser.parse_args()

    username = check_hf_login()
    print(f"HuggingFace 已登录: {username}")

    try:
        ckpt_path = download(args.version, args.output_dir)
    except Exception as e:
        msg = str(e)
        if "403" in msg or "gated" in msg.lower() or "access" in msg.lower():
            print("\n下载失败 — 可能是未通过 Meta 的权重访问授权。", file=sys.stderr)
            print(f"请访问 https://huggingface.co/{HF_REPOS[args.version]['repo_id']} 提交申请。", file=sys.stderr)
        else:
            print(f"\n下载失败: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\n下载完成。")
    print(f"  版本:    {args.version}")
    print(f"  权重:    {ckpt_path}")
    print(f"  配置:    {os.path.join(args.output_dir, 'config.json')}")
    print(f"\nSam3BuildConfig 使用:")
    print(f'  Sam3BuildConfig(version="{args.version}", checkpoint_path="{ckpt_path}")')


if __name__ == "__main__":
    main()
