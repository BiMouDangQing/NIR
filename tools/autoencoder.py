"""命令行入口：自编码器训练（兼容保留）。

算法实现已迁移到 ``model/`` 包，本文件仅保留 CLI。

用法：
    python -m tools.autoencoder <path.parquet> [--epochs 30] [--train-ratio 0.8]
"""
import argparse

import torch

from config.settings import LOG_DIR
from model.autoencoder import find_default_file
from model.service import run_training
from tools.logger import setup_logging


def main() -> None:
    setup_logging(LOG_DIR)
    parser = argparse.ArgumentParser(description="秋月梨 NIR 自编码器（自监督预训练）")
    parser.add_argument("path", nargs="?", default=None, help="parquet 文件路径（缺省自动搜索）")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bottleneck", type=int, default=16)
    parser.add_argument("--train-ratio", type=float, default=0.8, help="训练集占比，默认 0.8")
    parser.add_argument("--max-samples", type=int, default=None, help="可选：只取前 N 条用于快速试验")
    parser.add_argument("--out-dir", default="model_output", help="模型与结果输出目录")
    args = parser.parse_args()

    torch.manual_seed(0)
    path = args.path or find_default_file()
    if not path:
        print("[错误] 未找到数据文件，请显式传入路径")
        return

    summary = run_training(
        data_path=path,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        bottleneck=args.bottleneck,
        train_ratio=args.train_ratio,
        max_samples=args.max_samples,
    )
    print(f"\n训练完成，模型已保存到 {summary['out_dir']}")


if __name__ == "__main__":
    main()
