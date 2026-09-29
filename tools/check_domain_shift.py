"""核查预训练样本与标签样本的分布一致性（域偏移 / domain shift 检测）。

背景：自监督预训练用的是大量无标签光谱（如 4.5 万样本），有监督微调用的
是少量带真实标签的光谱（如 1596 样本）。若两批数据分布不一致（预训练覆盖
不到标签样本所在的区域），预训练学到的特征在下游就会退化。

本脚本对两批数据的光谱做 6 项检查：
  1. 逐波长均值 / 标准差对比（平均光谱是否重合）
  2. PCA 联合投影（两批质心在低维空间的距离）
  3. 马氏距离（标签样本是否落在预训练分布内部）
  4. KNN 最近邻覆盖度（标签样本离预训练样本有多远）
  5. 领域判别器（能否把两批样本区分开，AUC 越接近 0.5 越同分布）
  6. 标签分布对比（预训练 PredictedValue vs 标签 RealValue）

仅依赖 numpy / pandas / pyarrow，与项目保持一致（无 sklearn）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from model.domain_shift import run_domain_shift


def main() -> None:
    ap = argparse.ArgumentParser(description="核查预训练与标签样本的分布一致性")
    ap.add_argument("--pretrain", default="data/Spectrum_Data_20260928_144425.parquet")
    ap.add_argument("--label", default="data/秋月梨光谱数据_RealValue.parquet")
    ap.add_argument("--spectrum-column", default="SpectrumData")
    ap.add_argument("--n-comp", type=int, default=20, help="马氏距离/判别器用的 PC 维度")
    ap.add_argument("--max-pretrain", type=int, default=0, help="预训练抽样上限，0=全量")
    ap.add_argument("--preprocess", default="", help="对比前预处理，逗号分隔如 smooth,msc,snv")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    preprocess = None
    if args.preprocess:
        preprocess = {k.strip(): True for k in args.preprocess.split(",") if k.strip()}

    report = run_domain_shift(
        pretrain_path=args.pretrain,
        label_path=args.label,
        spectrum_column=args.spectrum_column,
        preprocess=preprocess,
        max_pretrain=args.max_pretrain,
        n_comp=args.n_comp,
        log=print,
    )

    print()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out_json:
        Path(args.out_json).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"报告已保存: {args.out_json}")


if __name__ == "__main__":
    main()
