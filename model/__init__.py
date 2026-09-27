"""model 包：算法模型（自监督预训练、特征学习、降维、异常检测）。"""
from model.anomaly import detect, threshold, top_anomalies
from model.autoencoder import (
    Autoencoder,
    encode_all,
    find_default_file,
    get_device,
    load_model,
    parse_spectra,
    reconstruction_error,
    save_model,
    standardize,
    train_ae,
    train_val_split,
)
from model.reduce import pca
from model.service import run_prediction, run_training, run_visualize

__all__ = [
    "Autoencoder",
    "get_device",
    "parse_spectra",
    "standardize",
    "train_ae",
    "train_val_split",
    "encode_all",
    "reconstruction_error",
    "save_model",
    "load_model",
    "find_default_file",
    "pca",
    "detect",
    "threshold",
    "top_anomalies",
    "run_training",
    "run_prediction",
    "run_visualize",
]
