import os
from utils import create_sequences, go_back_to_original_shape
from datasets import read_dataset, read_dataset_semisupervised

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from AE_LSTM import AE_LSTM
from VAE_LSTM import VAE_LSTM
from SAE_LSTM import SAE_LSTM
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
from loss_history import LossHistory
from sklearn.metrics import roc_auc_score
from pathlib import Path
import pandas as pd


def train_model(model, dataset, epoch=5):
    if model not in ["AE_LSTM", "VAE_LSTM", "SAE_LSTM"]:
        raise ValueError("Unsuported model")

    finished_model_path = Path(f"finished/{model}_{dataset}/final_model.ckpt")
    if finished_model_path.exists():
        print("Model already trained. Exiting.")
        return

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")

    #data setup
    X_train, y_train = np.array([]), np.array([])
    if model == "SAE_LSTM":
        for _, value in read_dataset_semisupervised(dataset).items():
            X_train = np.concatenate([X_train, value[0]], axis=0) if X_train.size else value[0]
            y_train = np.concatenate([y_train, value[1]], axis=0) if y_train.size else value[1]
    else:
        for _, value in read_dataset(dataset).items():
            X_train = np.concatenate([X_train, value[0]], axis=0) if X_train.size else value[0]
        y_train = np.array([0]*X_train.shape[0])

    seq_len = 50

    X_sequences_train = torch.from_numpy(X_train.astype(np.float32)).unfold(0, size=seq_len, step=1).permute(0,2,1)
    y_sequences_train = torch.from_numpy(y_train.astype(np.float32)).unfold(0, size=seq_len, step=1)

    train_dataset = TensorDataset(X_sequences_train, y_sequences_train)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=False, num_workers=4
    )

    if model == "AE_LSTM":
        checkpoint_path = f"checkpoints/AE_LSTM_{dataset}/last.ckpt"
        if Path(checkpoint_path).exists():
            autoencoder = AE_LSTM.load_from_checkpoint(
               checkpoint_path
            ).to(device)
        else:
            autoencoder = AE_LSTM(
                seq_len=seq_len, n_features=X_train.shape[1], hidden_size=64, num_layers=1
                ).to(device)
    elif model == "VAE_LSTM":
        checkpoint_path = f"checkpoints/VAE_LSTM_{dataset}/last.ckpt"
        if Path(checkpoint_path).exists():
            autoencoder = VAE_LSTM.load_from_checkpoint(
               checkpoint_path
            ).to(device)
        else:
            autoencoder = VAE_LSTM(seq_len=seq_len, n_features=X_train.shape[1], hidden_size=64, latent_dim=32).to(device)
    elif model == "SAE_LSTM":
        checkpoint_path = f"checkpoints/SAE_LSTM_{dataset}/last.ckpt"
        if Path(checkpoint_path).exists():
            autoencoder = SAE_LSTM.load_from_checkpoint(
               checkpoint_path
            ).to(device)
        else:
            autoencoder = SAE_LSTM(seq_len=seq_len, n_features=X_train.shape[1], hidden_size=64, num_layers=1
                ).to(device)

    early_stopping_callback = EarlyStopping(
        monitor="train_loss",
        patience=5,
        mode="min",
    )

    loss_history_callback = LossHistory()

    checkpoint_cb = ModelCheckpoint(
        dirpath=f"checkpoints/{model}_{dataset}",
        filename="{epoch}",
        save_top_k=1,
        every_n_epochs=1,
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=epoch,
        callbacks=[early_stopping_callback, loss_history_callback, checkpoint_cb],
        accelerator="auto",
        
    )
    trainer.fit(autoencoder, train_dataloader,ckpt_path=checkpoint_path if os.path.exists(checkpoint_path) else None)

    # Save final model
    trainer.save_checkpoint(finished_model_path)

def test_model(model, dataset_name, smisupervised = False):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")

    result_path = Path(f"time_series_ad_results/{model}.csv")
    result_df = pd.read_csv(result_path) if result_path.exists() else pd.DataFrame(columns=["dataset", "roc_auc", "test_set"])
    dataset_results = result_df[result_df["dataset"] == dataset_name]
    if smisupervised and "semisupervised_test" in dataset_results["test_set"].values:
        print("Semisupervised test results already exist. Exiting.")
        return result_df
    if not smisupervised and "unsupervised_test" in dataset_results["test_set"].values:
        print("Unsupervised test results already exist. Exiting.")
        return result_df

    #Load model from checkpoint
    if model == "AE_LSTM":
        autoencoder = AE_LSTM.load_from_checkpoint(
           f"finished/AE_LSTM_{dataset_name}/final_model.ckpt"
        ).to(device)
    elif model == "VAE_LSTM":
        autoencoder = VAE_LSTM.load_from_checkpoint(
           f"finished/VAE_LSTM_{dataset_name}/final_model.ckpt"
        ).to(device)
    elif model == "SAE_LSTM":
        autoencoder = SAE_LSTM.load_from_checkpoint(
           f"finished/SAE_LSTM_{dataset_name}/final_model.ckpt"
        ).to(device)

    seq_len = autoencoder.seq_len

    trainer = pl.Trainer(
        accelerator="auto",
    )

    X_test = []
    y_test = []
    if smisupervised:
        dataset = read_dataset_semisupervised(dataset_name)
        for key, value in dataset.items():
            X_test.append(value[2])
            y_test.append(value[3])
    else:
        dataset = read_dataset(dataset_name)
        for key, value in dataset.items():
            X_test.append(value[1])
            y_test.append(value[2])

    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    sequences_test = torch.from_numpy(X_test.astype(np.float32)).unfold(0, size=seq_len, step=1).permute(0,2,1)
    test_dataset = TensorDataset(sequences_test)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    trainer.test(autoencoder, test_dataloader)
    test_errors = np.concatenate(autoencoder.test_errors, axis=0)
    test_errors = go_back_to_original_shape(
        test_errors, X_test.shape[0], seq_step=1
    )
    roc_auc = roc_auc_score(y_test, test_errors)
    print(f"ROC-AUC Semisupervised Test Set Score: {roc_auc}")

    semisupervised_test_set_str = "semisupervised_test" if smisupervised else "unsupervised_test"
    new_row = {
        "dataset": dataset_name,
        "roc_auc": roc_auc,
        "test_set": semisupervised_test_set_str,
    }
    result_df = pd.concat([result_df, pd.DataFrame([new_row])], ignore_index=True)

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(result_path, index=False)
    return result_df

datasets = ["MSL","MITDB","GHL","SMD","SVDB","OPPORTUNITY","CATSv2","SMAP", ]
models = ["AE_LSTM", "VAE_LSTM", "SAE_LSTM"]
import itertools
for model, dataset in itertools.product(models, datasets):
    print(f"Running experiment for model: {model}, dataset: {dataset}")
    train_model(model, dataset, epoch=50)
    test_model(model, dataset,False)
    test_model(model, dataset,True)
