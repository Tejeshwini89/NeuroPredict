from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import config
from .data import (
    load_anomaly_windows,
    load_series,
    make_anomaly_labels,
    make_windows,
    split_and_scale,
)
from .model import TransformerForecaster


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)


def train_model(
    model,
    train_loader,
    val_loader,
    epochs,
    learning_rate,
    weight_decay,
):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            pred = model(x)

            loss = criterion(pred, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        val_losses = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)

                val_losses.append(
                    criterion(model(x), y).item()
                )

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))

        print(
            f"epoch={epoch:02d} "
            f"train_mse={train_loss:.6f} "
            f"val_mse={val_loss:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError(
            "No model checkpoint was produced."
        )

    model.load_state_dict(best_state)

    return model


def main():
    dataset_config = config["dataset"]
    model_config = config["model"]

    data_file = dataset_config["file"]

    df = load_series(
        RAW / Path(data_file).name
    )

    windows = load_anomaly_windows(
        RAW / "combined_windows.json",
        data_file,
    )

    labels = make_anomaly_labels(
        df["timestamp"],
        windows,
    )

    splits = split_and_scale(
        df,
        labels,
    )

    sequence_length = int(
        model_config["sequence_length"]
    )

    x_train, y_train = make_windows(
        splits.train,
        sequence_length,
    )

    x_val, y_val = make_windows(
        splits.validation,
        sequence_length,
    )

    # Training data contains only normal targets.
    train_labels = labels[: len(splits.train)]

    normal_mask = (
        train_labels[sequence_length:] == 0
    )

    x_train = x_train[normal_mask]
    y_train = y_train[normal_mask]

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train),
            torch.from_numpy(y_train),
        ),
        batch_size=int(
            model_config["batch_size"]
        ),
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_val),
            torch.from_numpy(y_val),
        ),
        batch_size=int(
            model_config["batch_size"]
        ),
        shuffle=False,
    )

    model = train_model(
        TransformerForecaster(
            d_model=int(
                model_config["d_model"]
            ),
            nhead=int(
                model_config["nhead"]
            ),
            num_layers=int(
                model_config["num_layers"]
            ),
            dropout=float(
                model_config["dropout"]
            ),
        ),
        train_loader,
        val_loader,
        epochs=int(
            model_config["epochs"]
        ),
        learning_rate=float(
            model_config["learning_rate"]
        ),
        weight_decay=float(
            model_config["weight_decay"]
        ),
    )

    torch.save(
        model.state_dict(),
        ARTIFACTS / "transformer_forecaster.pt",
    )

    torch.save(
        {
            "mean": float(
                splits.scaler.mean_[0]
            ),
            "scale": float(
                splits.scaler.scale_[0]
            ),
            "sequence_length": sequence_length,
        },
        ARTIFACTS / "scaler.pt",
    )

    metadata = {
        "dataset": data_file,
        "train_points": len(splits.train),
        "validation_points": len(
            splits.validation
        ),
        "test_points": len(splits.test),
        "sequence_length": sequence_length,
        "train_end": len(splits.train) - 1,
"validation_start": len(splits.train),
"test_start": len(splits.train) + len(splits.validation),
"test_end": (
    len(splits.train)
    + len(splits.validation)
    + len(splits.test)
    - 1
),
        "d_model": int(
            model_config["d_model"]
        ),
        "nhead": int(
            model_config["nhead"]
        ),
        "num_layers": int(
            model_config["num_layers"]
        ),
        "dropout": float(
            model_config["dropout"]
        ),
        "batch_size": int(
            model_config["batch_size"]
        ),
        "epochs": int(
            model_config["epochs"]
        ),
        "learning_rate": float(
            model_config["learning_rate"]
        ),
        "weight_decay": float(
            model_config["weight_decay"]
        ),
        "device": (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    }

    (
        ARTIFACTS / "training_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    print("Model artifacts saved.")


if __name__ == "__main__":
    main()