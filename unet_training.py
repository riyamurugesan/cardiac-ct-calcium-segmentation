import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from torch.utils.data import DataLoader, Dataset


class CalciumDataset(Dataset):
    def __init__(self, data_dir, pt_ids):
        self.data_dir = Path(data_dir)
        self.pt_ids = pt_ids

    def __len__(self):
        return len(self.pt_ids)

    def __getitem__(self, ind):
        pt_id = self.pt_ids[ind]
        pt_dir = self.data_dir / str(pt_id)

        ct = np.load(pt_dir / "ct_volume.npy")
        mask = np.load(pt_dir / "mask.npy")

        ct = np.clip(ct, -1000, 3000)
        ct = (ct + 1000) / 4000.0

        def pad_array(arr):
            h, l, w = arr.shape
            padding_h = (16 - h % 16) % 16
            padding_l = (16 - l % 16) % 16
            padding_w = (16 - w % 16) % 16
            padding = (
                (padding_h // 2, padding_h - padding_h // 2),
                (padding_l // 2, padding_l - padding_l // 2),
                (padding_w // 2, padding_w - padding_w // 2),
            )
            return np.pad(arr, padding, mode="constant", constant_values=0)

        ct = pad_array(ct)
        mask = pad_array(mask)
        ct = ct[np.newaxis, ...]
        mask = mask[np.newaxis, ...]

        ct = torch.from_numpy(ct).float()
        mask = torch.from_numpy(mask).float()
        return ct, mask


def parse_args():
    parser = argparse.ArgumentParser(description="Train a 3D UNet for calcium segmentation.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing per-patient training data folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where model checkpoints and plots will be saved.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Training batch size.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )
    return parser.parse_args()


def training_epoch(model, loader, optimizer, loss_func, device):
    model.train()
    epoch_loss = 0.0

    for ct, mask in loader:
        ct = ct.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        output = model(ct)
        loss = loss_func(output, mask)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)


def validate(model, loader, metric, device):
    model.eval()
    metric.reset()

    with torch.no_grad():
        for ct, mask in loader:
            ct = ct.to(device)
            mask = mask.to(device)

            output = model(ct)
            output = torch.sigmoid(output)
            output = (output > 0.5).float()

            metric(y_pred=output, y=mask)

    dice = metric.aggregate().item()
    metric.reset()
    return dice


def load_patient_ids(data_dir: Path):
    patient_ids = [f.name for f in data_dir.iterdir() if f.is_dir()]
    patient_ids = sorted(patient_ids, key=lambda x: int(x))
    if len(patient_ids) < 2:
        raise ValueError("Need at least 2 patient folders for train/validation split.")
    return patient_ids


def train_model(data_dir: Path, output_dir: Path, batch_size: int, num_epochs: int, lr: float):
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Missing training data directory: {data_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_pts = load_patient_ids(data_dir)
    train_ind = max(1, int(0.8 * len(all_pts)))
    train_ind = min(train_ind, len(all_pts) - 1)
    training_pts = all_pts[:train_ind]
    val_pts = all_pts[train_ind:]

    training_data = CalciumDataset(data_dir, training_pts)
    val_data = CalciumDataset(data_dir, val_pts)

    training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)

    loss_func = DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dice_metric = DiceMetric(include_background=True, reduction="mean")

    best_dice = 0.0
    hist = {"training_loss": [], "val_dice": []}

    for epoch in range(num_epochs):
        training_loss = training_epoch(model, training_loader, optimizer, loss_func, device)
        val_dice = validate(model, val_loader, dice_metric, device)

        hist["training_loss"].append(training_loss)
        hist["val_dice"].append(val_dice)

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), output_dir / "best_model.pth")

        print(
            f"Epoch {epoch + 1}/{num_epochs} "
            f"train_loss={training_loss:.4f} val_dice={val_dice:.4f}"
        )

    torch.save(model.state_dict(), output_dir / "final_model.pth")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(hist["training_loss"])
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Dice Loss")
    ax1.grid(True)

    ax2.plot(hist["val_dice"])
    ax2.set_title("Validation Dice Score")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
