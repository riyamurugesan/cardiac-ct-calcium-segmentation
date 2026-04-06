import argparse
from pathlib import Path

import numpy as np

from predictive_mask_creation import CalciumPredictor


def parse_args():
    parser = argparse.ArgumentParser(description="Run calcium segmentation on one CT volume.")
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to a trained model checkpoint.",
    )
    parser.add_argument(
        "--ct-path",
        type=Path,
        required=True,
        help="Path to an input ct_volume.npy file.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Path where the predicted mask .npy file will be saved.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override, for example cuda or cpu.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    predictor = CalciumPredictor(args.model_path, device=args.device)
    ct = np.load(args.ct_path)
    calcium = predictor.predict(ct)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_path, calcium)

    print(f"Input shape: {ct.shape}")
    print(f"Output shape: {calcium.shape}")
    print(f"Calcium voxels: {calcium.sum()}")
    print(f"Values: {np.unique(calcium)}")
    print(f"Saved prediction to {args.output_path}")


if __name__ == "__main__":
    main()
