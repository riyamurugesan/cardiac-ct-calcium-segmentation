import argparse
from pathlib import Path

import numpy as np

from create_mask_from_xml import create_mask_from_xml
from extract_calcification import extract_calcium_dict, load_ct_dicom


def parse_args():
    default_data_dir = Path(".")

    parser = argparse.ArgumentParser(
        description="Generate ct_volume.npy, mask.npy, and spacing.npy for each patient."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir,
        help=(
            "Dataset root directory. Expected to contain "
            "Gated_release_final/calcium_xml and Gated_release_final/patient."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training_data"),
        help="Directory where per-patient .npy files will be written.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Process at most patient IDs from 1 to size, inclusive.",
    )
    return parser.parse_args()


def build_training_data(data_dir: Path, output_dir: Path, size: int | None = None):
    gated_dir = data_dir / "Gated_release_final"
    xml_dir = gated_dir / "calcium_xml"
    dicom_dir = gated_dir / "patient"

    if not xml_dir.is_dir():
        raise FileNotFoundError(f"Missing XML directory: {xml_dir}")
    if not dicom_dir.is_dir():
        raise FileNotFoundError(f"Missing patient directory: {dicom_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    pt_folders = sorted(f for f in dicom_dir.iterdir() if f.is_dir())

    for pt_folder in pt_folders:
        pt_id = pt_folder.name
        if size is not None:
            try:
                pt_num = int(pt_id)
            except ValueError:
                continue
            if pt_num < 1 or pt_num > size:
                continue

        xml_file = xml_dir / f"{pt_id}.xml"
        if not xml_file.exists():
            continue

        annotations = extract_calcium_dict(str(xml_file))
        dicom_files = list(pt_folder.rglob("*.dcm"))
        if not dicom_files:
            continue

        dicom_folder = dicom_files[0].parent
        ct, spacing = load_ct_dicom(str(dicom_folder))
        mask = create_mask_from_xml(ct.shape, annotations)

        patient_output_dir = output_dir / pt_id
        patient_output_dir.mkdir(parents=True, exist_ok=True)

        np.save(patient_output_dir / "ct_volume.npy", ct)
        np.save(patient_output_dir / "mask.npy", mask)
        np.save(patient_output_dir / "spacing.npy", spacing)

        print(f"Saved patient {pt_id} to {patient_output_dir}")


def main():
    args = parse_args()
    build_training_data(args.data_dir, args.output_dir, args.size)


if __name__ == "__main__":
    main()
