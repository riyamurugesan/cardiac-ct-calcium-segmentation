# CalciSeg

Anil Chintapalli, Riya Murugesan and Haoze Li  
Cardiovascular Modeling and Simulation Laboratory  
University of North Carolina at Chapel Hill

Utilities for building per-patient training data from the gated coronary calcium CT dataset, training a 3D UNet, and running prediction on new CT volumes.

## Install Dependencies

```bash
pip install -r requirement.txt
```

## 1. Generate `ct_volume`, `mask`, `spacing`

```bash
python build_training_data.py \
  --data-dir <DATA_DIR> \
  --output-dir <OUTPUT_DIR>/training_data \
  --size 100
```

If `--data-dir` is omitted, the script uses the current working directory.
If `--size` is provided, the script only processes patient IDs from `1` to `size`.

### Expected Input Layout

`--data-dir` should contain:

```text
<DATA_DIR>/
  Gated_release_final/
    calcium_xml/
      0.xml
      1.xml
      ...
    patient/
      0/<dicom-series>/*.dcm
      1/<dicom-series>/*.dcm
      ...
```

### Output Layout

For each patient with both XML annotations and DICOM files, the script writes:

```text
<OUTPUT_DIR>/training_data/
  <patient_id>/
    ct_volume.npy
    mask.npy
    spacing.npy
```

Notes:

- `ct_volume.npy` is the 3D CT volume loaded from the DICOM series.
- `mask.npy` is the 3D calcium mask rasterized from the XML annotations.
- `spacing.npy` stores voxel spacing in `(z, y, x)` order.

## 2. Train The UNet

```bash
python unet_training.py \
  --data-dir <TRAINING_DATA_DIR> \
  --output-dir <MODEL_OUTPUT_DIR> \
  --batch-size 1 \
  --num-epochs 50 \
  --lr 1e-4
```

Expected training input:

```text
<TRAINING_DATA_DIR>/
  <patient_id>/
    ct_volume.npy
    mask.npy
    spacing.npy
```

Training outputs:

```text
<MODEL_OUTPUT_DIR>/
  best_model.pth
  final_model.pth
  training_curves.png
```

## 3. Run Prediction On One CT Volume

```bash
python make_prediction.py \
  --model-path <MODEL_PATH> \
  --ct-path <CT_VOLUME_PATH> \
  --output-path <PRED_MASK_PATH> \
  --device cpu
```

Prediction output:

```text
<PRED_MASK_PATH>
```

This file is a predicted binary calcium mask saved as `.npy`.
