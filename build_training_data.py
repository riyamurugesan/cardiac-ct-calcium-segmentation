import numpy as np
from create_mask_from_xml import create_mask_from_xml
from extract_calcification import extract_calcium_dict, load_ct_dicom
from pathlib import Path

HOME = Path.home()
BASE_DATA_DIR = HOME / "cocacoronarycalciumandchestcts-2"
GATED_DIR = BASE_DATA_DIR / "Gated_release_final"
XML_DIR = GATED_DIR / "calcium_xml"
DICOM_DIR = GATED_DIR / "patient"
output_path = Path("training_data")
output_path.mkdir(exist_ok=True)

pt_folders = []
for f in DICOM_DIR.iterdir():
    if f.is_dir():
        pt_folders.append(f)

for pt_folder in pt_folders:
    pt_id = pt_folder.name

    # see if this patient has an xml for mask generation
    xml_file = XML_DIR/f"{pt_id}.xml"
    if xml_file.exists():
        annotations = extract_calcium_dict(str(xml_file))
    else:
        continue
    dicom_files = list(pt_folder.rglob("*.dcm"))
    if not dicom_files:
        continue

    dicom_folder = dicom_files[0].parent

    ct, spacing = load_ct_dicom(str(dicom_folder))
    # create mask
    mask = create_mask_from_xml(ct.shape,annotations)
    # make output dir
    output_dir = output_path / pt_id
    output_dir.mkdir(exist_ok=True)

    np.save(output_dir / 'ct_volume.npy',ct)
    np.save(output_dir / 'mask.npy',mask)
    np.save(output_dir / 'spacing.npy',spacing)






        
