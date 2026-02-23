import pydicom as dicom
import matplotlib.pyplot as plt
import os
import xml.etree.ElementTree as ET

def view_dicom(patient, img_idx):
    """Opens and shows a single DICOM file."""
    file_path = f'/Users/riyamurugesan/Desktop/cocacoronarycalciumandchestcts-2/Gated_release_final/patient/{patient}/Pro_Gated_CS_3.0_I30f_3_70%'
    all_files = os.listdir(file_path)
    all_files = sorted(all_files)
    img_string = f"{all_files[img_idx]}"

    updated_path = f'/Users/riyamurugesan/Desktop/cocacoronarycalciumandchestcts-2/Gated_release_final/patient/{patient}/Pro_Gated_CS_3.0_I30f_3_70%/{img_string}'

    ds = dicom.dcmread(updated_path)
    image_array = ds.pixel_array
    
    plt.imshow(image_array, cmap='gray')
    plt.title(f"DICOM Image for Patient No. {patient}")
    plt.show()

def calc_coords(patient):
    # have to add an initial for loop that prints out and stores the image number of the patient (where the calcification is found).
    """Extracts coordinates of calcification for a patient"""
    tree = ET.parse(f'/Users/riyamurugesan/Desktop/cocacoronarycalciumandchestcts-2/Gated_release_final/calcium_xml/{patient}.xml')
    root = tree.getroot()
    for array in root.iter('array'):
        points = array.findall('string')
        for point in points:
            if len(point.text) < 30:
                print(point.text)

view_dicom(15,34)