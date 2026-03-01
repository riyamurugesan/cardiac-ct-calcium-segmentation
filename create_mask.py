import numpy as np
import pydicom as dicom
import matplotlib.pyplot as plt
import SimpleITK as sitk
from pathlib import Path
from skimage.draw import polygon
from skimage.segmentation import find_boundaries
import xml.etree.ElementTree as ET
import os
import glob as glob

HOME = Path.home()
BASE_DATA_DIR = HOME / "cocacoronarycalciumandchestcts-2"
GATED_DIR = BASE_DATA_DIR / "Gated_release_final"
XML_DIR = GATED_DIR / "calcium_xml"

def get_xml(patient):
    """Returns the XML file for a patient."""
    return XML_DIR / f"{str(patient)}.xml" 


def extract_calcium_dict(xml_file):
    """Takes an XML annotation file and extracts the pixel points of calcium regions."""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    #initialize annotations dictionary
    annotations = {}

    images_array = root.find('.//key[.="Images"]/../array')

    if images_array is None:
        return annotations
    
    all_image_dicts = images_array.findall('dict')
    
    for image_dict in all_image_dicts:
        children = list(image_dict)
        
        #current slice and roi
        slice_num = None
        rois_array = None
        for i in range(len(children)):
            current_elem = children[i]
            if current_elem.tag == 'key':
                
                if current_elem.text == 'ImageIndex':
                    slice_num = int(children[i+1].text)

                elif current_elem.text == 'ROIs':
                    rois_array = children[i+1]

        if slice_num is None:
            continue
        if rois_array is None:
            continue

        all_roi_dicts = rois_array.findall('dict')

        calcium_regions = []

        for roi_dict in all_roi_dicts:
            roi_children = list(roi_dict)

            points_count = 0
            points = []

            for i in range(len(roi_children)):
                current_elem = roi_children[i]

                if current_elem.text == 'NumberOfPoints':
                    points_count = int(roi_children[i+1].text)

                elif current_elem.text == 'Point_px':
                    points_array = roi_children[i+1]

                    all_point_str = points_array.findall('string')
                    for string in all_point_str:
                        point_text = string.text

                        point_text = point_text.strip('()')
                        coordinates = point_text.split(',')

                        if len(coordinates) == 2:
                            x = int(round(float(coordinates[0].strip())))
                            y = int(round(float(coordinates[1].strip())))

                            points.append((x,y))
            
            if points_count > 0:
                if len(points) > 0:
                    data = {'slice': slice_num, 'points': points}
                    calcium_regions.append(data)
        
        if len(calcium_regions) > 0:
            annotations[slice_num] = calcium_regions
        
    return annotations

def outline_calcium(patient_num, slice_num):
    """Outlines calcium regions in red."""
    patient_folder = GATED_DIR / "patient" / str(patient_num)
    # have to glob because the name of the folder that holds .dcm files varies
    patient_folder = list(patient_folder.glob("Pro*"))
    calc_coords = extract_calcium_dict(get_xml(patient_num))

    #from SimpleITK API Example (DicomSeriesReader/DicomSeriesReader.py)
    reader = sitk.ImageSeriesReader() 

    #takes the "Pro_Gated..." folder that has the .dcm files an
    dicom_names = reader.GetGDCMSeriesFileNames(str(patient_folder[0]))
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    volume = sitk.GetArrayFromImage(image)
    ct_slice = volume[slice_num]

    #first setting an empty array for calcium mask
    mask = np.zeros(ct_slice.shape, dtype=np.uint8) 
    if slice_num in calc_coords:
        calc_regions = calc_coords[slice_num]
        for region in calc_regions:
            points = region['points']
            if len(points) < 3: #cannot draw a polygon without 3 pts.
                continue
            x = [pt[0] for pt in points]
            y = [pt[1] for pt in points]
            rr, cc = polygon(y, x, ct_slice.shape)
            #changing mask to be "on" at calcification points
            mask[rr, cc] = 1 

    #converting image slice to support RGB and segmenting mask with red boundary
    rgb = np.stack([ct_slice]*3, axis=-1)
    boundary = find_boundaries(mask, mode='outer')
    rgb[boundary] = [255, 0, 0]
    plt.imshow(rgb, cmap = None)
    plt.title(f"Patient {patient_num} Slice {slice_num}")
    plt.show()