import numpy as np
import SimpleITK as sitk
from pathlib import Path
from skimage.draw import polygon
import xml.etree.ElementTree as ET
from tqdm import tqdm
import json

def FindPatientFolders(gated_dir):
    gated_path = Path(gated_dir)

    patient_ids = []
    #iterate over folders in path
    for f in gated_path.iterdir():
        #make sure folder is in directory
        if f.is_dir():
            patient_ids.append(f.name)
    
    return sorted(patient_ids)

def FindMatchingXML(patient_folder, xml_dir):
    xml_path = Path(xml_dir)


    pass

def XMLToDict(xml_file):
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

            name = 'Unknown'
            points_count = 0
            points = []

            for i in range(len(roi_children)):
                current_elem = roi_children[i]

                if current_elem.text == 'Name':
                    name = roi_children[i+1].text
                elif current_elem.text == 'NumberOfPoints':
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
                    data = {'name': name, 'points': points}
                    calcium_regions.append(data)
        
        if len(calcium_regions) > 0:
            annotations[slice_num] = calcium_regions
        
    return annotations



print(XMLToDict('0.xml'))                 
