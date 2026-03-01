import numpy as np
from skimage.draw import polygon


def create_mask_from_xml(shape, xml_calcification):
    mask_3d = np.zeros(shape) #dtype=unit8

    for num, annotations in xml_calcification.items():
        #make sure slice is in bounds
        if num >= shape[0]:
            continue

        for annotation in annotations:
            points = annotation['points']
            # skip if not enough points to fill
            if len(points) < 3:
                continue
            
            x = []
            y = []
            for coord in points:
                x.append(coord[0])
                y.append(coord[1])
            
            row_i, col_i = polygon(y,x,shape = (shape[1],shape[2])) 

            #set which coords are actually valid
            valid_coords = (row_i >= 0) & (row_i < shape[1]) & (col_i >= 0) & (col_i < shape[2])

            row_i = row_i[valid_coords]
            col_i = col_i[valid_coords]

            # finally, set the valid rows/cols to 1
            mask_3d[num, row_i,col_i] = 1
    return mask_3d


