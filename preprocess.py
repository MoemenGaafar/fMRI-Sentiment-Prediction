import os
import sys
import nibabel as nib
from nilearn import image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from nilearn import datasets, image, plotting

ROIs = [{"name": "left_amygdala", "coord": (-30, -4, -22)}, 
        {"name": "right_amygdala", "coord": (30, -4, -22)}, 
        {"name": "right_insula", "coord": (42, -2, 4)},
        {"name": "left_insula", "coord": (-42, -2, 4)},
        {"name": "periaqueductal", "coord": (0, -30, -10)},
        {"name": "left_ventral_striatum", "coord": (-10, 10, -6)},
        {"name": "right_ventral_striatum", "coord": (10, 10, -6)},
        {"name": "left_putamen", "coord": (-24, 0, 4)},
        {"name": "right_putamen", "coord": (24, 0, 4)},
        {"name": "anterior_cingulate", "coord": (0, 30, 18)},
        {"name": "ventromedial_prefronal", "coord": (0, 42, -12)},
        {"name": "ventral_tegmental", "coord": (4, -18, -14)},
        {"name": "V6", "coord": (9, -82, 36)},
        {"name": "V1", "coord": (-4, -88, -2)}, 
        {"name": "noise", "coord": (-60, -90, -44)}] # bottom left corner

smoothing = 10
cube_size = 5

def normalize_fmri(fmri, smoothing=0):
    fmri = image.smooth_img(fmri, smoothing)
    fmri = fmri.get_fdata()
#    mean_along_temporal_axis = np.mean(fmri, axis=-1)
#    std_along_temporal_axis = np.std(fmri, axis=-1)
#    mean = np.transpose(np.array([[mean_along_temporal_axis] * fmri.shape[-1]][0]), (1,2,3,0))
#    std = np.transpose(np.array([[std_along_temporal_axis] * fmri.shape[-1]][0]), (1,2,3,0))
#    return np.nan_to_num((fmri - mean) / std, nan=0);
    return fmri

def reverse_transform(coord, affine):
    # Transforms coordinates from MNI space to regular space
    x2, y2, z2 = coord
    reverse_affine = np.linalg.inv(affine)
    x1, y1, z1 = image.coord_transform(x2, y2, z2, reverse_affine)
    return (round(x1), round(y1), round(z1))

def get_roi_cube(roi_mni_coord, affine, fmri, cube_size):
    x, y, z = roi_mni_coord
    a = 0
    b = 0
    c = 0
    cube = np.zeros(shape=(cube_size,cube_size,cube_size,fmri.shape[-1]))
    for i in range(-int(cube_size/2), math.ceil(cube_size/2)):
        b = 0
        for j in range(-int(cube_size/2), math.ceil(cube_size/2)):
            c = 0
            for k in range(-int(cube_size/2), math.ceil(cube_size/2)):
                x2, y2, z2 = reverse_transform((x+i, y+j, z+k), affine)
                cube[a, b, c, :] = fmri[x2, y2, z2, :]
                c+=1
            b+=1
        a+=1

    return cube

def preprocess_data(input_folder, output_folder):
    fmri_file_list = os.listdir(input_folder)
    affine = nib.load(os.path.join(input_folder, fmri_file_list[0])).affine

    normalized_folder = os.path.join(output_folder, "normalized")
    rois_folder = os.path.join(output_folder, "ROIs")

    if not os.path.exists(normalized_folder) and not os.path.exists(rois_folder):
        os.mkdir(normalized_folder)
        for fmri_file in fmri_file_list:
            normalized_location = os.path.join(normalized_folder, fmri_file[:fmri_file.find("space")]+".npy")
            fmri = nib.load(os.path.join(input_folder, fmri_file))
            normalized_fmri = normalize_fmri(fmri, smoothing)
            np.save(normalized_location, normalized_fmri)
    
    if not os.path.exists(rois_folder):
        os.mkdir(rois_folder)
    
    for roi in ROIs:
        roi_folder = os.path.join(rois_folder, roi["name"])
        if not os.path.exists(roi_folder):
            os.mkdir(roi_folder)
        
        if len(os.listdir(roi_folder)) != len(fmri_file_list):
            for fmri_file in os.listdir(normalized_folder):
                normalized_fmri = np.load(os.path.join(normalized_folder, fmri_file))
                cube = get_roi_cube(roi["coord"], affine, normalized_fmri, cube_size)
                output_file = os.path.join(roi_folder, fmri_file)
                np.save(output_file, cube)
    
    if os.path.exists(normalized_folder):
        for file_name in os.listdir(normalized_folder):
            os.remove(os.path.join(normalized_folder, file_name))
        os.rmdir(normalized_folder)



if __name__ == "__main__":
    lang = sys.argv[1]
    processed_folder = "/media/moemen/Stuff/project/data/processed"
    data_location = "/media/moemen/Stuff/project/data/ds003643-download/derivatives"
    subject_list = [folder for folder in os.listdir(data_location) if lang in folder]

    for current_subject in subject_list:
        print("Now processing subject:", current_subject)
        subject_data_folder = os.path.join(data_location, current_subject, 'func')
        subject_processed_folder = os.path.join(processed_folder, 'smoothing_10', current_subject)
        if not os.path.exists(subject_processed_folder):
            os.mkdir(subject_processed_folder)
        
        preprocess_data(subject_data_folder, subject_processed_folder)

