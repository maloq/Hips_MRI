import pandas as pd
import numpy as np
import os
import pydicom
import re
from glob import glob
import cv2
from skimage import data, img_as_float
from skimage import exposure
from utils.transforms import *

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def get_folder_num(path):
    
    result = re.search('/(.*)/', path)
    filtered = (result.group(1))
    folder_num = [s for s in filtered.split('/') if s.isdigit()][0]

    return folder_num

def get_folders(master_folder, projection):
    
    tag_list = ['SPAIR','STIR', 'TRIM','T1', 'T2','PD', 'FS', 'DIRTY', 'SPINE', 'DARK', 'PELVIS', 'CROP']
    
    folders_list = []    
    folders = glob(master_folder + '/*/')
    for folder in folders:        
        projection_folder = os.path.join(folder, projection)
        subfolders = glob(projection_folder + '/*/')
        for scan_folder in subfolders:
            tags = []
            if (os.listdir(scan_folder)):
                # print(scan_folder)
                for tag in tag_list:
                    if tag in scan_folder.split('/')[-2]:
                        tags.append(tag)
                # if tags:
                if 0==0:
                    folder = {"path":scan_folder,  "projection": projection, "tags": tags}
                    folders_list.append(folder)              
                else:
                    print('{} no tags'.format(scan_folder))
                    
    return folders_list


    
     
def load_scan(path, verbose=False):
    if verbose:
        print(path)
    slices = []
    for s in os.listdir(path):
        if not os.path.basename(s).startswith("."):
            slices.append(pydicom.dcmread(path + '/' + s, force=True))

    try:
        slices.sort(key=lambda x: int(x.InstanceNumber))
        try:
            slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

        for s in slices:
            s.SliceThickness = slice_thickness
    except:
        print("No instance number")

    return slices

def get_images(scans, dim=(256, 256), resize=True, interpolation=cv2.INTER_LANCZOS4, verbose=False):
    # interpolation=cv2.INTER_AREA
    images = np.array([s.pixel_array for s in scans])

    if verbose:
        print(len(images))
        print(images[0].shape)

    if resize:
        uniform_images = np.zeros((len(images), dim[0], dim[1]))
    elif len(images.shape) > 2:
        return images
    else:
        print('not readed')
        return None

    for i, image in enumerate(images):
        if resize:
            image = cv2.resize(image, dim, interpolation=interpolation)
            image = np.array(image)
            image = np.float64(image)
            uniform_images[i] = image

    if resize:
        return uniform_images


def isNaN(num):
    return num != num


def make_square_box(top_left, bottom_right, prev_box=None):    
    
    width = bottom_right[0] - top_left[0]    
    h_middle = top_left[0] + int(width/2)    
    new_width = width + int(0.35*width)    
    height = top_left[1] - bottom_right[1] 
    v_middle = bottom_right[1] + int(height/2)
    
    if abs(height/width)>1:        
        new_v_middle = int((height*np.sqrt(abs(height/width)))/2) + bottom_right[1]
    else:
        new_v_middle = v_middle
        
    new_height = new_width   
   
    if prev_box:
        new_height = prev_box
        new_width = prev_box
        
    new_top_left = abs((h_middle - int(new_width/2))),abs((new_v_middle + int(new_height/2)))    
    new_bottom_right = abs((h_middle + int(new_width/2))),abs((new_v_middle - int(new_height/2)))
    
    return new_top_left, new_bottom_right, new_width
