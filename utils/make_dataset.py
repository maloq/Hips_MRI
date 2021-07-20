from typing import Dict, List, Any, Union
import json
import pandas as pd
import numpy as np
import os
import pydicom
import re
from glob import glob
import cv2
from utils.transforms import *


def convert(o):
    if isinstance(o, np.generic): return o.item()
    raise TypeError


def size_bbox_by_image(orig_bbox, height, width):    
    
    startX, startY, endX, endY = orig_bbox 
    #print(orig_bbox)
    #print( height, width)

    if startX <= 0:
        startX = 0
    if startY <= 0:
        startY = 0
    if endX/width > 0.99:
        print('bbox resized')
        endX = int(endX*0.98)
    if endY/height> 0.99:
        print('bbox resized')
        endY = int(endY*0.98) 
        
    return [startX, startY, endX, endY]   

class PreprocessedDatasetFull():
    
    def __init__(self, part1_json='dataset_p1.json', part2_json='dataset_p2.json'):

        self.drop_list = []
        with open(part1_json, 'r') as read_file:
            self.data_part1_info = json.load(read_file)
        print('Part 1 length ', len(self.data_part1_info))
        with open(part2_json, 'r') as read_file:
            self.data_part2_info = json.load(read_file)
        print('Part 2 length ', len(self.data_part2_info))

        small_boxes_json = 'small_boxes.json'        
        with open(small_boxes_json, 'r') as read_file:
            self.small_boxes_info = json.load(read_file)       

    def make_in_memory_dataset(self, task, projection, researh_types, drop_types, crop=True, best_slice=True,
                               resize=True, dim=(256, 256), slice_crop_const = 0.1, small_box = True):

        #data_list = self.data_part1_info + self.data_part2_info        
        data_list = self.data_part1_info        
        data_list_images: List[Dict[str, Union[List[int], Any]]] = []

        for element in data_list:
            if element['projection'] != projection or not any(
                    [researh_type in element['researh_type'] for researh_type in researh_types]) or any(
                [drop_type in element['researh_type'] for drop_type in drop_types]):
                continue
            images = np.load(element['path'], allow_pickle=True)
            element['label'] = int(element[task])          
            
            if small_box:
                box = element['small_box']
            else:
                box = element['box']
                
            if crop:
                images = square_crop(images, box)
            if best_slice:
                images = slice_crop(images, slice_crop_const, element['best_slice'])

            images = rescale_image(images)
            images = normalize_images_v1(images)           

            if resize:
                images = resize_cv2(images, dim)
            
            #for i, img in enumerate(images):
            #    img = create_blur_mask(img)
            #    images[i] = img

            images = np.stack((images, images, images), axis=1)

            images = normalize_images_v1(images)

            #images = np.stack((images, contrast_stretch(images, n=2),equalize_clahe(images) ), axis=1)
            #images = normalize_images_v1(images)

            element['images'] = images
            data_list_images.append(element)

        print(len(data_list_images), ' dataset length')

        return data_list_images


class PreprocessedDatasetPart1():

    def __init__(self, csv_file_part1='labels/marking_arrays_v2.csv',
                 root_dir='MRI_images_2'):

        self.dataset_info_part1 = pd.read_csv(csv_file_part1, usecols=['path', 'parent path', 'patient number',
                                                                       'researh type', 'projection', 'hip side',
                                                                       'cartialge lesion', 'subchondral cysts',
                                                                       'bone marrow lesion', 'synovitis',
                                                                       'ill', 'best_slice', 'bad_research',
                                                                       'rectangle'])
        self.root_dir = root_dir

      

    def prepare(self, save_json=False):
        marking = self.dataset_info_part1
        bad_mark = '-'
        proc_marking = marking[marking['bad_research'] != bad_mark]
        data_list = []
        length = 0

        for i, row in enumerate(proc_marking.iloc):
            element = {}
            path = row['path']

            raw_box = row['rectangle']
            box = re.findall(r"\d+", raw_box)
            integer_map = map(int, box)
            box = list(integer_map)
            
            
            try:
                images = np.load(path[2:], allow_pickle=True)
            except ValueError:
                print('not loaded ', path)
            else:
                if len(images.shape) > 2:
                    element['path'] = path[2:]
                    element['best_slice'] = (row['best_slice'])
                    element['box'] = box
                    element['researh_type'] = row['researh type']
                    element['projection'] = row['projection']
                    element['cartialge lesion'] = row['cartialge lesion']
                    element['subchondral cysts'] = row['subchondral cysts']
                    element['bone marrow lesion'] = row['bone marrow lesion']
                    element['synovitis'] = row['synovitis']
                    element['ill'] = row['ill']
                    
                    data_list.append(element)
                    length += 1

        assert len(data_list) == length

        if save_json:
            with open('dataset_p1.json', 'w') as fp:
                json.dump(data_list, fp, default=convert)
        return data_list

    def make_in_memory_dataset(self, task, projection, researh_types, drop_types, crop=True, best_slice=True,
                               resize=True, dim=(256, 256), stack_images = True):
        data_list = self.prepare()
        data_list_images: List[Dict[str, Union[List[int], Any]]] = []

        for element in data_list:
            if element['projection'] != projection or not any(
                    [researh_type in element['researh_type'] for researh_type in researh_types]) or any(
                [drop_type in element['researh_type'] for drop_type in drop_types]):
                continue
            images = np.load(element['path'], allow_pickle=True)
            
            element['label'] = int(element[task])
            
            element['box'] = size_bbox_by_image(element['box'], images.shape[1], images.shape[2])
            
            if crop:
                images = square_crop(images, element['box'])
            if best_slice:
                images = slice_crop(images, 0.2, element['best_slice'])
            if resize:
                images = resize_cv2(images, dim)       
            
            images = rescale_image(images)
            
            #images = normalize_images_v2(images)
            # images = np.stack((images, contrast_stretch(images, n=2),equalize_clahe(images) ), axis=1)
            
            images = normalize_images_v1(images)
            
            images, big_box =  crop_black_bars(images)
            element['rescaled_box'] = big_box
            element['box'] = resize_new_box(big_box, element['box'])
            
            if stack_images:
                images = np.stack((images, images, images), axis=1)
            
            element['images'] = images
            data_list_images.append(element)

        print(len(data_list_images), ' dataset length')

        return data_list_images


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


def get_boxes(boxes, path):
    row = boxes[boxes['path'] == path]
    L_box = (row['l_top'], row['l_bottom'], row['l_left'], row['l_right'])
    R_box = (row['r_top'], row['r_bottom'], row['r_left'], row['r_right'])
    return L_box, R_box


def make_square_box(top_left, bottom_right, prev_box=None):
    width = bottom_right[0] - top_left[0]
    h_middle = top_left[0] + int(width / 2)
    new_width = width + int(0.35 * width)
    height = top_left[1] - bottom_right[1]
    v_middle = bottom_right[1] + int(height / 2)
    if abs(height / width) > 1:
        new_v_middle = int((height * np.sqrt(abs(height / width))) / 2) + bottom_right[1]
    else:
        new_v_middle = v_middle
    new_height = new_width
    if prev_box:
        new_height = prev_box
        new_width = prev_box
    new_top_left = abs((h_middle - int(new_width / 2))), abs((new_v_middle + int(new_height / 2)))
    new_bottom_right = abs((h_middle + int(new_width / 2))), abs((new_v_middle - int(new_height / 2)))
    return new_top_left, new_bottom_right, new_width
