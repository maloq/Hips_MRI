import numpy as np
from detectron2.structures import BoxMode
from glob import glob
import os
import re
import cv2
from utils.transforms import *
    
def resize_new_box(big_box, box, align=True):     
    
    startX, startY, endX, endY = box
    startX -= big_box[0]
    endX -= big_box[0]
    startY -= big_box[1]
    endY -= big_box[1]
    
    if align:
        
        if startX<0:
            startX = 1
        if startY<0:
            startY = 1 
        if endX> big_box[2]-big_box[0]:
            endX = big_box[2]-big_box[0] - 1
        if endY> big_box[3]-big_box[1]:
            endY = big_box[3]-big_box[1] - 1
        
    resized_box = [startX, startY, endX, endY] 
    
    return resized_box 

def save_image(filename, image): 
    
    #image = np.moveaxis(image, 0, -1) 
    try:
        cv2.imwrite(filename, image)
    except:
        return False
    
    return True
    
def make_jpeg_dataset(data_list, base_path, images_already_exist, small_box=False): 
    
    index = 0    
    dataset_train = []
    num = 0
    
    for orig_element in data_list[:int(len(data_list)*0.9)]:
        
        images = orig_element['images']        
        best_slice = orig_element['best_slice']  
        
        if small_box:             
            
            if 'small_box' in orig_element.keys():
                orig_box = resize_new_box(orig_element['box'], orig_element['small_box'], align=True)
                images = square_crop(images, orig_element['box'])
            else:
                continue
        else:
            orig_box = orig_element['box']
            
        bbox = {'bbox': orig_box, 'bbox_mode': BoxMode.XYXY_ABS, 'category_id':0}     
        
        
        for i in range(-int(len(images)*0.08),int(len(images)*0.08) ):
            
            element = dict()
            current_slice = best_slice+i
            if current_slice<0:
                current_slice =0
            image = images[current_slice]
            #origin_path = orig_element['path'].split('/')[-1].split('.')[0]            
            filename = os.path.join(base_path, '{}_{}.jpg'.format(index, num)) 
            
            if not images_already_exist:
                saved = save_image(filename, image)
            element['file_name'] = filename
            element['height'] = image.shape[0]
            element['width'] = image.shape[1]
            element['image_id'] = index 
            element['annotations'] = []                   
            element['annotations'].append(bbox)            
            if saved:
                dataset_train.append(element)
                index += 1
        num+=1
            
    dataset_val = []
    
    for orig_element in data_list[int(len(data_list)*0.9):]:
        
        images = orig_element['images']        
        best_slice = orig_element['best_slice']   
        
        if small_box:             
            
            if 'small_box' in orig_element.keys():
                orig_box = resize_new_box(orig_element['box'], orig_element['small_box'], align=True)
                images = square_crop(images, orig_element['box'])
            else:
                continue
        else:
            orig_box = orig_element['box']
            
        bbox = {'bbox': orig_box, 'bbox_mode': BoxMode.XYXY_ABS, 'category_id':0}     
        
        for i in range(-int(len(images)*0.06),int(len(images)*0.06)):
            
            element = dict()
            current_slice = best_slice+i
            if current_slice<0:
                current_slice =0
            image = images[current_slice]        
            #origin_path = orig_element['path'].split('/')[-1].split('.')[0]            
            filename = os.path.join(base_path, '{}_{}.jpg'.format(index, num))   
            
            if not images_already_exist:
                saved = save_image(filename, image)
            element['file_name'] = filename
            element['height'] = image.shape[0]
            element['width'] = image.shape[1]
            element['image_id'] = index 
            element['annotations'] = []                   
            element['annotations'].append(bbox)
            if saved:
                dataset_val.append(element)
                index += 1
        num+=1
        
    return dataset_train, dataset_val