import numpy as np
from detectron2.structures import BoxMode
from glob import glob
import os
import re
import cv2
 
    

    
def make_jpeg_dataset(data_list, base_path, images_already_exist ): 
    
    index = 0    
    dataset_train = []
    num = 0
    
    for orig_element in data_list[:int(len(data_list)*0.9)]:
        images = orig_element['images']        
        best_slice = orig_element['best_slice']   
        
        
        bbox = {'bbox': orig_element['box'], 'bbox_mode': BoxMode.XYXY_ABS, 'category_id':0}     
        
        
        for i in range(-int(len(images)*0.08),int(len(images)*0.08) ):
            
            element = dict()
            current_slice = best_slice+i
            if current_slice<0:
                current_slice =0
            image = images[current_slice]
            #origin_path = orig_element['path'].split('/')[-1].split('.')[0]            
            filename = os.path.join(base_path, '{}_{}.jpg'.format(index, num)) 
            
            if not images_already_exist:
                #image = np.moveaxis(image, 0, -1)
                cv2.imwrite(filename, image)
            element['file_name'] = filename
            element['height'] = image.shape[0]
            element['width'] = image.shape[1]
            element['image_id'] = index 
            element['annotations'] = []    
               
            element['annotations'].append(bbox)
            dataset_train.append(element)
            index += 1
        num+=1
            
    dataset_val = []
    
    for orig_element in data_list[int(len(data_list)*0.9):]:
        
        images = orig_element['images']        
        best_slice = orig_element['best_slice']   
        bbox = {'bbox': orig_element['box'], 'bbox_mode': BoxMode.XYXY_ABS, 'category_id':0}     
        
        for i in range(-int(len(images)*0.06),int(len(images)*0.06)):
            
            element = dict()
            current_slice = best_slice+i
            if current_slice<0:
                current_slice =0
            image = images[current_slice]        
            #origin_path = orig_element['path'].split('/')[-1].split('.')[0]            
            filename = os.path.join(base_path, '{}_{}.jpg'.format(index, num))   
            
            if not images_already_exist:
                #image = np.moveaxis(image, 0, -1)
                cv2.imwrite(filename, image)
            element['file_name'] = filename
            element['height'] = image.shape[0]
            element['width'] = image.shape[1]
            element['image_id'] = index 
            element['annotations'] = []    
               
            element['annotations'].append(bbox)
            dataset_val.append(element)
            index += 1
        num+=1
        
    return dataset_train, dataset_val