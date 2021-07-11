import numpy as np
import os
import cv2
from skimage import data, img_as_float, util
from skimage import exposure


def crop_images(images, box):
    startX, startY, endX, endY = box
    crop = images[:, startY:endY, startX: endX]
    return crop


def square_crop(images, box):
    startX, startY, endX, endY = box
    width = endX - startX
    height = endY - startY
    if width > height:
        height = width
        # width = int(height)
    if height > width:
        width = height
        # height = int(width)
    if len(images.shape) == 3:
        
        if list(images.shape).index(min(list(images.shape))) == 0:            
            crop = images[:, int(startY):int(startY) + int(height), int(startX):int(startX) + int(width)]
        elif list(images.shape).index(min(list(images.shape))) == 2: 
            crop = images[int(startY):int(startY) + int(height), int(startX):int(startX) + int(width), :]
        else:
            print('wrong shape {}'.format(images.shape))

            
            
    elif len(images.shape) == 2:
        crop = images[int(startY):int(startY) + int(height), int(startX):int(startX) + int(width)]
    else:
        print('wrong shape {}, len {}'.format(images.shape, len(images.shape)))
    return crop


def slice_crop(images, ofset_const, slice):
    ofset = int(len(images) * ofset_const)
    if ofset < 2:
        ofset = 2
    if len(images) < 4:
        ofset = 1
    if slice + ofset > len(images):
        r_ofset = len(images) - slice - 1
    else:
        r_ofset = ofset
    if (slice - ofset) < 0:
        l_ofset = slice - 1
    else:
        l_ofset = ofset
    return images[slice - l_ofset:slice + r_ofset, :, :]


def resize_cv2(images, dim=(256, 256), interpolation=cv2.INTER_AREA):
    
    uniform_images = np.zeros((len(images), dim[0], dim[1]))
    for i, image in enumerate(images):
        image = cv2.resize(image, dim, interpolation)
        uniform_images[i] = image

    return uniform_images


def normalize_images_v1(images):
    #n_sigma = [images.mean() - 3 * images.std(), images.mean() + 3 * images.std()]
    #images = np.clip(images, n_sigma[0], n_sigma[1])
    normalized = images / np.max(images)
    #normalized = np.clip(normalized, 0.000000001, 1)
    normalized = np.uint8(normalized * 255)
    return normalized


def normalize_images_v2(images):
    # normalized = (images - np.mean(np.mean(images, axis = 0)))/np.mean(np.std(images, axis = 0))
    normalized = images / np.max(images)
    normalized = np.clip(normalized, -1, 1)
    normalized = img_as_float(normalized)
    return normalized


def equalize_clahe(images):
    equalized_images = np.zeros_like(images)
    # equalized_images = np.uint8(equalized_images)
    images = util.img_as_uint(images)
    for i, image in enumerate(images):
        image = exposure.equalize_adapthist(image, clip_limit=0.03)
        equalized_images[i] = image
    return equalized_images


def contrast_stretch(images, n=2):
    equalized_images = np.zeros_like(images)
    for i, image in enumerate(images):
        p2, p98 = np.percentile(image, (n, 100 - n))
        img_rescale = exposure.rescale_intensity(image, in_range=(p2, p98))
        equalized_images[i] = img_rescale
    return equalized_images


def equalize_histogram(images):
    equalized_images = np.zeros_like(images)
    for i, image in enumerate(images):
        image = img_eq = exposure.equalize_hist(image)
        equalized_images[i] = image
    return equalized_images


def split_in_half(images):
    if len(images.shape) == 3:
        pack_len, height, width = images.shape
    L_images = images[:, :, :int(width / 2)]
    R_images = images[:, :, int(width / 2):]
    return L_images, R_images


def create_circular_mask(h, w, center=None, radius=None):
    if center is None:
        center = (int(w / 2), int(h / 2))
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def create_blur_mask(img, n=0.8, k=1):
    img_deformed = cv2.blur(img, (10, 10))

    h, w = img.shape[:2]
    mask = create_circular_mask(h, w, center=(w / 2, n * h / 2), radius=h * 0.5 * k)
    masked_img = img.copy()
    masked_img[~mask] = img_deformed[~mask]

    return masked_img
