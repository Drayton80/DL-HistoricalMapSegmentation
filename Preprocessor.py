import os
import re
from pathlib import Path
from typing import List
from PIL import Image
from os import listdir
from os.path import isfile, join
from numpy import asarray, delete, savez_compressed
from keras.preprocessing.image import img_to_array, load_img

from utils import get_maps_mask_augmented, get_maps_source_augmented, downscale_image_pixels


def crop_image_in_squares(image:Image.Image, width:int=256, height:int=256, crop_rest_redundancy:bool=True) -> List[Image.Image]:
    image_width, image_height = image.size 
    image_crops:List[Image.Image] = []

    y = 0
    while y + height < image_height:
        x = 0
        while x + width < image_width:
            image_crops.append(image.crop((x, y, x+width, y+height)))                    
            x = x + width
            
        if crop_rest_redundancy and image_width % width != 0:
            image_crops.append(image.crop((image_width-width, y, image_width, y+height)))

        y = y + height
    
    if crop_rest_redundancy and image_height % height != 0:
        x = 0
        while x + width < image_width:
            image_crops.append(image.crop((x, image_height-height, x+width, image_height)))
            x = x + width             
        if image_width % width != 0:
            image_crops.append(image.crop((image_width-width, image_height-height, image_width, image_height)))
    
    return image_crops

def crop_images_in_squares(image_list:list, width:int=256, height:int=256, crop_rest_redundancy:bool=True) -> list:
    all_images_cropped = []
    
    for image in image_list:
        all_images_cropped.append(crop_image_in_squares(image, width, height, crop_rest_redundancy))
    
    return all_images_cropped

def crop_original_maps(width:int=256, height:int=256, crop_rest_redundancy:bool=True) -> list:
    return crop_images_in_squares(get_maps_source_augmented(), width, height, crop_rest_redundancy)

def crop_mask_maps(width:int=256, height:int=256, crop_rest_redundancy:bool=True) -> list:
    return crop_images_in_squares(get_maps_mask_augmented(), width, height, crop_rest_redundancy)

def run(compress_in:str='') -> None:
    if compress_in != '':
        original_list = []
        mask_list = []

    cropped_original_maps = crop_original_maps()
    cropped_mask_maps = crop_mask_maps()
        
    for map_index in range(len(cropped_original_maps)):            
        preprocessed_map_folder = 'maps/preprocessed/' + str(map_index)
        preprocessed_original_folder = preprocessed_map_folder + '/original/'
        preprocessed_mask_folder = preprocessed_map_folder + '/mask/'

        Path(preprocessed_map_folder).mkdir(parents=True, exist_ok=True)
        Path(preprocessed_original_folder).mkdir(parents=True, exist_ok=True)
        Path(preprocessed_mask_folder).mkdir(parents=True, exist_ok=True)

        for crop_index in range(len(cropped_original_maps[map_index])):
            cropped_original_maps[map_index][crop_index].save(preprocessed_original_folder + str(crop_index) + '.png')
            cropped_mask_maps[map_index][crop_index].save(preprocessed_mask_folder + str(crop_index) + '.png')
            
            if compress_in != '':
                original_pixels = asarray(cropped_original_maps[map_index][crop_index])
                mask_pixels = asarray(cropped_mask_maps[map_index][crop_index])
                # scale from [0,255] to [-1,1]
                original_downscale = downscale_image_pixels(original_pixels)
                mask_downscale = downscale_image_pixels(mask_pixels)
                # Deleta a coluna alfa caso a imagem possua 4 canais (RGB e A) em vez de 3 (RGB)
                original_list.append(original_downscale if original_downscale.shape[2] <= 3 else delete(original_downscale, 3, 2))
                mask_list.append(mask_downscale if mask_downscale.shape[2] <= 3 else delete(mask_downscale, 3, 2))
    
    if compress_in != '':
        savez_compressed(compress_in, original_list, mask_list)
