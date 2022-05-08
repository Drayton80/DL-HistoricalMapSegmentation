from posixpath import split
import numpy as np
from typing import List
from PIL import Image
from pathlib import Path

from utils import get_masks_augmented, get_maps_augmented, downscale_image_pixels


def split_ndarray_in_tiles(image:np.ndarray, tiles_width:int=256, tiles_height:int=256) -> List[np.ndarray]:
    tiles = []
    
    for h in range(0, image.shape[0], tiles_height):
        for w in range(0, image.shape[1], tiles_width):
            tile_height_surpass_image = h + tiles_height > image.shape[0]
            tile_height_start = h if not tile_height_surpass_image else image.shape[0] - tiles_height
            tile_height_end = h + tiles_height if not tile_height_surpass_image else image.shape[0]

            tile_width_surpass_image = w + tiles_width > image.shape[1] 
            tile_width_start = w if not tile_width_surpass_image else image.shape[1] - tiles_width
            tile_width_end = w + tiles_width if not tile_width_surpass_image else image.shape[1]

            tiles.append(image[tile_height_start:tile_height_end, tile_width_start:tile_width_end])

    return tiles
            


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
    return crop_images_in_squares(get_maps_augmented(), width, height, crop_rest_redundancy)

def crop_mask_maps(width:int=256, height:int=256, crop_rest_redundancy:bool=True) -> list:
    return crop_images_in_squares(get_masks_augmented(), width, height, crop_rest_redundancy)

def preprocess_images(path_folder:str, file_preffix:str, images:list):
    for idx, image in enumerate(images):
        # scale from [0,255] to [-1,1] and save in the compress list:
        compress_list = [downscale_image_pixels(tile) for tile in split_ndarray_in_tiles(image)]
        file_name = file_preffix + "_" + str(idx)
        file_path = path_folder + file_name + ".npz"
        np.savez_compressed(file_path, compress_list)
        print('> Preprocessed: ' + file_name)

def run(path_folder:str) -> None:
    Path(path_folder).mkdir(parents=True, exist_ok=True)
    preprocess_images(path_folder, "map", get_maps_augmented())
    preprocess_images(path_folder, "mask", get_masks_augmented())
    
