import random
import numpy as np
from posixpath import split
from typing import List, Tuple
from PIL import Image
from pathlib import Path
from utils import downscale_pair_image_pixels, get_masks_augmented, get_maps_augmented, downscale_image_pixels


def split_ndarray_in_tiles(pair:Tuple[np.ndarray, np.ndarray], tiles_width:int=256, tiles_height:int=256) -> List[Tuple[np.ndarray, np.ndarray]]:
    pair_tiles = []
    
    for h in range(0, pair[0].shape[0], tiles_height):
        for w in range(0, pair[0].shape[1], tiles_width):
            # Tile height range:
            tile_height_surpass_image = h + tiles_height > pair[0].shape[0]
            tile_height_start = h if not tile_height_surpass_image else pair[0].shape[0] - tiles_height
            tile_height_end = h + tiles_height if not tile_height_surpass_image else pair[0].shape[0]
            # Tile width range:
            tile_width_surpass_image = w + tiles_width > pair[0].shape[1] 
            tile_width_start = w if not tile_width_surpass_image else pair[0].shape[1] - tiles_width
            tile_width_end = w + tiles_width if not tile_width_surpass_image else pair[0].shape[1]
            # Save the tile:
            map_tile = pair[0][tile_height_start:tile_height_end, tile_width_start:tile_width_end]
            mask_tile = pair[1][tile_height_start:tile_height_end, tile_width_start:tile_width_end]
            pair_tiles.append((map_tile, mask_tile))

    return pair_tiles

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

def remove_blank_tiles(pair_tiles:List[Tuple[np.ndarray, np.ndarray]], keep_proportion=0.5) -> List[Tuple[np.ndarray, np.ndarray]]:
    rgb_blank_range_min = 250
    # Remove from both if the mask tile only contains white pixels:
    blankless_pairs = list(filter(lambda pairs_tile : not np.all(pairs_tile[1] >= rgb_blank_range_min), pair_tiles))
    # Add some blank pairs if the proportion is not 0.0
    if keep_proportion > 0.0:
        blank_keep_max = int(len(blankless_pairs) * keep_proportion)
        blank_pairs = list(filter(lambda pairs_tile : not np.all(pairs_tile[1] < rgb_blank_range_min), pair_tiles))
        result_pairs = blankless_pairs + random.sample(blank_pairs, blank_keep_max)
    else:
        result_pairs = blankless_pairs
    # Unzip the blankless pairs:
    return result_pairs

def add_pairs_to_chunk(path_folder:str, pairs:List[Tuple[np.ndarray, np.ndarray]], chunk_size:int, chunk_name:str, chunk_tiles:List[Tuple[np.ndarray, np.ndarray]], chunk_index:int, chunk_tile:int, is_last_image:bool):
    new_chunk_tiles = chunk_tiles.copy()
    
    for tile_index, pair_tile in enumerate(pairs):
        print('> %s Preprocessing chunk[%d] image[%d]' % (chunk_name.capitalize(), chunk_index, chunk_tile+1))
        new_chunk_tiles.append(pair_tile)
        # Save the chunks:
        if (chunk_tile+1) % chunk_size == 0 or (is_last_image and tile_index == len(pairs)-1):
            # Save the map tiles:
            print('> Saving %s Chunk' % chunk_name.capitalize())
            file_name = str(chunk_name) + "_chunk" + str(chunk_index) + "_size" + str(len(new_chunk_tiles))
            file_path = path_folder + file_name + ".npz"
            np.savez_compressed(file_path, new_chunk_tiles)
            new_chunk_tiles = []
            # Next chunk:
            chunk_tile = 0
            chunk_index += 1
        else:
            chunk_tile += 1
    
    return new_chunk_tiles, chunk_index, chunk_tile 

def run(path_folder:str = './maps/preprocessed/', chunk_size:int=1000, test_proportion=0.1) -> None:
    Path(path_folder).mkdir(parents=True, exist_ok=True)
    pair_images = list(zip(get_maps_augmented(), get_masks_augmented()))
    images_last_index = len(pair_images) - 1
    test_chunk_tiles: List[Tuple[np.ndarray, np.ndarray]] = []
    test_chunk_index = 0
    test_chunk_tile = 0
    train_chunk_tiles: List[Tuple[np.ndarray, np.ndarray]] = []
    train_chunk_index = 0
    train_chunk_tile = 0
    for image_index, pair_image in enumerate(pair_images):
        # Split each image in chunk of tiles:
        pair_tiles = split_ndarray_in_tiles(pair_image)
        # Remove masks of blank tiles and the maps that are their pairs:
        pair_tiles = remove_blank_tiles(pair_tiles)
        # scale from [0,255] to [-1,1] and save in the compress list:
        pair_tiles = [downscale_pair_image_pixels(pair_tile) for pair_tile in pair_tiles]
        # Get info about the indexes:
        all_indexes = range(len(pair_tiles))
        is_last_image = (image_index == images_last_index)
        # Save in the test chunk:
        test_sample_max = int(len(pair_tiles) * test_proportion)
        test_indexes = random.sample(all_indexes, test_sample_max)
        test_pairs = [pair_tiles[i] for i in test_indexes]
        test_chunk_tiles, test_chunk_index, test_chunk_tile = add_pairs_to_chunk(path_folder, test_pairs, chunk_size, 'test', test_chunk_tiles, test_chunk_index, test_chunk_tile, is_last_image)
        # Save in the train chunk:
        train_indexes = tuple(filter(lambda index : index not in test_indexes, all_indexes))
        train_pairs = [pair_tiles[i] for i in train_indexes]
        train_chunk_tiles, train_chunk_index, train_chunk_tile = add_pairs_to_chunk(path_folder, train_pairs, chunk_size, 'train', train_chunk_tiles, train_chunk_index, train_chunk_tile, is_last_image)
    
