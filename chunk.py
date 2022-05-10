import numpy as np
from typing import List, Tuple
from utils import load_dataset_map, load_dataset_mask


def boundaries(tiles_per_image:List[int], max_chunk_size:int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    boundaries_images = []
    bondaries_tiles = []
    tiles_sum = 0
    tiles_sum_accumulative = 0

    for idx, tiles_number in enumerate(tiles_per_image):
        tiles_sum += tiles_number
        tiles_sum_accumulative += tiles_number
        if tiles_sum > max_chunk_size or idx + 1 == len(tiles_per_image):
            tiles_sum = 0
            boundaries_images.append((boundaries_images[-1][1]+1, idx) if len(boundaries_images) > 0 else (0, idx))
            bondaries_tiles.append((bondaries_tiles[-1][1]+1, tiles_sum_accumulative) if len(bondaries_tiles) > 0 else (0, tiles_sum_accumulative))

    return boundaries_images, bondaries_tiles

def load_maps(images_folder:str, start:int, end:int) -> List[np.ndarray]:
	return [load_dataset_map(images_folder, index) for index in range(start, end)]

def load_masks(images_folder:str, start:int, end:int) -> List[np.ndarray]:
	return [load_dataset_mask(images_folder, index) for index in range(start, end)]

def load_pairs(images_folder:str, start:int, end:int) -> List[Tuple[np.ndarray, np.ndarray]]:
    print("> Chunk load: (" + str(start) + ", " + str(end) + ")")
    map_images:List[np.ndarray] = load_maps(images_folder, start, end)
    mask_images:List[np.ndarray] = load_masks(images_folder, start, end)
    pairs:List[Tuple[np.ndarray, np.ndarray]] = []

    for image_index in range(len(map_images)):
        for tile_index in range(len(map_images[image_index])):
            pairs.append((map_images[image_index][tile_index], mask_images[image_index][tile_index]))

    return pairs