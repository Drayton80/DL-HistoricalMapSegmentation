import re
import numpy as np
from typing import List, Tuple
from PIL import Image
from os import listdir
from matplotlib import pyplot as plt
from pathlib import Path

# Load the entire data set in numpy arrays, not used anymore due to memory consumption issues
def load_dataset(preprocessed_folder:str) -> List[np.ndarray]:
	all_file_names = listdir(preprocessed_folder)

	maps:List[np.ndarray] = []
	map_files = filter(lambda file_name : 'map_' in file_name, all_file_names)
	for map_file in map_files:
		map_tiles:np.ndarray = np.load(preprocessed_folder + map_file)['arr_0']
		maps.append(map_tiles)

	masks:List[np.ndarray] = []
	mask_files = filter(lambda file_name : 'mask_' in file_name, all_file_names)
	for mask_file in mask_files:
		map_tiles:np.ndarray = np.load(preprocessed_folder + mask_file)['arr_0']
		masks.append(map_tiles)

	return [maps, masks]

def load_dataset_map(preprocessed_folder:str, index:int) -> np.ndarray:
	return np.load(preprocessed_folder + 'map_' + str(index) + '.npz')['arr_0']

def load_dataset_mask(preprocessed_folder:str, index:int) -> np.ndarray:
	return np.load(preprocessed_folder + 'mask_' + str(index) + '.npz')['arr_0']

def load_dataset_pair(preprocessed_folder:str, index:int) -> Tuple[np.ndarray, np.ndarray]:
	return (load_dataset_map(preprocessed_folder, index), load_dataset_mask(preprocessed_folder, index))

def get_images_names_in_folder(folder_path:str, match_regex: str) -> List[str]:
    image_names: List[str] = []
    print('> Getting images in ' + folder_path)

    for file_name in listdir(folder_path):
        if re.match(match_regex, file_name):
            image_names.append(file_name)
    
    return image_names

def get_images_in_folder(folder_path:str, match_regex: str) -> List[Image.Image]:
    return [Image.open(folder_path + image_name) for image_name in get_images_names_in_folder(folder_path, match_regex)]

def get_images_ndarray_in_folder(folder_path:str, match_regex: str) -> List[np.ndarray]:
    return np.asarray([np.asarray(Image.open(folder_path + image_name)) for image_name in get_images_names_in_folder(folder_path, match_regex)])

def get_maps() -> List[Image.Image]:
    return get_images_in_folder('maps/training/', r".+-original\.(png|jpg)$")
        
def get_masks() -> List[Image.Image]:
    return get_images_in_folder('maps/training/', r".+-mask\.(png|jpg)$")

def get_maps_augmented() -> List[Image.Image]:
    return get_images_ndarray_in_folder('maps/augmented/', r".+-original-.+\.(png|jpg)$")

def get_masks_augmented() -> List[Image.Image]:
    return get_images_ndarray_in_folder('maps/augmented/', r".+-mask-.+\.(png|jpg)$")

def get_test_maps() -> List[Image.Image]:
    return get_images_in_folder('maps/test/', r".+-original\.(png|jpg)$")

def get_test_masks() -> List[Image.Image]:
    return get_images_in_folder('maps/test/', r".+-mask\.(png|jpg)$")

def image_single_channel_to_rgb(image:np.ndarray) -> np.ndarray:
    reshaped_image:np.ndarray = np.zeros((image.shape[0], image.shape[1], 3))

    for row_index, _ in enumerate(reshaped_image):
        for pixel_index, _ in enumerate(reshaped_image[row_index]):
            for channel_index, _ in enumerate(reshaped_image[row_index][pixel_index]):
                reshaped_image[row_index][pixel_index][channel_index] = image[row_index][pixel_index][0]
    
    return reshaped_image

def save_ndarray_as_image(folder:str, image_name:str, image:np.ndarray) -> None:
    Path(folder).mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(folder + image_name + '.png')

# scale from [0,255] to [-1,1]
def downscale_image_pixels(image:np.ndarray) -> np.ndarray:
    return (image - 127.5) / 127.5

# scale from [0,255] to [-1,1]
def downscale_pair_image_pixels(pair_image:Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    return (downscale_image_pixels(pair_image[0]), downscale_image_pixels(pair_image[1]))

# scale from [-1,1] to [0,255] 
def upscale_image_pixels(image:np.ndarray) -> np.ndarray:
    return image * 127.5 + 127.5