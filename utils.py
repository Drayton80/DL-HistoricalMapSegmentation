import re
from numpy import ndarray, zeros
from typing import List
from PIL import Image
from os import listdir
from matplotlib import pyplot as plt
from pathlib import Path


def get_images_in_folder(folder_path:str, match_regex: str) -> List[Image.Image]:
    image_names: List[str] = []

    for file_name in listdir(folder_path):
        if re.match(match_regex, file_name):
            print('> Getting image:', file_name)
            image_names.append(file_name)

    return [Image.open(folder_path + image_name) for image_name in image_names]

def get_maps_source() -> List[Image.Image]:
    return get_images_in_folder('maps/training/', r".+-original\.(png|jpg)$")
        
def get_maps_mask() -> List[Image.Image]:
    return get_images_in_folder('maps/training/', r".+-mask\.(png|jpg)$")

def get_maps_source_augmented() -> List[Image.Image]:
    return get_images_in_folder('maps/augmented/', r".+-original-.+\.(png|jpg)$")

def get_maps_mask_augmented() -> List[Image.Image]:
    return get_images_in_folder('maps/augmented/', r".+-mask-.+\.(png|jpg)$")

def get_test_maps_source() -> List[Image.Image]:
    return get_images_in_folder('maps/test/', r".+-original\.(png|jpg)$")

def get_test_maps_mask() -> List[Image.Image]:
    return get_images_in_folder('maps/test/', r".+-mask\.(png|jpg)$")

def image_single_channel_to_rgb(image:ndarray) -> ndarray:
    reshaped_image:ndarray = zeros((image.shape[0], image.shape[1], 3))

    for row_index, _ in enumerate(reshaped_image):
        for pixel_index, _ in enumerate(reshaped_image[row_index]):
            for channel_index, _ in enumerate(reshaped_image[row_index][pixel_index]):
                reshaped_image[row_index][pixel_index][channel_index] = image[row_index][pixel_index][0]
    
    return reshaped_image

def save_ndarray_as_image(folder:str, image_name:str, image:ndarray) -> None:
    Path(folder).mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(folder + image_name + '.png')

# scale from [0,255] to [-1,1]
def downscale_image_pixels(image:ndarray) -> ndarray:
    return (image - 127.5) / 127.5

# scale from [-1,1] to [0,255] 
def upscale_image_pixels(image:ndarray) -> ndarray:
    return image * 127.5 + 127.5