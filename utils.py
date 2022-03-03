import re
from typing import List
from PIL import Image
from os import listdir


def get_images_in_folder(folder_path:str, match_regex: str) -> List[Image.Image]:
    image_names: List[str] = []

    for file_name in listdir(folder_path):
        if re.match(match_regex, file_name):
            print('getting image:', file_name)
            image_names.append(file_name)

    return [Image.open(folder_path + image_name) for image_name in image_names]

def get_maps_source() -> List[Image.Image]:
    return get_images_in_folder('maps/pairs/', r".+-original\.(png|jpg)$")

def get_maps_source_augmented() -> List[Image.Image]:
    return get_images_in_folder('maps/pairs/', r".+-original-.+\.(png|jpg)$")
        
def get_maps_mask() -> List[Image.Image]:
    return get_images_in_folder('maps/pairs/', r".+-mask\.(png|jpg)$")

def get_maps_mask_augmented() -> List[Image.Image]:
    return get_images_in_folder('maps/pairs/', r".+-mask-.+\.(png|jpg)$")