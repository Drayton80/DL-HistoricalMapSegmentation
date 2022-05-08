import re
from typing import List
from PIL import Image
from pathlib import Path

from utils import get_maps, get_masks


def run() -> None:
    # Concatenates the maps's sources and the maps's masks
    maps_all: List[Image.Image] = [*get_maps(), *get_masks()]
    augmented_folder = "./maps/augmented/"
    Path(augmented_folder).mkdir(parents=True, exist_ok=True)
    
    # Apply the augmentation for each map in the pair of mask and source
    for map_img in maps_all:
        # Path of the map plus its name without its file type
        map_name: str = re.sub(r'\.(png|jpg)$', '', map_img.filename).replace('maps/training/', augmented_folder)
        print('> Augmenting map: ' + map_name)
        # Getting map's size:
        map_width: int = map_img.size[0]
        map_height: int = map_img.size[1]

        images_extension = '.jpg'
        # Removing the Alpha channel:
        map_img = map_img.convert('RGB')
        # Saving the source:
        map_img.save(map_name + images_extension)
        # Half-size
        half_suffix = '-halfSize'
        map_half = map_img.resize((int(map_width/2), int(map_height/2)))
        map_half.save(map_name + half_suffix + images_extension)
        # Rotation
        rotate_suffix = '-rotate180'
        map_img.rotate(180).save(map_name + rotate_suffix + images_extension)
        map_half.rotate(180).save(map_name + half_suffix + rotate_suffix + images_extension)
        # Horizontal Flip
        flip_suffix = '-flipHorizontal'
        map_img.transpose(Image.FLIP_LEFT_RIGHT).save(map_name + flip_suffix + images_extension)
        map_half.transpose(Image.FLIP_LEFT_RIGHT).save(map_name + half_suffix + flip_suffix + images_extension)
        


