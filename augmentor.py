import re
from typing import List
from PIL import Image
from pathlib import Path

from utils import get_maps_source, get_maps_mask


def run() -> None:
    # Concatenates the maps's sources and the maps's masks
    maps_all: List[Image.Image] = [*get_maps_source(), *get_maps_mask()]
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
        # Resize
        map_img.resize((int(map_width/2), int(map_height/2))).save(map_name + '-halfSize.png')
        # Rotation
        #map_img.rotate(90).save(map_name + '-rotate90.png')
        map_img.rotate(180).save(map_name + '-rotate180.png')
        map_img.resize((int(map_width/2), int(map_height/2))).rotate(180).save(map_name + '-halfSize-rotate180.png')
        #map_img.rotate(270).save(map_name + '-rotate270.png')
        # Horizontal Flip
        map_img.transpose(Image.FLIP_LEFT_RIGHT).save(map_name + '-flipHorizontal.png')
        map_img.resize((int(map_width/2), int(map_height/2))).transpose(Image.FLIP_LEFT_RIGHT).save(map_name + '-halfSize-flipHorizontal.png')
        


