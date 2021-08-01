import os
from pathlib import Path
from PIL import Image
from os import listdir
from os.path import isfile, join


class Preprocessor:
    def get_images_in_folder(self, folder_path:str) -> list:
        images_names = [file_name for file_name in listdir(folder_path) if isfile(join(folder_path, file_name))]
        return [Image.open(folder_path + image_name) for image_name in images_names]

    def get_original_maps(self) -> list:
        return self.get_images_in_folder('maps/original/')
          
    def get_roadline_maps(self) -> list:
        return self.get_images_in_folder('maps/road lines/')

    def crop_images_in_squares(self, image_list, width=150, height=150, crop_rest_redundancy=True) -> list:
        all_images_cropped = []
        
        for image in image_list:
            image_width, image_height = image.size 
            image_crops = []

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

            all_images_cropped.append(image_crops)
        
        return all_images_cropped

    def crop_original_maps(self, width=150, height=150, crop_rest_redundancy=True) -> list:
        return self.crop_images_in_squares(self.get_original_maps(), width, height, crop_rest_redundancy)

    def crop_roadline_maps(self, width=150, height=150, crop_rest_redundancy=True) -> list:
        return self.crop_images_in_squares(self.get_roadline_maps(), width, height, crop_rest_redundancy)
    
    def run(self):
        cropped_original_maps = self.crop_original_maps()
        cropped_roadline_maps = self.crop_roadline_maps()
            
        for map_index in range(len(cropped_original_maps)):            
            preprocessed_map_folder = 'maps/preprocessed/' + str(map_index)
            preprocessed_original_folder = preprocessed_map_folder + '/original/'
            preprocessed_roadline_folder = preprocessed_map_folder + '/roadline/'

            Path(preprocessed_map_folder).mkdir(parents=True, exist_ok=True)
            Path(preprocessed_original_folder).mkdir(parents=True, exist_ok=True)
            Path(preprocessed_roadline_folder).mkdir(parents=True, exist_ok=True)

            for crop_index in range(len(cropped_original_maps[map_index])):
                cropped_original_maps[map_index][crop_index].save(preprocessed_original_folder + str(crop_index) + '.png')
                cropped_roadline_maps[map_index][crop_index].save(preprocessed_roadline_folder + str(crop_index) + '.png')


            

Preprocessor().run()
'''
prep = Preprocessor()
cropped = prep.crop_images_in_squares(prep.get_original_maps())

for crops_of_image in cropped:
    for crop in crops_of_image:
        crop.show()
'''