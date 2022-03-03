from tensorflow import keras
import tensorflow as tf
import numpy as np
import re
from os import listdir
from matplotlib import pyplot as plt
from pathlib import Path
from numpy import ndarray


def crop_image_in_squares(image, width:int=256, height:int=256, crop_rest_redundancy:bool=True) -> list:
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
    
    return image_crops

def ndarray_single_channel_to_rgb(ndarray):
    reshaped_ndarray = np.zeros((ndarray.shape[0], ndarray.shape[1], 3))

    for row_index, _ in enumerate(reshaped_ndarray):
        for pixel_index, _ in enumerate(reshaped_ndarray[row_index]):
            for channel_index, _ in enumerate(reshaped_ndarray[row_index][pixel_index]):
                reshaped_ndarray[row_index][pixel_index][channel_index] = ndarray[row_index][pixel_index][0]
    
    return reshaped_ndarray

def save_prediction_as_images(img_name:str, source:ndarray, prediction:ndarray) -> None:
    plt.subplot(2, 1, 1)
    plt.axis('off')
    plt.imshow(source)

    plt.subplot(2, 1, 2)
    plt.axis('off')
    plt.imshow(prediction)

    img_folder:str = 'results/'

    Path(img_folder).mkdir(parents=True, exist_ok=True)
    plt.savefig(img_folder + img_name + '.png')
    plt.close()



model = keras.models.load_model('areia_unet_ep_.957_iou_.0.906_loss_.0.112.h5', compile=False)

for i, file_name in enumerate(listdir('maps')):
    print(file_name)
    image = tf.keras.preprocessing.image.load_img('maps/' + file_name)
    cropped_images = crop_image_in_squares(image, width=800, height=800)
    
    for j, preprocessed_image in enumerate(cropped_images):
        input_array = tf.keras.preprocessing.image.img_to_array(preprocessed_image)
        input_array = np.array([input_array])
        
        prediction = model.predict(input_array)
        prediction_rgb = ndarray_single_channel_to_rgb(prediction[0])

        save_prediction_as_images(str(i) + '_' + str(j), preprocessed_image, prediction_rgb)
        