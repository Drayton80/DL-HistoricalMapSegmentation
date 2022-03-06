from os import listdir
from tensorflow import keras
from numpy import ndarray, array, uint8, zeros
from typing import Union, List
from PIL import Image
from tqdm import tqdm

from preprocessor import crop_image_in_squares
from utils import save_ndarray_as_image, get_test_maps_mask, get_test_maps_source, downscale_image_pixels, upscale_image_pixels

def remove_alpha_channel(image:ndarray) -> ndarray:
    return image[:, :, :3] if len(image[0][0]) == 4 else image

def preprocess_image(image:Image.Image) -> ndarray:   
    # Adding image's array inside an list, because the model accepts predict a list of images at once
    return array([remove_alpha_channel(downscale_image_pixels(array(image)))])

def classify_each_pixel(image:ndarray) -> ndarray:
    image_pixels_as_classes:ndarray = zeros((image.shape[0], image.shape[1], 1))

    for line_index, line in enumerate(image):
        for pixel_index, pixel in enumerate(line):
            r, g, b = pixel
            
            if r > g*2 and r > b*2: # Defines the houses (red) class:
                image_pixels_as_classes[line_index][pixel_index] = 2
            elif r < 128 and g < 128 and b < 128: # Defines the roads (black) class:
                image_pixels_as_classes[line_index][pixel_index] = 1
            else: # Defines the empty space (white) class:
                image_pixels_as_classes[line_index][pixel_index] = 0
    
    return image_pixels_as_classes


def predict_test_images(model_path:str) -> None:
    print('> loading model ' + model_path)
    model:keras.Model = keras.models.load_model(model_path, compile=False)
    images_source:List[Image.Image] = get_test_maps_source()
    images_mask:List[Image.Image] = get_test_maps_mask()
    predictions_rgb:List[ndarray] = []
    masks_rgb:List[ndarray] = []
    
    for image_index, image_source in enumerate(images_source):
        cropped_images:List[Image.Image] = crop_image_in_squares(image_source)
        cropped_masks:List[Image.Image]  = crop_image_in_squares(images_mask[image_index])

        for cropped_index, cropped_image_source in enumerate(tqdm(cropped_images)):
            # Predict returns a list of predictions, so we get only the first one:
            prediction:ndarray = model.predict(preprocess_image(cropped_image_source))[0]
            # Its necessary to convert the prediction to range [0,255] and put it only as integers:
            prediction_rgb:ndarray = uint8(upscale_image_pixels(prediction))
            # Generates the name of each prediction:
            prediction_name:str = 'img' + str(image_index) + '_crop' + str(cropped_index)
            
            predictions_rgb.append(prediction_rgb)
            masks_rgb.append(array(cropped_masks[cropped_index]))

            save_ndarray_as_image('maps/predictions/', prediction_name, prediction_rgb)

    predictions_classes:List[ndarray] = [classify_each_pixel(prediction) for prediction in tqdm(predictions_rgb)]
    masks_classes:List[ndarray] = [classify_each_pixel(remove_alpha_channel(mask)) for mask in tqdm(masks_rgb)]
    
    mean_IoU = keras.metrics.MeanIoU(num_classes=3)
    mean_IoU_results = []

    for index in range(len(predictions_classes)):
        mean_IoU.update_state(predictions_classes[index], masks_classes[index])
        mean_IoU_results.append(mean_IoU.result().numpy())

    [print ('IoU of index ' + str(index) + ' resulted in ' + str(result)) for index, result in enumerate(mean_IoU_results)]
    print('IoU total mean: ' + str(sum(mean_IoU_results)/len(mean_IoU_results)))