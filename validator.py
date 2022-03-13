from logging import error
from os import listdir
from tensorflow import keras
from numpy import ndarray, array, uint8, zeros, ones, where, count_nonzero
from typing import Tuple, List
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

def predict_test_images(model_path:str, save_predictions:bool) -> Tuple[List[ndarray], List[ndarray]]:
    print('> loading model ' + model_path)
    model:keras.Model = keras.models.load_model(model_path, compile=False)
    maps:List[Image.Image] = get_test_maps_source()
    masks:List[Image.Image] = get_test_maps_mask()
    masks_rgb:List[ndarray] = []
    predictions_rgb:List[ndarray] = []
    
    for image_index, image_source in enumerate(maps):
        cropped_maps:List[Image.Image]  = crop_image_in_squares(image_source)
        cropped_masks:List[Image.Image] = crop_image_in_squares(masks[image_index])

        for cropped_index, cropped_image_source in enumerate(tqdm(cropped_maps)):
            # Predict returns a list of predictions, so we get only the first one:
            prediction:ndarray = model.predict(preprocess_image(cropped_image_source))[0]
            # Its necessary to convert the prediction to range [0,255] and put it only as integers:
            prediction_rgb:ndarray = uint8(upscale_image_pixels(prediction))
            # Generates the name of each prediction:
            group_name:str = 'img' + str(image_index) + '_crop' + str(cropped_index)
            
            masks_rgb.append(array(cropped_masks[cropped_index]))
            predictions_rgb.append(prediction_rgb)
            
            if save_predictions:
                save_ndarray_as_image('maps/predictions/', group_name + '_map', array(cropped_maps[cropped_index]))
                save_ndarray_as_image('maps/predictions/', group_name + '_mask', array(cropped_masks[cropped_index]))
                save_ndarray_as_image('maps/predictions/', group_name + '_prediction', prediction_rgb)
    
    return (masks_rgb, predictions_rgb)

def segment_binary_matrix(image:ndarray, class_value:int) -> ndarray:
    return where(image == class_value, ones(image.shape), zeros(image.shape))

# Returns the intersect over unioun result of binary matrices
def intersection_over_union(mask:ndarray, prediction:ndarray) -> float:
    mask_prediction_sum:ndarray = mask + prediction
    # Intersections only occurs when has 1 in both matrices, so 2 will be the result in these positions after the sum:
    intersection:ndarray = where(mask_prediction_sum == 2, ones(mask_prediction_sum.shape), zeros(mask_prediction_sum.shape))
    # All places that has 1 or 2 counts as union:
    union:ndarray = where(mask_prediction_sum >= 1, ones(mask_prediction_sum.shape), zeros(mask_prediction_sum.shape))

    return count_nonzero(intersection) / count_nonzero(union)

def validate_predictions_with_IoU(masks:List[ndarray], predictions:List[ndarray]):
    if len(masks) != len(predictions):
        return error
    images_total = len(masks)
    # Matrices with all classes specified:
    predictions_classes:List[ndarray] = [classify_each_pixel(prediction) for prediction in tqdm(predictions)]
    masks_classes:List[ndarray] = [classify_each_pixel(remove_alpha_channel(mask)) for mask in tqdm(masks)]
    # Binary matrices for each segmentation class that are relevant to apply IoU:
    predictions_road_class:List[ndarray] = [segment_binary_matrix(prediction, 1) for prediction in tqdm(predictions_classes)]
    predictions_buildings_class:List[ndarray] = [segment_binary_matrix(prediction, 2) for prediction in tqdm(predictions_classes)]
    masks_road_class:List[ndarray] = [segment_binary_matrix(mask, 1) for mask in tqdm(masks_classes)]
    masks_buildings_class:List[ndarray] = [segment_binary_matrix(mask, 2) for mask in tqdm(masks_classes)]
    # Gets the IoU for each segmentation class:
    roads_IoU:List[float] = []
    buildings_IoU:List[float] = []
    for index in range(images_total):
        roads_IoU.append(intersection_over_union(predictions_road_class[index], masks_road_class[index]))
        buildings_IoU.append(intersection_over_union(predictions_buildings_class[index], masks_buildings_class[index]))
    # Gets the mean IoU of all classes with empty space (0 class) included:
    mean_IoU = keras.metrics.MeanIoU(num_classes=3)
    mean_IoU_results = []
    for index in range(images_total):
        mean_IoU.update_state(predictions_classes[index], masks_classes[index])
        mean_IoU_results.append(mean_IoU.result().numpy())

    roads_IoU_mean = sum(roads_IoU)/images_total
    buildings_IoU_mean = sum(buildings_IoU)/images_total
    all_IoU_mean = sum(mean_IoU_results)/images_total
    
    with open('maps/predictions/IoU Results.txt', 'w+') as file:
        file.write('IoU of road segmentation: ' + str(roads_IoU_mean) + '\n')
        file.write('IoU of building segmentation: ' + str(buildings_IoU_mean) + '\n')
        file.write('Mean IoU of all classes (empty spaces included): ' + str(all_IoU_mean) + '\n')