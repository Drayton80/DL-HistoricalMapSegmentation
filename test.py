import metrics
import numpy as np
from os import listdir
from tensorflow import keras
from numpy import ndarray, zeros, ones, where
from typing import Tuple, List
from PIL import Image
from utils import save_ndarray_as_image, upscale_image_pixels

def remove_alpha_channel(image:ndarray) -> ndarray:
    return image[:, :, :3] if len(image[0][0]) == 4 else image

def classify_each_pixel(image:ndarray) -> ndarray:
    # Classify red as any pixel that has times 2 the value of blue and green:
    red_double_green = image[:, :, 0] > np.uint16(image[:, :, 1])*2
    red_double_blue = image[:, :, 0] > np.uint16(image[:, :, 2])*2
    builds_pixels = np.where(np.logical_and(red_double_green, red_double_blue), 2, 0)
    # Classify black as any pixel that has all RGB values darker than 20:
    dark_max = 20
    red_dark = image[:, :, 0] < dark_max
    blue_dark = image[:, :, 1] < dark_max
    green_dark = image[:, :, 2] < dark_max
    roads_pixels = np.where(np.logical_and(red_dark, np.logical_and(blue_dark, green_dark)), 1, 0)
    # Unite all classes
    united_classes = builds_pixels + roads_pixels
    classified_pixels = np.where(united_classes > 2, 0, united_classes)
    # Return all classes united:
    return np.uint8(classified_pixels)

def predict_test_images(images_folder:str, predict_folder:str, model_path:str, save_predictions:bool) -> Tuple[List[ndarray], List[ndarray]]:
    print('> loading model ' + model_path)
    model:keras.Model = keras.models.load_model(model_path, compile=False)
    masks_rgb:List[ndarray] = []
    predictions_rgb:List[ndarray] = []
    
    files_name = tuple(filter(lambda name : 'test' in name, listdir(images_folder)))
    chunks_file_path = [images_folder + chunk_file_name for chunk_file_name in files_name]
    
    for chunk_index, chunk_file_path in enumerate(chunks_file_path):
        for pair_index, pair in enumerate(np.load(chunk_file_path)['arr_0']):
            print('> Predicted: chunk[%d] pair[%d]' % (chunk_index+1, pair_index+1))
            # Predict returns a list of predictions, so we get only the first one:
            prediction:ndarray = model.predict(np.array([pair[0]]))[0]
            # Its necessary to convert the ndarrays from [-1,1] to range [0,255] and put it only as integers:
            prediction_rgb:np.ndarray = np.uint8(upscale_image_pixels(prediction))
            map_rgb:np.ndarray = np.uint8(upscale_image_pixels(pair[0]))
            mask_rgb:np.ndarray = np.uint8(upscale_image_pixels(pair[1]))  
            # Save the masks and predictions in arrays to post check the performance: 
            masks_rgb.append(mask_rgb)
            predictions_rgb.append(prediction_rgb)
            # Save predictions if necessary:
            if save_predictions:
                # Generates the name of each prediction:
                group_name:str = 'chunk' + str(chunk_index) + '_pair' + str(pair_index)
                # Save all the images tested:
                save_ndarray_as_image(predict_folder, group_name + '_map', map_rgb)
                save_ndarray_as_image(predict_folder, group_name + '_mask', mask_rgb)
                save_ndarray_as_image(predict_folder, group_name + '_prediction', prediction_rgb)
    
    return (masks_rgb, predictions_rgb)

def segment_binary_matrix(image:ndarray, class_value:int) -> ndarray:
    return np.uint8(where(image == class_value, ones(image.shape), zeros(image.shape)))

# Matrices with all classes specified:
def classify_pairs(masks:List[ndarray], predictions:List[ndarray]) -> Tuple[List[ndarray], List[ndarray]]:
    print('> Classifying pixels')
    return ([classify_each_pixel(prediction) for prediction in predictions], [classify_each_pixel(remove_alpha_channel(mask)) for mask in masks])

# Binary matrices for each segmentation class that are relevant to apply IoU:
def segment_pairs(predictions_classes:List[np.ndarray], masks_classes:List[np.ndarray], class_label:int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    print('> Segmenting class matrices')
    return ([segment_binary_matrix(prediction, class_label) for prediction in predictions_classes], [segment_binary_matrix(mask, class_label) for mask in masks_classes])      

# Run the validation of one given model:
def run(model_path:str, images_folder:str='./maps/preprocessed/', save_predictions:bool=True) -> None:
    predict_folder = 'maps/predictions/'

    blanks_IoU_results_file = predict_folder + 'IoU Results Blanks.txt'
    blanks_swIoU_results_file = predict_folder + 'swIoU Results Blanks.txt'
    blanks_pError_results_file = predict_folder + 'Pixel Error Results Blanks.txt'
    roads_IoU_results_file = predict_folder + 'IoU Results Roads.txt'
    roads_swIoU_results_file = predict_folder + 'swIoU Results Roads.txt'
    roads_pError_results_file = predict_folder + 'Pixel Error Results Roads.txt'
    builds_IoU_results_file = predict_folder + 'IoU Results Buildings.txt'
    builds_swIoU_results_file = predict_folder + 'swIoU Results Buildings.txt'
    builds_pError_results_file = predict_folder + 'Pixel Error Results Buildings.txt'
    
    open(blanks_IoU_results_file, 'w+').close()
    open(blanks_swIoU_results_file, 'w+').close()
    open(blanks_pError_results_file, 'w+').close()
    open(roads_IoU_results_file, 'w+').close()
    open(roads_swIoU_results_file, 'w+').close()
    open(roads_pError_results_file, 'w+').close()
    open(builds_IoU_results_file, 'w+').close()
    open(builds_swIoU_results_file, 'w+').close()
    open(builds_pError_results_file, 'w+').close()

    (masks, predictions) = predict_test_images(images_folder, predict_folder, model_path, save_predictions)
    (predictions_classes, masks_classes) = classify_pairs(masks, predictions)
    
    (predictions_blank_classes, masks_blank_classes) = segment_pairs(predictions_classes, masks_classes, 0)
    (predictions_road_classes, masks_road_classes) = segment_pairs(predictions_classes, masks_classes, 1)
    (predictions_build_classes, masks_build_classes) = segment_pairs(predictions_classes, masks_classes, 2)
    
    metrics.calculate_IoU('blanks', predictions_blank_classes, masks_blank_classes, blanks_IoU_results_file)
    metrics.calculate_IoU('roads', predictions_road_classes, masks_road_classes, roads_IoU_results_file)
    metrics.calculate_IoU('buildings', predictions_build_classes, masks_build_classes, builds_IoU_results_file)

    metrics.calculate_swIoU('blanks', predictions_blank_classes, masks_blank_classes, blanks_swIoU_results_file)
    metrics.calculate_swIoU('roads', predictions_road_classes, masks_road_classes, roads_swIoU_results_file)
    metrics.calculate_swIoU('buildings', predictions_build_classes, masks_build_classes, builds_swIoU_results_file)

    metrics.calculate_pixel_error('blanks', predictions_blank_classes, masks_blank_classes, blanks_pError_results_file)
    metrics.calculate_pixel_error('roads', predictions_road_classes, masks_road_classes, roads_pError_results_file)
    metrics.calculate_pixel_error('buildings', predictions_build_classes, masks_build_classes, builds_pError_results_file)