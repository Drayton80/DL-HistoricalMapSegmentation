import numpy as np
from logging import error
from os import listdir
from tensorflow import keras
from numpy import ndarray, array, zeros, ones, where, count_nonzero
from typing import Tuple, List
from PIL import Image
from tqdm import tqdm
from utils import force_2d_to_rgb, save_ndarray_as_image, downscale_image_pixels, upscale_image_pixels

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

# Returns the intersect over unioun result of binary matrices
def intersection_over_union(mask:ndarray, prediction:ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    # Intersections only occurs when has 1 in both matrices, so 2 will be the result in these positions after the sum:
    intersection:ndarray = where((mask + prediction) == 2, ones(mask.shape), zeros(mask.shape))
    intersection_count:int = count_nonzero(intersection)
    # All places that has 1 or 2 counts as union:
    union:ndarray = where((mask + prediction) >= 1, ones(mask.shape), zeros(mask.shape))
    union_count:int = count_nonzero(union) if count_nonzero(union) != 0 else 1
    # If both mask and predictions are blank, then the IoU is maximum:
    IoU = 1.0 if (mask == 0).all() and (prediction == 0).all() else intersection_count/union_count

    return (IoU, intersection, union)

# Matrices with all classes specified:
def classify_pairs(masks:List[ndarray], predictions:List[ndarray]) -> Tuple[List[ndarray], List[ndarray]]:
    print('> Classifying pixels')
    return ([classify_each_pixel(prediction) for prediction in predictions], [classify_each_pixel(remove_alpha_channel(mask)) for mask in masks])

# Binary matrices for each segmentation class that are relevant to apply IoU:
def segment_pairs(predictions_classes:List[np.ndarray], masks_classes:List[np.ndarray], class_label:int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    print('> Segmenting class matrices')
    return ([segment_binary_matrix(prediction, class_label) for prediction in predictions_classes], [segment_binary_matrix(mask, class_label) for mask in masks_classes])

# Gets the IoU for each segmentation class:
def validate_predictions_with_IoU(class_name:str, predictions_classes:List[np.ndarray], masks_classes:List[np.ndarray], results_file_path:str, predict_folder: str):
    pairs_total = len(predictions_classes)
    pairs_IoU:List[float] = []
    with open(results_file_path, 'a') as file:
        for index in range(pairs_total):
            (IoU, intersection, union) = intersection_over_union(predictions_classes[index], masks_classes[index])
            pairs_IoU.append(IoU)
            #save_ndarray_as_image(predict_folder, 'pair' + str(index) + '_' + class_name + '_intersection', force_2d_to_rgb(intersection, 255))
            #save_ndarray_as_image(predict_folder, 'pair' + str(index) + '_' + class_name + '_union', force_2d_to_rgb(union, 255))
            summarizing_result = '> Calculating ' + class_name + ' IoU: pair[%d] - result[%.4f]' % (index, IoU)
            file.write(summarizing_result + '\n')
            print(summarizing_result)

    IoU_mean = sum(pairs_IoU)/pairs_total
    with open(results_file_path, 'a') as file:
        file.write('IoU of road segmentation: ' + str(IoU_mean) + '\n')

# Run the validation of one given model:
def run(model_path:str, images_folder:str='./maps/preprocessed/', save_predictions:bool=True) -> None:
    predict_folder = 'maps/predictions/'
    roads_IoU_results_file = predict_folder + 'IoU Results Roads.txt'
    builds_IoU_results_file = predict_folder + 'IoU Results Buildings.txt'
    
    open(roads_IoU_results_file, 'w+').close()
    open(builds_IoU_results_file, 'w+').close()

    (masks, predictions) = predict_test_images(images_folder, predict_folder, model_path, save_predictions)
    (predictions_classes, masks_classes) = classify_pairs(masks, predictions)
        
    (predictions_road_classes, masks_road_classes) = segment_pairs(predictions_classes, masks_classes, 1)
    #[save_ndarray_as_image(predict_folder, 'pair' + str(i) + '_roads_predictions', force_2d_to_rgb(pred, 255)) for i, pred in enumerate(predictions_road_classes)]
    #[save_ndarray_as_image(predict_folder, 'pair' + str(i) + '_roads_masks', force_2d_to_rgb(pred, 255)) for i, pred in enumerate(masks_road_classes)]  
    (predictions_build_classes, masks_build_classes) = segment_pairs(predictions_classes, masks_classes, 2)
    #[save_ndarray_as_image(predict_folder, 'pair' + str(i) + '_buildings_predictions', force_2d_to_rgb(pred, 255)) for i, pred in enumerate(predictions_build_classes)]
    #[save_ndarray_as_image(predict_folder, 'pair' + str(i) + '_buildings_masks', force_2d_to_rgb(pred, 255)) for i, pred in enumerate(masks_build_classes)]

    validate_predictions_with_IoU('roads', predictions_road_classes, masks_road_classes, roads_IoU_results_file, predict_folder)
    validate_predictions_with_IoU('buildings', predictions_build_classes, masks_build_classes, builds_IoU_results_file, predict_folder)