import numpy as np
from typing import List


# Returns the intersect over unioun result of binary matrices
def single_pair_IoU(mask:np.ndarray, prediction:np.ndarray) -> float:
    # Intersections only occurs when has 1 in both matrices, so 2 will be the result in these positions after the sum:
    intersection:np.ndarray = np.where((mask + prediction) == 2, np.ones(mask.shape), np.zeros(mask.shape))
    intersection_count:int = np.count_nonzero(intersection)
    # All places that has 1 or 2 counts as union:
    union:np.ndarray = np.where((mask + prediction) >= 1, np.ones(mask.shape), np.zeros(mask.shape))
    union_count:int = np.count_nonzero(union) if np.count_nonzero(union) != 0 else 1
    # If both mask and predictions are blank, then the IoU is maximum:
    IoU = 1.0 if (mask == 0).all() and (prediction == 0).all() else intersection_count/union_count

    return IoU

# Returns the intersect over unioun result of binary matrices
def single_pair_pixel_error(mask:np.ndarray, prediction:np.ndarray) -> float:
    # The difference of pixels between the images is the class pixels that results in 1 after sum each pixel:
    difference:np.ndarray = np.where((mask + prediction) == 1, np.ones(mask.shape), np.zeros(mask.shape))
    # Get the values to do the math:
    difference_count:int = np.count_nonzero(difference)
    total_pixels:int = mask.shape[0] * mask.shape[1]

    return difference_count / total_pixels

def calculate_metric(metric_name:str, metric_callback, class_name:str, predictions_classes:List[np.ndarray], masks_classes:List[np.ndarray], results_file_path:str):
    pairs_total = len(predictions_classes)
    results:List[float] = []
    
    with open(results_file_path, 'a') as file:
        for index in range(pairs_total):
            result = metric_callback(predictions_classes[index], masks_classes[index])
            results.append(result)
            
            summarizing_result = '> Calculating ' + class_name + ' ' + metric_name + ': pair[%d] - result[%.4f]' % (index, result)
            print(summarizing_result)
            file.write(summarizing_result + '\n')
        file.write('%s of %s segmentation [%.6f]\n' % (metric_name, class_name, str(sum(results)/pairs_total)))

# Gets the IoU for each segmentation class:
def calculate_IoU(class_name:str, predictions_classes:List[np.ndarray], masks_classes:List[np.ndarray], results_file_path:str):
    calculate_metric('IoU', single_pair_IoU, class_name, predictions_classes, masks_classes, results_file_path)

# Gets the IoU for each segmentation class:
def calculate_pixel_error(class_name:str, predictions_classes:List[np.ndarray], masks_classes:List[np.ndarray], results_file_path:str):
    calculate_metric('Pixel Error', single_pair_pixel_error, class_name, predictions_classes, masks_classes, results_file_path)