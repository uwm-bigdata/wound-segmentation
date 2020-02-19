import cv2
import numpy as np
import multiprocessing
from tqdm import tqdm
from utils.io.data import get_png_filename_list
from utils.postprocessing.hole_filling import fill_holes
from utils.postprocessing.remove_small_noise import remove_small_areas


def evaluate(threshold, file_list, label_path, post_prosecced_path):
    false_positives = 0
    false_negatives = 0
    true_positives = 0

    for img_name in tqdm(file_list):
        img = cv2.imread(pred_dir + img_name)
        _, threshed = cv2.threshold(img, threshold, 255, type=cv2.THRESH_BINARY)
        ################################################################################################################
        # call image post processing functions
        mask = np.zeros((226, 226, 3))
        filled = fill_holes(threshed, threshold,0.1)
        denoised = remove_small_areas(filled, threshold, 0.05)
        ################################################################################################################
        cv2.imwrite('whatever/filled/' + img_name, filled)
        cv2.imwrite('whatever/post_processed/' + img_name, denoised)


    for filename in tqdm(file_list):
        label = cv2.imread(label_path + filename,0)
        post_prosecced = cv2.imread(post_prosecced_path + filename,0)
        xdim = label.shape[0]
        ydim = label.shape[1]
        for x in range(xdim):
            for y in range(ydim):
                if post_prosecced[x, y] and label[x, y] > threshold:
                    true_positives += 1
                if label[x, y] > threshold > post_prosecced[x, y]:
                    false_negatives += 1
                if label[x, y] < threshold < post_prosecced[x, y]:
                    false_positives += 1

    IOU = float(true_positives) / (true_positives + false_negatives + false_positives)
    Dice = 2*float(true_positives) / (2*true_positives + false_negatives + false_positives)

    print("--------------------------------------------------------")
    print("Weight file: ",post_prosecced_path.rsplit("/")[1])
    print("--------------------------------------------------------")
    print("Threshold: ", threshold)
    print("True  pos = " + str(true_positives))
    print("False neg = " + str(false_negatives))
    print("False pos = " + str(false_positives))
    print("IOU = " + str(IOU))
    print("Dice = " + str(Dice))


# change to your own folder names
pred_dir = './whatever/'
img_filename_list = get_png_filename_list(pred_dir)
print(img_filename_list)
label_path = './data/azh_wound_care_center_dataset_patches/test/labels/'
post_path = './whatever/post_processed/'
num_threads = multiprocessing.cpu_count()
# test your own threshold
threshold = 120
evaluate(threshold, img_filename_list, label_path, post_path)
