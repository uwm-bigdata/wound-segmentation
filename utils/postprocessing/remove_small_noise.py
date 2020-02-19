import cv2
import numpy as np
from scipy.ndimage.measurements import label


def remove_small_areas(img, threshold, rate):
    structure = np.ones((3, 3, 3), dtype=np.int)
    labeled, ncomponents = label(img, structure)
    # print(labeled.shape, ncomponents)
    count_list = []
    # count
    for pixel_val in range(ncomponents):
        count = 0
        for y in range(labeled.shape[1]):
            for x in range(labeled.shape[0]):
                if labeled[x][y][0] == pixel_val + 1:
                    count += 1
        count_list.append(count)
    # print(count_list)

    for i in range(len(count_list)):
        # print(i)
        if sum(count_list) != 0:
            if count_list[i] / sum(count_list) < rate:
                for y in range(labeled.shape[1]):
                    for x in range(labeled.shape[0]):
                        if labeled[x][y][0] == i + 1:
                            labeled[x][y] = [0, 0, 0]
    labeled = np.where(labeled < 1, 0, 1)
    labeled *= 255
    return labeled