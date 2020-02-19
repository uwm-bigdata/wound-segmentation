import numpy as np
from PIL import Image
import os


def paddingjpg(path):
    xmax = 0
    ymax = 0
    file_list = []
    for FileNameLength in range(0, 100):
        for dirName, subdirList, fileList in os.walk(path):
            for filename in fileList:
                # check file extension
                if ".jpg" in filename.lower() and len(filename) == FileNameLength:
                    file_list.append(filename)
            break
    file_list.sort()
    print(file_list)
    temp_list = []
    for filename in file_list:
        image = Image.open(path + filename)
        padded_image = Image.new("RGB", [560, 560])
        padded_image.paste(image, (0,0))
        padded_image.save(path + 'padded/' + filename)

def paddingpng(path):
    xmax = 0
    ymax = 0
    file_list = []
    for FileNameLength in range(0, 100):
        for dirName, subdirList, fileList in os.walk(path):
            for filename in fileList:
                # check file extension
                if ".png" in filename.lower() and len(filename) == FileNameLength:
                    file_list.append(filename)
            break
    file_list.sort()
    print(file_list)
    temp_list = []
    for filename in file_list:
        image = Image.open(path + filename)
        padded_image = Image.new("L", [560, 560])
        padded_image.paste(image, (0,0))
        padded_image.save(path + 'padded/' + filename)


paddingjpg('../data/train/images/')
paddingpng('../data/train/labels/')
paddingjpg('../data/test/images/')
paddingpng('../data/test/labels/')