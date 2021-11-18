import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


# im = Image.open("m-001-01.pgm")
def build_dictionary_pic():
    filepath = "AR"
    file_list = os.listdir(filePath)
    pic_dic = []
    for pic_file in file_list:
        im = Image.open(pic_file)
        im = np.array(im)
        im = im.reshape(-1)
        pic_dic.append(im)
    return pic_dic
