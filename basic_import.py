import os
from PIL import Image
import numpy as np
import math


# im = Image.open("m-001-01.pgm")
def build_dictionary_pic(gender, tag=0,):
    file_path = "AR"
    file_list = os.listdir(file_path)       # Get all the files in the path
    file_list.sort()                        # sort the path_file
    pic_dic = []                            # establish a list first
    for pic_file in file_list:
        if (1 <= int(pic_file[6:8]) <= 7 or 14 <= int(pic_file[6:8]) <= 20) and gender == pic_file[0:1]:
            im = Image.open("AR/"+pic_file)
            if tag == 0:
                im = im.resize((30, 42), Image.ANTIALIAS)
            im = np.array(im)
            im = im.reshape(-1)
            pic_dic.append(im/np.sqrt(np.sum(im ** 2)))

    pic_dic = np.array(pic_dic)             # transform list's the format
    pic_dic = pic_dic.T                     # transpose
    return pic_dic


def find_file(index, gender):
    file_path = "AR"
    file_list = os.listdir(file_path)       # Get all the files in the path
    file_list.sort()
    pic_dic = []                            # establish a list first
    for pic_file in file_list:
        if (1 <= int(pic_file[6:8]) <= 7 or 14 <= int(pic_file[6:8]) <= 20) and gender == pic_file[0:1]:
            pic_dic.append(pic_file)
    return pic_dic[index]


def read_test(filepath, tag = 0):
    im = Image.open(filepath)
    if tag == 0:
        #im.show()
        im = im.resize((30, 42), Image.ANTIALIAS)
        # im.show()
    im = np.array(im)
    im = im.reshape(-1)
    im = im/np.sqrt(np.sum(im ** 2))
    return im


def sepatate_dict_blocks(dictionary, index, blocks):
    return dictionary[int(dictionary.shape[0]*index/blocks):int(dictionary.shape[0]*(index+1)/blocks)]


# Add the identity matrix to the dictionary to represent error
def add_identity_matrix(pic_dic):
    ident_matrix = np.identity(pic_dic.shape[0])  # number of rows
    diction_with_error = np.column_stack((pic_dic, ident_matrix))   # append the identity matrix
    return diction_with_error


#   Image Recovery Method1: Ax = y-e
def recover_image(gender, x, tag):
    # y = Ax+e
    # Method 1: Computer Ax = y-e
    pic_dict = build_dictionary_pic(gender, tag)
    print("pic_dict.size", pic_dict.shape)
    b_0 = pic_dict.dot(x)
    print("b_0.size", b_0.shape)
    if tag == 1:
        b_0 = b_0.reshape(165, -1)
    else :
        b_0 = b_0.reshape(42, -1)

    return b_0


#   Image recovery method2: y-e = Ax
def recover_image2(gender, b, e, tag):
    b_1 = b - e
    if tag == 1:
        b_1 = b_1.reshape(165, -1)
    else:
        b_1 = b_1.reshape(42, -1)
    return b_1


if __name__ == "__main__":
    test_file_name = "m-021-21.pgm"
    test_file_path = "AR/"+test_file_name
    gender = test_file_name[0:1]

    #################################
    #       build dictionary        #
    #       test    image           #
    #################################
    # build dictionary
    pic_dictionary = build_dictionary_pic(gender)
    whole_pic_dictionary = build_dictionary_pic(gender, 1)

    # build dictionary with error
    pic_dictionary_with_error = add_identity_matrix(pic_dictionary)
    whole_pic_dictionary_with_error = add_identity_matrix(whole_pic_dictionary)
    print(pic_dictionary_with_error.shape)
    # column size
    p_whole = whole_pic_dictionary_with_error.shape[0]
    p = pic_dictionary_with_error.shape[0]
    # read test image
    b = read_test(test_file_path)
    b_whole = read_test(test_file_path, 1)