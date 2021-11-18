from omp_agrm import *
import numpy as np
import matplotlib.pyplot as plt
import math

if __name__ == "__main__":
    test_file_set = []
    file_path = "AR"
    file_list = os.listdir(file_path)       # Get all the files in the path
    file_list.sort()                        # sort the path_file

    right_cnt = []
    wrong_cnt = []
    sum_cnt = []
    for i in range(27):
        right_cnt.append(0)
        wrong_cnt.append(0)
        sum_cnt.append(0)

    # tag = 1: whole mode
    # tag = 0: down sample mode
    tag = 0

    for test_file in file_list:
        if 8 <= int(test_file[6:8]) <= 13 or 21 <= int(test_file[6:8]) <= 26:
            test_file_set.append(test_file)

    accuracy = []
    name = []

    # test_file_name = "m-021-21.pgm"
    for test_file_name in test_file_set:
        out_file_name = "result/out-" + test_file_name[6:8]
        with open(out_file_name, 'a+') as f:
            print("Open successfully")
            f.write(test_file_name+ '\n')
            test_file_path = "AR/" + test_file_name
            gender = test_file_name[0:1]

            #################################
            #       build dictionary        #
            #       test    image           #
            #################################
            # build dictionary

            if tag == 0:
                pic_dictionary = build_dictionary_pic(gender)
                # build dictionary with error
                pic_dictionary_with_error = add_identity_matrix(pic_dictionary)
                # read test image
                b = read_test(test_file_path)
            else:
                pic_dictionary = build_dictionary_pic(gender, 1)
                # build dictionary with error
                pic_dictionary_with_error = add_identity_matrix(pic_dictionary)
                # read test image
                b = read_test(test_file_path, 1)

            #################################
            #       separate blocks         #
            #################################

            pic_dictionary_blocks = []
            pic_dictionary_with_error_blocks = []
            b_blocks = []
            block_num = 11
            for i in range(block_num):
                pic_dictionary_blocks.append(sepatate_dict_blocks(pic_dictionary, i, block_num))
                pic_dictionary_with_error_blocks.append(sepatate_dict_blocks(pic_dictionary_with_error, i, block_num))
                b_blocks.append(sepatate_dict_blocks(b, i, block_num))

            ###############################
            #         GET OMP solve       #
            #     x_hat, x, e, e_whole    #
            ###############################
            result_list = []
            for i in range(51):
                result_list.append(0)

            for i in range(block_num):
                x = omp(pic_dictionary_blocks[i], b_blocks[i])
                result1, result2 = x_select(x, pic_dictionary_blocks[i], b_blocks[i], gender)
                result_list[result1] += 1
                result_list[result2] += 1
                f.write("Classification result:" + str(result1) + str(result2) + '\n')
                # print(e)
                # x_hat: the non-zero parameters are all concentrated in one class A_i
                # x: the origin dictionary representation solve
                # e: the error which is solved from OMP algorithm

            ###############################
            #        Calculate the        #
            #        classification       #
            #           Result            #
            ###############################

            result_list = np.array(result_list)
            FCR_num = np.argsort(-result_list)[0]
            f.write("Frequent Classification Result:" + str(FCR_num) + '\n')
            if int(test_file_name[2:5]) == FCR_num:
                right_cnt[int(test_file_name[6:8])] += 1
                sum_cnt[int(test_file_name[6:8])] += 1
                f.write("RIGHT    " + str(float(right_cnt[int(test_file_name[6:8])] / sum_cnt[int(test_file_name[6:8])]))+'\n')

            else:
                wrong_cnt[int(test_file_name[6:8])] += 1
                sum_cnt[int(test_file_name[6:8])] += 1
                f.write("WRONG    "+str(float(right_cnt[int(test_file_name[6:8])] / sum_cnt[int(test_file_name[6:8])]))+'\n')
            f.write("------------------\n")
            print(test_file_name, " Finished  ")
            print(float(right_cnt[int(test_file_name[6:8])] / sum_cnt[int(test_file_name[6:8])]))
            if test_file_name[:5] == "w-050":
                accuracy.append(float(right_cnt[int(test_file_name[6:8])] / sum_cnt[int(test_file_name[6:8])]))
                name.append(test_file_name[6:8])

        f.close()

    plt.bar(name, accuracy)
    for a, b in zip(name, accuracy):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
    plt.plot(name, accuracy, color = 'r')
    plt.ylabel('Accuracy')
    plt.xlabel('Picture Type')
    plt.grid(True)
    plt.show()
