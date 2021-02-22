'''
@File        : S6_generate_integrated_images.py
@Author      : xiaw@sibet.ac.cn
@Date        : 2021/02/12
@Description :
Deep Learning for Automatic Differential Diagnosis of Primary Central Nervous System Lymphoma and
Glioblastoma: Multi-parametric MRI based Convolutional Neural Network Model

## script for integrated slice generation.
'''

import SimpleITK as itk
import os
import cv2
import openpyxl
import numpy as np
from PIL import Image
from tensorflow.python.keras.preprocessing import image


def integrated_img_generate(img_file_r, img_file_g, img_file_b, merge_path):
    img_r = Image.open(img_file_r)
    img_g = Image.open(img_file_g)
    img_b = Image.open(img_file_b)
    x_r = image.img_to_array(img_r)
    x_g = image.img_to_array(img_g)
    x_b = image.img_to_array(img_b)
    merge_img = cv2.merge([x_r, x_g, x_b])
    merge_img = np.uint8(merge_img)
    # save the img
    cv2.imwrite(merge_path, merge_img)


if __name__ == '__main__':
    ######################
    tumor_type = 'GBM'
    ######################
    info_path = 'R:\\brain_tumor\\GBM-PCNSL\\data\\' + tumor_type + '\\standard\\' + tumor_type + '-info-in-slice.xlsx'
    slices_path = 'R:\\brain_tumor\\GBM-PCNSL\\data\\' + tumor_type + '\\standard\\' + tumor_type + '-slice-bet\\'

    # determine the patient ID and the data set label
    info = openpyxl.load_workbook(info_path)
    shenames = info.get_sheet_names()
    print(shenames)
    # data sheet
    all_sheet = info[shenames[0]]
    # number of patients
    all_num = all_sheet.max_row - 1
    # id column in sheet
    all_id = list(all_sheet.columns)[0]
    # slice start_ind in sheet
    all_start_ind = list(all_sheet.columns)[1]
    # slice end_ind in sheet
    all_end_ind = list(all_sheet.columns)[2]

    for index in range(1, 2):
        # pid_list indicate the patients to be processed
        # original img path
        i = all_id[index].value

        ######## slices containing the tumor tissues #########
        start_ind = all_start_ind[index].value
        end_ind = all_end_ind[index].value

        pid = str(i)

        for j in range(start_ind, end_ind + 1):
            T1C_slice_path = slices_path + tumor_type + '_' + pid + '\\T1C\\' \
                             + tumor_type + '_T1C_' + pid + '_' + str(j) + '.jpg'

            T2F_slice_path = slices_path + tumor_type + '_' + pid + '\\T2F\\' \
                             + tumor_type + '_T2F_' + pid + '_' + str(j) + '.jpg'

            ADC_slice_path = slices_path + tumor_type + '_' + pid + '\\ADC\\' \
                             + tumor_type + '_ADC_' + pid + '_' + str(j) + '.jpg'

            integrated_slice_path = slices_path + tumor_type + '_' + pid + '\\cross_modal\\' \
                                        + tumor_type + '_cross_modal_' + pid + '_' + str(j) + '.jpg'

            if not os.path.exists(slices_path + tumor_type + '_' + pid + '\\cross_modal\\'):
                os.makedirs(slices_path + tumor_type + '_' + pid + '\\cross_modal\\')

            integrated_img_generate(T1C_slice_path, T2F_slice_path, ADC_slice_path, integrated_slice_path)

        print('patient ' + pid + ' integrated slices generated.')
