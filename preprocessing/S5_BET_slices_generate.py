'''
@File        : S5_BET_slices_generate.py
@Author      : xiaw@sibet.ac.cn
@Date        : 2021/02/12
@Description :
Deep Learning for Automatic Differential Diagnosis of Primary Central Nervous System Lymphoma and
Glioblastoma: Multi-parametric MRI based Convolutional Neural Network Model

## script for normalization, z-score and slice generation.
'''

import numpy as np
import SimpleITK as itk
import os
import cv2
import openpyxl

# normalization
def img_denoise(img):
    img_std = np.std(img)
    img_mean = np.mean(img)
    high_threshold = img_mean + img_std * 3
    low_threshold = img_mean - img_std * 3
    img_denoised = img
    img_denoised[np.where(img > high_threshold)] = high_threshold
    img_denoised[np.where(img < low_threshold)] = low_threshold
    return img_denoised

# image normalization by Z-Score
def img_norl(img):
    # z-score
    img_std = np.std(img)
    img_mean = np.mean(img)
    img_normalized = (img-img_mean)/img_std
    return img_normalized

def extract_mask_boundingBox(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    m = mask
    # Bounding box.
    horizontal_indicies = np.where(np.any(m, axis=0))[0]
    vertical_indicies = np.where(np.any(m, axis=1))[0]
    if horizontal_indicies.shape[0]:
        x1, x2 = horizontal_indicies[[0, -1]]  # 选取水平方向即列方向的第一维和倒数第一维
        y1, y2 = vertical_indicies[[0, -1]]   # 选取竖直方向的第一维和倒数第一维
        # x2 and y2 should not be part of the box. Increment by 1.
        x2 += 1
        y2 += 1
    else:
        # No mask for this instance. Might happen due to
        # resizing or cropping. Set bbox to zeros
        x1, x2, y1, y2 = 0, 0, 0, 0
    center_y = np.floor((y1+y2)/2)
    center_x = np.floor((x1+x2)/2)
    # boxes_center = np.array([center_y, center_x])
    boxes_coords = np.array([x1, x2, y1, y2])
    return boxes_coords.astype(np.int32)

def slice_generate(pid_volume_dir, slices_result_dir, brain_mask_img,
                   tumor_type, sequence_type, pid, start_ind, end_ind):
    # the slice index of T1C slices containing the tumor tissue
    slice_ind_start = start_ind - 1
    slice_ind_end = end_ind - 1

    vol_i = itk.ReadImage(pid_volume_dir)
    vol_img = itk.GetArrayFromImage(vol_i)

    # adjust window level and width using std * 3
    vol_img = img_denoise(vol_img)

    # z-score standardization
    vol_img = img_norl(vol_img)

    for j in range(slice_ind_start, slice_ind_end + 1):
        brain_mask_slice = brain_mask_img[j, :, :]
        bbox = extract_mask_boundingBox(brain_mask_slice)  # boxes_coords = np.array([x1, x2, y1, y2])
        if sum(bbox) != 0:
            vol_img_slice = vol_img[j, bbox[2]:bbox[3], bbox[0]:bbox[1]]
            vol_img_slice = cv2.convertScaleAbs(vol_img_slice, alpha=(255.0 / vol_img_slice.max()))
            cv2.imwrite(slices_result_dir + tumor_type + '_' + sequence_type + '_' + pid + '_' + str(j + 1) + '.jpg',
                        vol_img_slice)



if __name__ == '__main__':
    ######################
    tumor_type = 'GBM'
    ######################
    info_path = 'R:\\brain_tumor\\GBM-PCNSL\\data\\' + tumor_type + '\\standard\\' + tumor_type + '-info-in-slice.xlsx'

    bet_path = 'R:\\brain_tumor\\GBM-PCNSL\\data\\' + tumor_type + '\\standard\\' + tumor_type + '-nii-bet\\'

    slices_path = 'R:\\brain_tumor\\GBM-PCNSL\\data\\' + tumor_type + '\\standard\\' + tumor_type + '-slice-bet\\'
    ######### slices stroing path #########
    ## '-slice-bet\\'
    ## '-slice-all\\'

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
        ######## slices containing the tumor tissues #########
        pid = str(i)
        brain_mask_nii_dir = bet_path + pid + '\\' + 'brain_bet_mask.nii.gz'
        # read brain mask and multiple with original volume
        brain_mask_i = itk.ReadImage(brain_mask_nii_dir)
        brain_mask_img = itk.GetArrayFromImage(brain_mask_i)

        # ######### obtain all slices  ########
        # num_z, height, width = brain_mask_img.shape
        # start_ind = 1
        # end_ind = num_z
        # ######### obtain all slices  ########

        sequence_type_list = ['T1C', 'T2F', 'ADC']

        for sequence_type in sequence_type_list:
            pid_volume_dir = bet_path + pid + '\\' + sequence_type + "_bet.nii.gz"
            slices_path_p_tumor = slices_path + tumor_type + '_' + pid + '\\'
            slices_result_dir = slices_path_p_tumor + sequence_type + '\\'
            if not os.path.exists(slices_path_p_tumor):
                os.makedirs(slices_path_p_tumor)
            if not os.path.exists(slices_result_dir):
                os.makedirs(slices_result_dir)
            slice_generate(pid_volume_dir, slices_result_dir, brain_mask_img,
                           tumor_type, sequence_type, pid, start_ind, end_ind)

        print('patient '+ pid + ' bet slices generated.')
