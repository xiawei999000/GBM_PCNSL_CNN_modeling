'''
@File        : S2_ADC_calculate_batch.py
@Author      : xiaw@sibet.ac.cn
@Date        : 2021/02/12
@Description :
Deep Learning for Automatic Differential Diagnosis of Primary Central Nervous System Lymphoma and
Glioblastoma: Multi-parametric MRI based Convolutional Neural Network Model

## script for computing ADC volume map
'''

import numpy as np
import SimpleITK as itk
import os
from math import log



# 计算ADC图谱
def compute_ADC(DWI_H, DWI_L, b_value):
    # compute the ADC value of the DWI image
    #:param DWI_H: DWI image with the higher b value
    #:param DWI_L: DWI image with the lower b value,
    # the compute formular is ADC_value = [-loge(DWI_H/DWI_L)]/(b_DWI_H-b_DWI_L)
    #:return: the ADC value image corresponding to the DWI image
    [h, w, z] = DWI_H.shape
    adc_img = np.zeros([h, w, z], dtype=np.float)
    for i in range(h):
        for j in range(w):
            for k in range(z):
                if DWI_H[i, j, k] > 5 and DWI_L[i, j, k] > 5:
                    temp = DWI_L[i, j, k] / DWI_H[i, j, k]
                    adc_img[i, j, k] = log(temp) / b_value
    return adc_img


# image normalization
def img_norl(img, h_val):  # , DWIH, DWIL
    img_std = img * h_val
    img_std = np.int16(img_std)
    return img_std


if __name__ == '__main__':
    img_path = 'R:\\brain_tumor\\GBM-PCNSL\\data\\PCNSL\\standard\\PCNSL-nii\\'
    pid_list = os.listdir(img_path)
    for i in pid_list:  # pid_list indicate the patients to be processed
        pid_dir_ADC = img_path + '\\' + str(i) + '\\' + 'ADC.nii.gz'
        pid_dir_DWI_L = img_path + '\\' + str(i) + '\\' + 'DWI_L.nii.gz'
        pid_dir_DWI_H = img_path + '\\' + str(i) + '\\' + 'DWI_H.nii.gz'

        if not os.path.exists(pid_dir_ADC):
            # read volume data
            DWI_L_i = itk.ReadImage(pid_dir_DWI_L)
            DWI_H_i = itk.ReadImage(pid_dir_DWI_H)

            DWI_L = itk.GetArrayFromImage(DWI_L_i)
            DWI_H = itk.GetArrayFromImage(DWI_H_i)

            b_value = 1000
            ADC_volume = compute_ADC(DWI_H, DWI_L, b_value)
            ADC_volume_std = img_norl(ADC_volume, 1000000)

            ADC = itk.GetImageFromArray(ADC_volume_std)
            ADC.CopyInformation(DWI_H_i)

            itk.WriteImage(ADC, pid_dir_ADC)
            print('patient ' + i + ' ADC volume generated')
