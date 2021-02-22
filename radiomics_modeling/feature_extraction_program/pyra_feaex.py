# author: Xia Wei, xiaw@sibet.ac.cn
# paper: Deep Learning for Automatic Differential Diagnosis of
# Primary Central Nervous System Lymphoma and
# Glioblastoma: Multi-parametric MRI based Convolutional
# Neural Network Model
# date: 02/13/2021

# for radiomics modelling

import radiomics.featureextractor as FEE
import os
import csv
import pandas as pd
import numpy as np
import SimpleITK as sitk
import cv2

para_path = './Params.yaml'
extractor = FEE.RadiomicsFeaturesExtractor(para_path)
extractor.addProvenance(provenance_on=False)
print("Extraction parameters:\n\t", extractor.settings)
print("Enabled filters:\n\t", extractor._enabledImagetypes)
print("Enabled features:\n\t", extractor._enabledFeatures)

tumor_type = 'GBM'
img_name = 'T2F'
suffix = '.nii.gz'
mask_name = 'WholeTumor'

img_path = 'R:\\brain_tumor\\' + tumor_type + '\\' + tumor_type + '-nii\\'
result_path = './rad_features_' + tumor_type + '_' + img_name + '.csv'
##
print('img dir is: ', img_path)

df = pd.DataFrame()

if os.path.exists(img_path):
    dir_name_list = os.listdir(img_path)
    dir_name_list = list(map(int, dir_name_list))
    dir_name_list.sort()
    num = 0
    for nii_dir in dir_name_list:  # nii_dir -- patient ID, and include all imgs and mask
        img_path_i = img_path + str(nii_dir) + '\\' + img_name + suffix
        mask_path_i = img_path + str(nii_dir) + '\\' + mask_name + suffix
        assert os.path.exists(img_path_i) and os.path.exists(mask_path_i)

        roi = sitk.ReadImage(mask_path_i)
        image = sitk.ReadImage(img_path_i)
        # image = sitk.Normalize(image)  # normalize: True

        result = extractor.execute(image, roi)
        # except Exception:
        # print(dirName)
        # exit(1)
        keys, values = ['id'], [str(nii_dir)]
        for k, v in result.items():
            keys.append(k)
            values.append(v)

        if num == 0:
            df = pd.DataFrame(columns=keys)
            df.loc[str(nii_dir)] = values
        else:
            df.loc[str(nii_dir)] = values

        print("Result type:", type(result))  # result is returned in a Python ordered dictionary
        print()
        print(str(nii_dir))
        print("patient " + str(nii_dir) + "--" + tumor_type + '_' + img_name + " Features Calculated")
        num = num + 1
    # df.drop(df.columns[0], axis=1, inplace=True)
    df.to_csv(result_path, index=False)
