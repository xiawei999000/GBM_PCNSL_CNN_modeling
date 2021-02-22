'''
@File        : S4_BET_volume_generate.py
@Author      : xiaw@sibet.ac.cn
@Date        : 2021/02/12
@Description :
Deep Learning for Automatic Differential Diagnosis of Primary Central Nervous System Lymphoma and
Glioblastoma: Multi-parametric MRI based Convolutional Neural Network Model

## script for brain mask volume extraction.
'''



import subprocess
import numpy as np
import SimpleITK as itk
import os

# for image registration and BET
def data_preprocess(bet2_exe_path, img_path, bet_path):
    I_ref_nii_dir = img_path + 'T2F.nii.gz'
    T1C_reg_nii_dir = img_path + 'T1C_resampled_RigidReg.nii.gz'
    DWI_H_reg_nii_dir = img_path + 'DWI_H_resampled.nii.gz'
    DWI_L_reg_nii_dir = img_path + 'DWI_L_resampled.nii.gz'
    ADC_reg_nii_dir = img_path + 'ADC_resampled.nii.gz'
    brain_mask_nii_dir = bet_path + 'brain_bet'

    cmd_bet2 = [bet2_exe_path, I_ref_nii_dir, brain_mask_nii_dir]
    subprocess.call(cmd_bet2)  # excute the registration for BET2
    print(img_path, ' brain mask generated!')

    # read brain mask and multiple with original volume
    brain_mask_i = itk.ReadImage(brain_mask_nii_dir + '_mask.nii.gz')
    brain_mask_img = itk.GetArrayFromImage(brain_mask_i)

    T1C_i = itk.ReadImage(T1C_reg_nii_dir)
    T1C_img = itk.GetArrayFromImage(T1C_i)
    T1C_img[np.where(brain_mask_img == 0)] = 0
    T1C_bet = itk.GetImageFromArray(T1C_img)
    T1C_bet.CopyInformation(T1C_i)

    T2F_i = itk.ReadImage(I_ref_nii_dir)
    T2F_img = itk.GetArrayFromImage(T2F_i)
    T2F_img[np.where(brain_mask_img == 0)] = 0
    T2F_bet = itk.GetImageFromArray(T2F_img)
    T2F_bet.CopyInformation(T2F_i)

    DWIH_i = itk.ReadImage(DWI_H_reg_nii_dir)
    DWIH_img = itk.GetArrayFromImage(DWIH_i)
    DWIH_img[np.where(brain_mask_img == 0)] = 0
    DWIH_bet = itk.GetImageFromArray(DWIH_img)
    DWIH_bet.CopyInformation(DWIH_i)

    DWIL_i = itk.ReadImage(DWI_L_reg_nii_dir)
    DWIL_img = itk.GetArrayFromImage(DWIL_i)
    DWIL_img[np.where(brain_mask_img == 0)] = 0
    DWIL_bet = itk.GetImageFromArray(DWIL_img)
    DWIL_bet.CopyInformation(DWIL_i)

    ADC_i = itk.ReadImage(ADC_reg_nii_dir)
    ADC_img = itk.GetArrayFromImage(ADC_i)
    ADC_img[np.where(brain_mask_img == 0)] = 0
    ADC_bet = itk.GetImageFromArray(ADC_img)
    ADC_bet.CopyInformation(ADC_i)

    # generate the preprocessed volume
    result_dir_T1C_bet = bet_path + "T1C_bet.nii.gz"
    result_dir_T2F_bet = bet_path + "T2F_bet.nii.gz"
    result_dir_DWIH_bet = bet_path + "DWI_H_bet.nii.gz"
    result_dir_DWIL_bet = bet_path + "DWI_L_bet.nii.gz"
    result_dir_ADC_bet = bet_path + "ADC_bet.nii.gz"

    itk.WriteImage(T1C_bet, result_dir_T1C_bet)
    itk.WriteImage(T2F_bet, result_dir_T2F_bet)
    itk.WriteImage(DWIH_bet, result_dir_DWIH_bet)
    itk.WriteImage(DWIL_bet, result_dir_DWIL_bet)
    itk.WriteImage(ADC_bet, result_dir_ADC_bet)
    return

if __name__ == '__main__':
    bet2_exe_path = './preprocesser/bet2/bet2.exe'
    nii_path = 'R:\\brain_tumor\\GBM-PCNSL\\data\\PCNSL\\standard\\PCNSL-nii'
    dir_name_list = os.listdir(nii_path)

    for pid_dir in dir_name_list:
        img_path = nii_path + '\\' + pid_dir + '\\'
        bet_path = nii_path + '-bet\\' + pid_dir + '\\'
        if not os.path.exists(bet_path):
            os.mkdir(bet_path)
        data_preprocess(bet2_exe_path, img_path, bet_path)