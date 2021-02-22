'''
@File        : S3_nii_resample_batch_process.py
@Author      : xiaw@sibet.ac.cn
@Date        : 2021/02/12
@Description :
Deep Learning for Automatic Differential Diagnosis of Primary Central Nervous System Lymphoma and
Glioblastoma: Multi-parametric MRI based Convolutional Neural Network Model

## script for resampling nii volumes to the T2F template space.
'''

import os
import subprocess

if __name__ == '__main__':
    ori_path = 'R:\\brain_tumor\\GBM-PCNSL\\data\\PCNSL\\standard\\PCNSL-nii\\'
    exe_path = './preprocesser/Resampler/Resampler.exe'
    I_modality_to_be_transformed = '\\T1C'
    I_ref = '\\T2F'
    I_format = '.nii.gz'

    if os.path.exists(ori_path):
        dir_name_list = os.listdir(ori_path)
        for nii_dir in dir_name_list:
                I_nii_dir = ori_path + nii_dir + I_modality_to_be_transformed + I_format
                I_ref_nii_dir = ori_path + nii_dir + I_ref + I_format
                resampled_nii_dir = ori_path + nii_dir + I_modality_to_be_transformed + '_resampled' + I_format
                if os.path.exists(I_nii_dir) and not os.path.exists(resampled_nii_dir):
                    cmd = [exe_path, I_ref_nii_dir, I_nii_dir, resampled_nii_dir]
                    subprocess.call(cmd)
                    print('pid_', nii_dir, ' ', I_modality_to_be_transformed, ' transformed!')
                else:
                    print('pid_', nii_dir, "file dir not exist or already had resample.")